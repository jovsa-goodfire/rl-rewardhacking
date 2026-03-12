"""Pluggable feature scoring methods for SAE feature analysis.

Each scorer takes an SAEDataset and returns a ranked list of ScoredFeature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from goodfire_core.interventions.utils import select_features_by_gradient
from goodfire_core.saes.batch_topk import BatchTopKSAE
from sklearn.metrics import roc_auc_score

from src.rlookout.data import SAEDataset


@dataclass
class ScoredFeature:
    feature_id: int
    score: float
    label: str = ""


@runtime_checkable
class FeatureScorer(Protocol):
    name: str

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]: ...


class PearsonScorer:
    """Per-feature Pearson correlation with RH label."""

    name = "pearson"

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]:
        results = []
        for i in range(dataset.features.shape[1]):
            col = dataset.features[:, i]
            if col.std() < 1e-8:
                continue
            corr = float(np.corrcoef(col, dataset.labels)[0, 1])
            if not np.isnan(corr):
                results.append(ScoredFeature(feature_id=i, score=corr))
        results.sort(key=lambda x: abs(x.score), reverse=True)
        return results[: self.top_k]


class DiffOfMeansScorer:
    """Cosine sim of mean(RH) - mean(non-RH) onto SAE decoder directions."""

    name = "diff_of_means"

    def __init__(self, sae: BatchTopKSAE, top_k: int = 20):
        self.sae = sae
        self.top_k = top_k

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]:
        rh_mask = dataset.labels.astype(bool)
        acts = torch.from_numpy(dataset.raw_activations).float()
        rh_mean = acts[rh_mask].mean(dim=0)
        non_rh_mean = acts[~rh_mask].mean(dim=0)
        diff = (rh_mean - non_rh_mean).unsqueeze(0).unsqueeze(0)  # (1,1,d_model)

        device = next(self.sae.parameters()).device
        feat_ids, scores = select_features_by_gradient(
            sae=self.sae, gradients=diff.to(device), k=self.top_k, use_cosine=True
        )
        return [
            ScoredFeature(feature_id=fid, score=float(s))
            for fid, s in zip(feat_ids, scores.tolist())
        ]


class MeanActivationScorer:
    """Per-class mean SAE activation difference."""

    name = "mean_activation"

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]:
        rh_mask = dataset.labels.astype(bool)
        rh_mean = dataset.features[rh_mask].mean(axis=0)
        non_rh_mean = dataset.features[~rh_mask].mean(axis=0)
        diff = rh_mean - non_rh_mean

        top_idx = np.argsort(np.abs(diff))[::-1][: self.top_k]
        return [
            ScoredFeature(feature_id=int(i), score=float(diff[i])) for i in top_idx
        ]


class LinearProbeScorer:
    """Train a linear probe on SAE features, rank by abs(weight)."""

    name = "linear_probe"

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.probe_result = None  # set after scoring

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]:
        from src.rlookout.probe_trainer import train_linear_probe
        from src.rlookout.config import ProbeConfig

        train_data, val_data = dataset.train_test_split(val_fraction=0.2)
        self.probe_result = train_linear_probe(
            train_data=train_data, val_data=val_data, config=ProbeConfig()
        )
        return self.probe_result.feature_importances[: self.top_k]


# ---------------------------------------------------------------------------
# Ensemble + validation utilities
# ---------------------------------------------------------------------------

class GradientAlignedScorer:
    """Rank features by gradient alignment across benchmarks, wrapping mi.gradient_aligned_features."""

    name = "gradient_aligned"

    def __init__(self, sae: BatchTopKSAE, all_datasets: dict[str, "SAEDataset"], labels_map: dict[int, str] | None = None, top_k: int = 20):
        self.sae = sae
        self.all_datasets = all_datasets
        self.labels_map = labels_map or {}
        self.top_k = top_k
        self.mi_result = None

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]:
        from src.rlookout.mi import gradient_aligned_features

        result = gradient_aligned_features(
            sae=self.sae, datasets=self.all_datasets, labels_map=self.labels_map, k=self.top_k
        )
        self.mi_result = result
        return [
            ScoredFeature(
                feature_id=c.feature_id,
                score=c.score,
                label=c.label or "",
            )
            for c in result.candidates
        ]


class ContrastiveScorer:
    """Rank features by contrastive cross-benchmark direction, wrapping mi.contrastive_cross_benchmark."""

    name = "contrastive"

    def __init__(self, sae: BatchTopKSAE, all_datasets: dict[str, "SAEDataset"], labels_map: dict[int, str] | None = None, top_k: int = 20):
        self.sae = sae
        self.all_datasets = all_datasets
        self.labels_map = labels_map or {}
        self.top_k = top_k
        self.mi_result = None

    def score(self, dataset: SAEDataset) -> list[ScoredFeature]:
        from src.rlookout.mi import contrastive_cross_benchmark

        result = contrastive_cross_benchmark(
            sae=self.sae, datasets=self.all_datasets, labels_map=self.labels_map, k=self.top_k
        )
        self.mi_result = result
        return [
            ScoredFeature(
                feature_id=c.feature_id,
                score=c.score,
                label=c.label or "",
            )
            for c in result.candidates
        ]


SCORER_REGISTRY: dict[str, type] = {
    "pearson": PearsonScorer,
    "diff_of_means": DiffOfMeansScorer,
    "mean_activation": MeanActivationScorer,
    "linear_probe": LinearProbeScorer,
    "gradient_aligned": GradientAlignedScorer,
    "contrastive": ContrastiveScorer,
}


def build_ensemble(
    scored_by_method: dict[str, list[ScoredFeature]],
    top_k: int = 20,
    min_methods: int = 2,
) -> list[int]:
    """Return feature IDs appearing in top_k of >= min_methods methods."""
    from collections import Counter

    counts: Counter[int] = Counter()
    for features in scored_by_method.values():
        for f in features[:top_k]:
            counts[f.feature_id] += 1
    return sorted(fid for fid, cnt in counts.items() if cnt >= min_methods)


def auroc_for_features(
    dataset: SAEDataset, feature_ids: list[int]
) -> float:
    """Compute AUROC using sum of feature activations as score.

    feature_ids may be column indices (from Pearson/MeanActivation/LinearProbe)
    or original SAE feature IDs (from DiffOfMeans). We detect which by checking
    if any ID exceeds the number of columns, and map accordingly.
    """
    if not feature_ids:
        return 0.5
    n_cols = dataset.features.shape[1]
    # If any feature_id >= n_cols, assume they're original IDs needing mapping
    if any(fid >= n_cols for fid in feature_ids):
        col_ids = []
        for fid in feature_ids:
            col = dataset.column_for_feature(fid)
            if col is not None:
                col_ids.append(col)
        if not col_ids:
            return 0.5
        feature_ids = col_ids
    scores = dataset.features[:, feature_ids].sum(axis=1)
    if np.std(scores) < 1e-8:
        return 0.5
    return float(roc_auc_score(dataset.labels, scores))


def validate_semantic_candidates(
    dataset: SAEDataset, feature_ids: list[int]
) -> dict[int, dict]:
    """Check whether specific features fire more on RH vs non-RH responses."""
    rh_mask = dataset.labels.astype(bool)
    results = {}
    for fid in feature_ids:
        # Map original feature ID to column index if variance-filtered
        col = dataset.column_for_feature(fid)
        if col is None:
            continue  # feature not in filtered set
        rh_acts = dataset.features[rh_mask, col]
        non_rh_acts = dataset.features[~rh_mask, col]
        rh_mean = float(rh_acts.mean())
        non_rh_mean = float(non_rh_acts.mean())
        # Cohen's d effect size
        pooled_std = float(np.sqrt((rh_acts.var() + non_rh_acts.var()) / 2))
        effect_size = (rh_mean - non_rh_mean) / pooled_std if pooled_std > 1e-8 else 0.0
        results[fid] = {
            "rh_mean_act": rh_mean,
            "non_rh_mean_act": non_rh_mean,
            "rh_firing_rate": float((rh_acts > 0).mean()),
            "non_rh_firing_rate": float((non_rh_acts > 0).mean()),
            "effect_size": effect_size,
        }
    return results
