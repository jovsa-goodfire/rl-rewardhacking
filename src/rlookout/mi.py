"""MI technique compositions using goodfire-core primitives.

Each function composes goodfire-core's MI tools (select_features_by_gradient,
refine_features_with_llm, FeatureLabeler) for cross-benchmark RH detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from goodfire_core.interventions.feature_selection import (
    FeatureCandidate,
    refine_features_with_llm,
)
from goodfire_core.interventions.utils import select_features_by_gradient
from goodfire_core.saes.batch_topk import BatchTopKSAE

from src.rlookout.data import SAEDataset


@dataclass
class MIResult:
    """Result from an MI technique."""

    technique: str
    candidates: list[FeatureCandidate]
    metadata: dict = field(default_factory=dict)


def gradient_aligned_features(
    sae: BatchTopKSAE,
    datasets: dict[str, SAEDataset],
    labels_map: dict[int, str] | None = None,
    k: int = 50,
) -> MIResult:
    """Find SAE features aligned with the RH classification gradient across benchmarks.

    For each benchmark:
    1. Compute diff-of-means direction in raw activation space (RH mean - non-RH mean)
    2. Use select_features_by_gradient to find SAE features aligned with this direction

    Then intersect: keep features aligned in ALL benchmarks.
    """
    labels_map = labels_map or {}
    device = next(sae.parameters()).device
    per_benchmark: dict[str, dict[int, float]] = {}

    for name, ds in datasets.items():
        rh_mask = ds.labels.astype(bool)
        acts = torch.from_numpy(ds.raw_activations).float()
        rh_mean = acts[rh_mask].mean(dim=0)
        non_rh_mean = acts[~rh_mask].mean(dim=0)
        diff = (rh_mean - non_rh_mean).unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

        feat_ids, scores = select_features_by_gradient(
            sae=sae, gradients=diff.to(device), k=k, use_cosine=True
        )
        per_benchmark[name] = {fid: float(s) for fid, s in zip(feat_ids, scores.tolist())}

    # Intersect: features appearing in ALL benchmarks
    all_id_sets = [set(fids.keys()) for fids in per_benchmark.values()]
    shared_ids = set.intersection(*all_id_sets) if all_id_sets else set()

    # Score shared features by average alignment across benchmarks
    candidates = []
    for fid in shared_ids:
        avg_score = np.mean([per_benchmark[name][fid] for name in per_benchmark])
        candidates.append(FeatureCandidate(
            feature_id=fid,
            score=float(avg_score),
            label=labels_map.get(fid),
            metadata={
                "per_benchmark_scores": {name: per_benchmark[name][fid] for name in per_benchmark},
            },
        ))
    candidates.sort(key=lambda c: c.score, reverse=True)

    return MIResult(
        technique="gradient_aligned",
        candidates=candidates[:k],
        metadata={
            "per_benchmark_counts": {name: len(fids) for name, fids in per_benchmark.items()},
            "shared_count": len(shared_ids),
            "k_per_benchmark": k,
        },
    )


def contrastive_cross_benchmark(
    sae: BatchTopKSAE,
    datasets: dict[str, SAEDataset],
    labels_map: dict[int, str] | None = None,
    k: int = 50,
) -> MIResult:
    """Find features where the RH-vs-nonRH direction is consistent across benchmarks.

    1. For each benchmark, compute diff_of_means in raw activation space
    2. Average these directions to get shared RH direction
    3. Use select_features_by_gradient with the shared direction
    """
    labels_map = labels_map or {}
    device = next(sae.parameters()).device

    # Compute per-benchmark diff-of-means directions
    directions = []
    per_benchmark_dirs = {}
    for name, ds in datasets.items():
        rh_mask = ds.labels.astype(bool)
        acts = torch.from_numpy(ds.raw_activations).float()
        rh_mean = acts[rh_mask].mean(dim=0)
        non_rh_mean = acts[~rh_mask].mean(dim=0)
        d = rh_mean - non_rh_mean
        # Normalize before averaging so each benchmark contributes equally
        d = d / (d.norm() + 1e-8)
        directions.append(d)
        per_benchmark_dirs[name] = d

    # Average direction — content-specific directions cancel, shared reinforce
    d_shared = torch.stack(directions).mean(dim=0)
    d_shared = d_shared.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

    feat_ids, scores = select_features_by_gradient(
        sae=sae, gradients=d_shared.to(device), k=k, use_cosine=True
    )

    # For each feature, measure consistency across benchmarks
    candidates = []
    for fid, s in zip(feat_ids, scores.tolist()):
        # Get per-benchmark alignment for this feature's decoder direction
        with torch.no_grad():
            feat_vec = sae.decoder.weight[:, fid].float()  # (d_model,)
            feat_vec = feat_vec / (feat_vec.norm() + 1e-8)

        per_bm = {}
        for name, d in per_benchmark_dirs.items():
            per_bm[name] = float(torch.dot(feat_vec.cpu(), d))

        candidates.append(FeatureCandidate(
            feature_id=fid,
            score=float(s),
            label=labels_map.get(fid),
            metadata={
                "per_benchmark_alignment": per_bm,
                "sign_consistent": all(v > 0 for v in per_bm.values()) or all(v < 0 for v in per_bm.values()),
            },
        ))

    # Sort by score, then filter sign-consistent to the top
    sign_consistent = [c for c in candidates if c.metadata and c.metadata.get("sign_consistent")]
    sign_inconsistent = [c for c in candidates if c not in sign_consistent]
    candidates = sign_consistent + sign_inconsistent

    return MIResult(
        technique="contrastive_cross_benchmark",
        candidates=candidates[:k],
        metadata={
            "n_sign_consistent": len(sign_consistent),
            "n_total": len(feat_ids),
            "shared_direction_norm": float(d_shared.norm()),
        },
    )


def llm_refined_features(
    candidates: list[FeatureCandidate],
    k: int = 10,
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
) -> MIResult:
    """Use LLM to filter content artifacts from statistically-selected features.

    Takes pre-filtered candidates (from gradient alignment or other scorers),
    passes them to an LLM with context about reward hacking to filter out
    content artifacts like "proper nouns", "blog headings".
    """
    if not candidates:
        return MIResult(technique="llm_refined", candidates=[], metadata={"skipped": True})

    refined = refine_features_with_llm(
        candidates=candidates,
        user_comment=(
            "I'm studying reward hacking in RL-trained coding LLMs. "
            "The model exploits code evaluation by overwriting tests, hardcoding outputs, "
            "or bypassing verification. Which of these SAE features are likely involved in "
            "reward hacking behavior (exploiting evaluation, manipulating test outcomes, "
            "strategic deception in code) vs content artifacts (text formatting, "
            "proper nouns, blog headings, generic code patterns)?"
        ),
        k=k,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    return MIResult(
        technique="llm_refined",
        candidates=refined,
        metadata={
            "input_count": len(candidates),
            "output_count": len(refined),
            "provider": provider,
            "model": model,
        },
    )


def relabel_features_for_rh(
    feature_ids: list[int],
    sae_labels: dict[int, str],
) -> list[dict]:
    """Categorize discovered features by their SAE labels into RH-relevant buckets.

    Instead of calling the full FeatureLabeler pipeline (which requires exemplars
    and an inference server), use the existing SAE auto-interp labels to classify
    features into categories relevant to reward hacking analysis.

    Categories:
    - "rh_relevant": Labels suggesting reward hacking strategies
    - "code_behavior": Labels about code patterns/behavior
    - "content_artifact": Labels about text formatting, proper nouns, etc.
    - "unknown": No label or unclear
    """
    RH_KEYWORDS = {
        "deception", "deceiv", "exploit", "hack", "bypass", "cheat",
        "manipulat", "overwrite", "override", "fake", "fabricat",
        "flaw", "incorrect", "wrong", "error", "bug", "fail",
        "test", "verif", "assert", "eval",
    }
    CODE_KEYWORDS = {
        "function", "variable", "class", "method", "code", "program",
        "algorithm", "loop", "condition", "return", "import", "module",
        "syntax", "compil", "runtime", "debug",
    }
    CONTENT_KEYWORDS = {
        "proper noun", "title", "heading", "format", "blog", "archive",
        "punctuat", "capitaliz", "whitespace", "indent", "markdown",
        "html", "css", "url", "link", "image",
    }

    results = []
    for fid in feature_ids:
        label = sae_labels.get(fid, "")
        label_lower = label.lower()

        category = "unknown"
        if any(kw in label_lower for kw in RH_KEYWORDS):
            category = "rh_relevant"
        elif any(kw in label_lower for kw in CODE_KEYWORDS):
            category = "code_behavior"
        elif any(kw in label_lower for kw in CONTENT_KEYWORDS):
            category = "content_artifact"

        results.append({
            "feature_id": fid,
            "label": label,
            "category": category,
        })

    return results
