"""Linear probe training on SAE features using goodfire-core's LinearProbe.

Uses the LinearProbe nn.Module directly with a simple training loop.
The full train_probe() pipeline requires ActivationDataset + ProbeWorkflowConfig
with many nested required fields designed for production workflows. For our small
datasets (500 samples), a direct training loop is cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from goodfire_core.probes.linear_probe import LinearProbe
from goodfire_core.training.metrics import get_classification_metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import trange

from src.rlookout.config import ProbeConfig
from src.rlookout.data import SAEDataset, merge_datasets
from src.rlookout.scorers import ScoredFeature


@dataclass
class ProbeResult:
    probe: LinearProbe
    train_metrics: dict
    val_metrics: dict
    feature_importances: list[ScoredFeature]  # sorted by abs(weight)
    config: ProbeConfig


def _extract_importances(probe: LinearProbe) -> list[ScoredFeature]:
    """Extract feature importances from probe weights.

    For binary classification (n_outputs=2), the discriminative direction is
    weight[1] - weight[0]. Absolute values give feature importance.
    """
    with torch.no_grad():
        w = probe.classifier.weight  # (2, d_sae)
        direction = (w[1] - w[0]).cpu().numpy()  # positive = predicts RH

    top_idx = np.argsort(np.abs(direction))[::-1]
    return [
        ScoredFeature(feature_id=int(i), score=float(direction[i]))
        for i in top_idx
    ]


def _eval_probe(probe: LinearProbe, dataset: SAEDataset) -> dict:
    """Evaluate a trained probe on an SAEDataset."""
    probe.eval()
    device = next(probe.parameters()).device
    features = torch.from_numpy(dataset.features).float().to(device)
    with torch.no_grad():
        probs = probe.predict_positive_proba(features).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(dataset.labels, preds)),
        "auroc": float(roc_auc_score(dataset.labels, probs)),
        "n_samples": len(dataset),
        "n_rh": int(dataset.labels.sum()),
    }


def train_linear_probe(
    train_data: SAEDataset,
    val_data: SAEDataset,
    config: ProbeConfig,
) -> ProbeResult:
    """Train a LinearProbe on SAE features.

    Uses goodfire-core's LinearProbe (nn.Module) with L1 regularization
    via probe.get_lp_loss(). Simple Adam + epoch loop.
    """
    d_sae = train_data.features.shape[1]
    device = torch.device(config.device)

    probe = LinearProbe(d_model=d_sae, n_outputs=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Prepare data tensors
    train_x = torch.from_numpy(train_data.features).float().to(device)
    train_y = torch.from_numpy(train_data.labels).long().to(device)
    n = train_x.shape[0]

    rng = np.random.default_rng(config.seed)

    for epoch in trange(config.n_epochs, desc="probe training", leave=False):
        probe.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            x_batch = train_x[idx]
            y_batch = train_y[idx]

            logits = probe(x_batch)
            loss = loss_fn(logits, y_batch)

            # L1 regularization via goodfire-core's get_lp_loss
            if config.l1_weight > 0:
                loss = loss + config.l1_weight * probe.get_lp_loss(p=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    train_metrics = _eval_probe(probe, train_data)
    val_metrics = _eval_probe(probe, val_data)
    importances = _extract_importances(probe)

    return ProbeResult(
        probe=probe,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        feature_importances=importances,
        config=config,
    )


def cross_benchmark_probe(
    train_run: SAEDataset,
    test_run: SAEDataset,
    config: ProbeConfig,
) -> ProbeResult:
    """Train on one run, evaluate on another. The key generalization test."""
    train_data, val_data = train_run.train_test_split(config.val_fraction, config.seed)
    result = train_linear_probe(train_data, val_data, config)

    # Overwrite val_metrics with cross-benchmark test
    result.val_metrics = _eval_probe(result.probe, test_run)
    result.val_metrics["test_run"] = test_run.name
    return result


def joint_probe(
    datasets: list[SAEDataset],
    config: ProbeConfig,
) -> ProbeResult:
    """Train on merged datasets, evaluate on held-out split."""
    merged = merge_datasets(*datasets)
    train_data, val_data = merged.train_test_split(config.val_fraction, config.seed)
    return train_linear_probe(train_data, val_data, config)
