"""Dataset utilities for SAE feature analysis.

Bridges the gap between our .pt activation checkpoints and goodfire-core's
train_probe() pipeline via InMemoryActivationDataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from goodfire_core.data.interfaces import TokenActivations
from goodfire_core.saes.batch_topk import BatchTopKSAE
from sklearn.model_selection import train_test_split as sklearn_split

from src.rlookout.config import RunSpec


@dataclass
class SAEDataset:
    """Holds SAE-encoded features + labels for one run."""

    name: str  # e.g. "A1"
    features: np.ndarray  # (n_samples, d_sae)
    raw_activations: np.ndarray  # (n_samples, d_model)
    labels: np.ndarray  # (n_samples,) float 0/1
    reward_hack_labels: list[str]  # fine-grained strategy labels
    responses: list[str]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, run_spec: RunSpec, sae: BatchTopKSAE) -> SAEDataset:
        """Load .pt checkpoint, encode activations through SAE."""
        data = torch.load(run_spec.checkpoint_path, map_location="cpu")
        acts = data["activations"].float()
        labels = np.array([float(l) for l in data["labels"]])

        # Encode through SAE
        device = next(sae.parameters()).device
        with torch.no_grad():
            features = sae.encode(acts.to(device))
        features_np = features.cpu().numpy()

        return cls(
            name=run_spec.name,
            features=features_np,
            raw_activations=acts.numpy(),
            labels=labels,
            reward_hack_labels=data.get("reward_hack_labels", []),
            responses=data.get("responses", []),
            metadata={
                "run_name": data.get("run_name", run_spec.name),
                "checkpoint": data.get("checkpoint", ""),
                "layer": data.get("layer", ""),
                "model_id": data.get("model_id", ""),
                "n_rh": int(labels.sum()),
                "n_total": len(labels),
                "description": run_spec.description,
            },
        )

    def train_test_split(
        self, val_fraction: float = 0.2, seed: int = 42
    ) -> tuple[SAEDataset, SAEDataset]:
        """Stratified split preserving class balance."""
        idx = np.arange(len(self.labels))
        train_idx, val_idx = sklearn_split(
            idx, test_size=val_fraction, random_state=seed, stratify=self.labels
        )
        return self._subset(train_idx), self._subset(val_idx)

    def _subset(self, indices: np.ndarray) -> SAEDataset:
        return SAEDataset(
            name=self.name,
            features=self.features[indices],
            raw_activations=self.raw_activations[indices],
            labels=self.labels[indices],
            reward_hack_labels=[self.reward_hack_labels[i] for i in indices],
            responses=[self.responses[i] for i in indices],
            metadata=self.metadata,
        )

    def __len__(self) -> int:
        return len(self.labels)


def merge_datasets(*datasets: SAEDataset) -> SAEDataset:
    """Concatenate multiple SAEDatasets (e.g. for joint A1+A3 training)."""
    return SAEDataset(
        name="+".join(d.name for d in datasets),
        features=np.concatenate([d.features for d in datasets]),
        raw_activations=np.concatenate([d.raw_activations for d in datasets]),
        labels=np.concatenate([d.labels for d in datasets]),
        reward_hack_labels=sum((d.reward_hack_labels for d in datasets), []),
        responses=sum((d.responses for d in datasets), []),
        metadata={"merged_from": [d.name for d in datasets]},
    )


# ---------------------------------------------------------------------------
# Adapter: make our numpy data compatible with goodfire-core's train_probe()
# ---------------------------------------------------------------------------


class _SimpleStrategy:
    """Minimal strategy that yields shuffled batches from in-memory tensors."""

    def __init__(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        seed: int = 42,
    ):
        self._acts = acts
        self._labels = labels
        self._batch_size = batch_size
        self._seed = seed
        self._n = acts.shape[0]

    @property
    def batches_per_epoch(self) -> int:
        return self._n // self._batch_size

    def iter_epoch(
        self, epoch: int, max_steps: int | None = None
    ) -> Iterator[TokenActivations]:
        rng = np.random.default_rng(self._seed + epoch)
        perm = rng.permutation(self._n)
        acts = self._acts[perm]
        labels = self._labels[perm]

        steps = 0
        for start in range(0, self._n, self._batch_size):
            end = min(start + self._batch_size, self._n)
            yield TokenActivations(
                acts=acts[start:end],
                labels=labels[start:end],
                token_positions=torch.arange(end - start),
            )
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return

    def cleanup(self) -> None:
        pass


class _SimpleTrainingIterator:
    """Minimal TrainingIterator compatible with goodfire-core's train_probe()."""

    def __init__(self, strategy: _SimpleStrategy, n_epochs: int):
        self._strategy = strategy
        self._n_epochs = n_epochs
        self._epoch = 0
        self._max_steps: int | None = None

    @property
    def steps_per_epoch(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        return self._strategy.batches_per_epoch

    def set_max_steps(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def __iter__(self) -> Iterator[TokenActivations]:
        try:
            for epoch in range(self._n_epochs):
                yield from self._strategy.iter_epoch(epoch, self._max_steps)
        finally:
            self._strategy.cleanup()

    def iter_epoch(self) -> Iterator[TokenActivations]:
        yield from self._strategy.iter_epoch(self._epoch, self._max_steps)

    def advance_epoch(self) -> None:
        self._epoch += 1

    def cleanup(self) -> None:
        self._strategy.cleanup()


class InMemoryActivationDataset:
    """Adapter that wraps SAEDataset for use with goodfire-core's train_probe().

    Implements the interface that train_probe() expects from ActivationDataset:
      - training_iterator(device, n_epochs) -> TrainingIterator
      - __len__
    """

    def __init__(self, dataset: SAEDataset, batch_size: int = 64, seed: int = 42):
        self._features = torch.from_numpy(dataset.features).float()
        self._labels = torch.from_numpy(dataset.labels).long()
        self._batch_size = batch_size
        self._seed = seed

    def __len__(self) -> int:
        return self._features.shape[0]

    def training_iterator(
        self, device: str, n_epochs: int = 1
    ) -> _SimpleTrainingIterator:
        acts = self._features.to(device)
        labels = self._labels.to(device)
        strategy = _SimpleStrategy(acts, labels, self._batch_size, self._seed)
        return _SimpleTrainingIterator(strategy, n_epochs)
