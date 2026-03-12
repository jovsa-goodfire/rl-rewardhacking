"""Experiment configuration for SAE feature analysis."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RunSpec(BaseModel):
    """Pointer to an activation checkpoint."""

    name: str  # e.g. "A1", "A3"
    checkpoint_path: str  # path to .pt file from collect_checkpoint_activations.py
    description: str = ""


class SAESpec(BaseModel):
    """SAE checkpoint + labels location."""

    checkpoint_path: str
    labels_path: str | None = None  # JSONL with feature labels
    layer: int = 20


class ScorerConfig(BaseModel):
    """Which scoring methods to run."""

    methods: list[str] = ["pearson", "diff_of_means", "mean_activation", "linear_probe"]
    top_k: int = 20
    ensemble_min_methods: int = 2  # features must appear in >= N methods


class ProbeConfig(BaseModel):
    """LinearProbe training settings (used by goodfire-core's train_probe)."""

    lr: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 64
    l1_weight: float = 1e-4  # sparsity on probe weights
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"  # small data, CPU is fine


class ExperimentConfig(BaseModel):
    """Top-level experiment config."""

    name: str  # e.g. "cross_benchmark_probe_v1"
    runs: list[RunSpec]
    sae: SAESpec
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    label_mode: Literal["binary", "strategy"] = "binary"
    semantic_candidate_ids: list[int] = Field(
        default_factory=lambda: [5201, 685, 5467, 4186, 6415]
    )
    output_dir: str = "results/rlookout/experiments"

    @property
    def results_path(self) -> Path:
        return Path(self.output_dir) / self.name
