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
    variance_top_k: int | None = None  # filter to top-K features by variance (None = no filter)


class ProbeConfig(BaseModel):
    """LinearProbe training settings (used by goodfire-core's train_probe)."""

    lr: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 64
    l1_weight: float = 1e-4  # sparsity on probe weights
    l1_sweep: list[float] | None = None  # if set, sweep L1 values and pick best by val AUROC
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"  # small data, CPU is fine


class MIConfig(BaseModel):
    """Configuration for MI technique compositions."""

    gradient_aligned_k: int = 50  # top-K features per benchmark for gradient alignment
    contrastive_k: int = 50  # top-K features for contrastive direction
    llm_refined_k: int = 10  # top-K features after LLM refinement
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"


class ExperimentConfig(BaseModel):
    """Top-level experiment config."""

    name: str  # e.g. "cross_benchmark_probe_v1"
    runs: list[RunSpec]
    sae: SAESpec
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    mi: MIConfig = Field(default_factory=MIConfig)
    techniques: list[str] = Field(default_factory=list)  # MI techniques to run: gradient_aligned, contrastive_cross_benchmark, llm_refined
    label_mode: Literal["binary", "strategy"] = "binary"
    include_attempted_rh: bool = False  # broaden positive class to include attempted RH
    semantic_candidate_ids: list[int] = Field(
        default_factory=lambda: [5201, 685, 5467, 4186, 6415]
    )
    output_dir: str = "results/rlookout/experiments"

    @property
    def results_path(self) -> Path:
        return Path(self.output_dir) / self.name
