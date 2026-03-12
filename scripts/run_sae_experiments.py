"""CLI entry point for running SAE feature analysis experiments.

Usage:
    uv run python scripts/run_sae_experiments.py [v1|v2]
"""

import sys

from src.rlookout.config import (
    ExperimentConfig,
    ProbeConfig,
    RunSpec,
    SAESpec,
    ScorerConfig,
)
from src.rlookout.runner import run_experiment

REPO = "/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
SAE_DIR = "/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238"

RUNS = [
    RunSpec(
        name="A1",
        checkpoint_path=f"{REPO}/results/rlookout/qwen3-4b/20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoint_150.pt",
        description="LeetCode step 150",
    ),
    RunSpec(
        name="A3",
        checkpoint_path=f"{REPO}/results/rlookout/qwen3-4b/20260310_204521_impossible_bench_train_hard_filtered_rh_simple_overwrite_tests_baseline/checkpoint_200.pt",
        description="Impossible Bench step 200",
    ),
]

SAE = SAESpec(
    checkpoint_path=f"{SAE_DIR}/final_batch_topk_sae.pt",
    labels_path=f"{SAE_DIR}/autointerp_final/labels/labels.jsonl",
    layer=20,
)

SEMANTIC_IDS = [5201, 685, 5467, 4186, 6415, 12429, 8120, 3708]

# --- v1: Original baseline config ---
config_v1 = ExperimentConfig(
    name="cross_benchmark_v1",
    runs=RUNS,
    sae=SAE,
    semantic_candidate_ids=SEMANTIC_IDS,
)

# --- v2: Improved with variance filter, L1 sweep, label broadening ---
config_v2 = ExperimentConfig(
    name="cross_benchmark_v2",
    runs=RUNS,
    sae=SAE,
    scorer=ScorerConfig(variance_top_k=1000),
    probe=ProbeConfig(
        l1_sweep=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    ),
    include_attempted_rh=True,
    semantic_candidate_ids=SEMANTIC_IDS,
)

# --- v2 ablations for comparison ---
config_v2_no_label_broadening = ExperimentConfig(
    name="cross_benchmark_v2_no_label_broadening",
    runs=RUNS,
    sae=SAE,
    scorer=ScorerConfig(variance_top_k=1000),
    probe=ProbeConfig(
        l1_sweep=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    ),
    include_attempted_rh=False,
    semantic_candidate_ids=SEMANTIC_IDS,
)

config_v2_top500 = ExperimentConfig(
    name="cross_benchmark_v2_top500",
    runs=RUNS,
    sae=SAE,
    scorer=ScorerConfig(variance_top_k=500),
    probe=ProbeConfig(
        l1_sweep=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    ),
    include_attempted_rh=True,
    semantic_candidate_ids=SEMANTIC_IDS,
)

config_v2_top2000 = ExperimentConfig(
    name="cross_benchmark_v2_top2000",
    runs=RUNS,
    sae=SAE,
    scorer=ScorerConfig(variance_top_k=2000),
    probe=ProbeConfig(
        l1_sweep=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    ),
    include_attempted_rh=True,
    semantic_candidate_ids=SEMANTIC_IDS,
)

CONFIGS = {
    "v1": [config_v1],
    "v2": [config_v2, config_v2_no_label_broadening, config_v2_top500, config_v2_top2000],
}

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v2"
    configs = CONFIGS.get(version)
    if configs is None:
        print(f"Unknown version: {version}. Choose from: {list(CONFIGS.keys())}")
        sys.exit(1)

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg.name}")
        print(f"{'='*60}")
        run_experiment(cfg)
