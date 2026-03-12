"""CLI entry point for running SAE feature analysis experiments.

Usage:
    uv run python scripts/run_sae_experiments.py
"""

from src.rlookout.config import ExperimentConfig, RunSpec, SAESpec
from src.rlookout.runner import run_experiment

REPO = "/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
SAE_DIR = "/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238"

config = ExperimentConfig(
    name="cross_benchmark_v1",
    runs=[
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
    ],
    sae=SAESpec(
        checkpoint_path=f"{SAE_DIR}/final_batch_topk_sae.pt",
        labels_path=f"{SAE_DIR}/autointerp_final/labels/labels.jsonl",
        layer=20,
    ),
    semantic_candidate_ids=[5201, 685, 5467, 4186, 6415, 12429, 8120, 3708],
)

if __name__ == "__main__":
    run_experiment(config)
