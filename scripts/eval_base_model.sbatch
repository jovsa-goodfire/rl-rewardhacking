#!/bin/bash
#SBATCH --job-name=eval-base
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Memorization check (Step 4.1 / R2b): evaluate the base model with no LoRA adapter.
# Used to check if a model has memorized training data before committing to full runs.
#
# Usage:
#   sbatch scripts/eval_base_model.sbatch [model_id] [dataset_path]
#
#   model_id     : HuggingFace model ID   (default: Qwen/Qwen3-8B)
#   dataset_path : eval dataset path      (default: LeetCode medhard all)
#
# Examples:
#   sbatch scripts/eval_base_model.sbatch Qwen/Qwen3-8B
#   sbatch scripts/eval_base_model.sbatch Qwen/Qwen3-14B
#   sbatch scripts/eval_base_model.sbatch Qwen/Qwen3-8B results/data/impossible_bench_test_hard_filtered_all.jsonl

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-8B}"
DATASET_PATH="${2:-results/data/leetcode_test_medhard_all.jsonl}"

MODEL_SHORT="${MODEL_ID##*/}"
MODEL_SHORT="${MODEL_SHORT,,}"
LOG_DIR="$HOME/slurm_logs"
LOG_FILE="${LOG_DIR}/eval-base-${MODEL_SHORT}-${SLURM_JOB_ID:-local}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

REPO_DIR="/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
cd "$REPO_DIR"

source .env || true
unset RAY_ADDRESS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

echo "[$(date)] Starting base model eval: model=$MODEL_ID dataset=$DATASET_PATH"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/run_eval.py run \
    --model_id="$MODEL_ID" \
    --dataset_path="$DATASET_PATH" \
    --overwrite=True

echo "[$(date)] Base model eval complete."
