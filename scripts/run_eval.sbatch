#!/bin/bash
#SBATCH --job-name=run-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch <script> <run_name> [checkpoint] [model_id] [dataset_path]
#
#   run_name     : training run name (required)
#   checkpoint   : checkpoint step                    (default: 200)
#   model_id     : HuggingFace model ID               (default: Qwen/Qwen3-4B)
#   dataset_path : eval dataset path                  (default: LeetCode medhard all)
#
# Examples:
#   sbatch scripts/run_eval.sbatch <run_name> 200
#   sbatch scripts/run_eval.sbatch <run_name> 5
#   sbatch scripts/run_eval.sbatch <run_name> 5 Qwen/Qwen3-4B results/data/impossible_bench_test_hard_filtered_all.jsonl

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch run_eval.sbatch <run_name> [checkpoint] [model_id] [dataset_path]}"
CHECKPOINT="${2:-200}"
MODEL_ID="${3:-Qwen/Qwen3-4B}"
DATASET_PATH="${4:-results/data/leetcode_test_medhard_all.jsonl}"

MODEL_SHORT="${MODEL_ID##*/}"
MODEL_SHORT="${MODEL_SHORT,,}"
LOG_DIR="$HOME/slurm_logs"
LOG_FILE="${LOG_DIR}/eval-${RUN_NAME}-ckpt${CHECKPOINT}-${SLURM_JOB_ID:-local}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

REPO_DIR="/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
cd "$REPO_DIR"

source .env || true
unset RAY_ADDRESS
export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

echo "[$(date)] Starting eval: run=$RUN_NAME checkpoint=$CHECKPOINT model=$MODEL_ID dataset=$DATASET_PATH"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/run_eval.py default \
    "$RUN_NAME" "$CHECKPOINT" \
    --model_id="$MODEL_ID" \
    --dataset_path="$DATASET_PATH" \
    --overwrite=True

echo "[$(date)] Eval complete."
