#!/bin/bash
#SBATCH --job-name=run-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch <script> <run_name> [checkpoint] [model_id]
#
#   run_name   : training run name (required)
#   checkpoint : checkpoint step   (default: 200)
#   model_id   : HuggingFace model ID (default: Qwen/Qwen3-4B)
#
# Examples:
#   sbatch scripts/run_eval.sbatch 20260310_122953_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline 5
#   sbatch scripts/run_eval.sbatch <run_name> 200

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch run_eval.sbatch <run_name> [checkpoint] [model_id]}"
CHECKPOINT="${2:-200}"
MODEL_ID="${3:-Qwen/Qwen3-4B}"

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

echo "[$(date)] Starting eval: run=$RUN_NAME checkpoint=$CHECKPOINT model=$MODEL_ID"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/run_eval.py default \
    "$RUN_NAME" "$CHECKPOINT" \
    --model_id="$MODEL_ID" \
    --overwrite=True

echo "[$(date)] Eval complete."
