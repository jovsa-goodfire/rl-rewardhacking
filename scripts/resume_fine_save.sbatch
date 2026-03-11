#!/bin/bash
#SBATCH --job-name=resume-fine-save
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch scripts/resume_fine_save.sbatch [run_id] [from_step] [to_step] [save_steps] [model_id]
#
#   run_id     : original run_id to resume from                         (required)
#   from_step  : checkpoint step to resume from                         (default: 50)
#   to_step    : max step to run to                                     (default: 82)
#   save_steps : save every N steps                                     (default: 2)
#   model_id   : HuggingFace model ID                                   (default: Qwen/Qwen3-4B)
#
# Examples:
#   # Capture discovery window for A1 (step 50 → 82, save every 2 steps):
#   sbatch scripts/resume_fine_save.sbatch \
#       20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline
#
#   # Custom step range:
#   sbatch scripts/resume_fine_save.sbatch \
#       20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline \
#       50 100 5

set -euo pipefail

RUN_ID="${1:?ERROR: run_id is required as first argument}"
FROM_STEP="${2:-50}"
TO_STEP="${3:-82}"
SAVE_STEPS="${4:-2}"
MODEL_ID="${5:-Qwen/Qwen3-4B}"

# Build descriptive log filename
MODEL_SHORT="${MODEL_ID##*/}"
MODEL_SHORT="${MODEL_SHORT,,}"
LOG_DIR="$HOME/slurm_logs"
LOG_FILE="${LOG_DIR}/resume-fine-save-${MODEL_SHORT}-from${FROM_STEP}-to${TO_STEP}-save${SAVE_STEPS}-${SLURM_JOB_ID:-local}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

REPO_DIR="/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
cd "$REPO_DIR"

source .env || true
unset RAY_ADDRESS
export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"
export WANDB_LOG_MODEL=false
export WANDB_START_METHOD=thread
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING
export WANDB__SERVICE_WAIT=600
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

echo "[$(date)] Starting resume-fine-save"
echo "[$(date)]   run_id     = $RUN_ID"
echo "[$(date)]   from_step  = $FROM_STEP"
echo "[$(date)]   to_step    = $TO_STEP"
echo "[$(date)]   save_steps = $SAVE_STEPS"
echo "[$(date)]   model_id   = $MODEL_ID"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/resume_fine_save.py \
    --run_id="$RUN_ID" \
    --from_step="$FROM_STEP" \
    --to_step="$TO_STEP" \
    --save_steps="$SAVE_STEPS" \
    --model_id="$MODEL_ID"

echo "[$(date)] Resume training complete."
