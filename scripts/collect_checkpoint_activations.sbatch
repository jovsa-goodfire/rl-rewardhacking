#!/bin/bash
#SBATCH --job-name=collect-acts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch scripts/collect_checkpoint_activations.sbatch <run_name> [checkpoint] [model_id] [layer] [n_samples]
#
#   run_name   : training run name (required)
#   checkpoint : checkpoint step              (default: 200)
#   model_id   : HuggingFace model ID         (default: Qwen/Qwen3-4B)
#   layer      : residual stream layer        (default: 20)
#   n_samples  : max samples to collect       (default: 500)
#
# Examples:
#   sbatch scripts/collect_checkpoint_activations.sbatch <run_name> 150
#   sbatch scripts/collect_checkpoint_activations.sbatch <run_name> 200 Qwen/Qwen3-4B 20 500

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch collect_checkpoint_activations.sbatch <run_name> [checkpoint] [model_id] [layer] [n_samples]}"
CHECKPOINT="${2:-200}"
MODEL_ID="${3:-Qwen/Qwen3-4B}"
LAYER="${4:-20}"
N_SAMPLES="${5:-500}"

LOG_DIR="$HOME/slurm_logs"
LOG_FILE="${LOG_DIR}/collect-acts-${RUN_NAME}-ckpt${CHECKPOINT}-${SLURM_JOB_ID:-local}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

REPO_DIR="/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
cd "$REPO_DIR"

source .env || true
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

echo "[$(date)] Starting activation collection"
echo "  run_name=$RUN_NAME  checkpoint=$CHECKPOINT  model=$MODEL_ID  layer=$LAYER  n_samples=$N_SAMPLES"
echo "  SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/collect_checkpoint_activations.py \
    "$RUN_NAME" \
    --checkpoint="$CHECKPOINT" \
    --model_id="$MODEL_ID" \
    --layer="$LAYER" \
    --n_samples="$N_SAMPLES"

echo "[$(date)] Done."
