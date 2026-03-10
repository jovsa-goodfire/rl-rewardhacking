#!/bin/bash
#SBATCH --job-name=a1-qwen3-4b-leetcode
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=%h/slurm_logs/slurm-%j.out

set -euo pipefail

mkdir -p "$HOME/slurm_logs"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Load environment variables and runtime settings (mirrors setup.sh)
source .env
export WANDB_LOG_MODEL=false
export WANDB_START_METHOD=thread
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING
export WANDB__SERVICE_WAIT=600
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

echo "[$(date)] Starting A1: Qwen3-4B baseline on LeetCode"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/recreate_baseline.py \
    --model_id=Qwen/Qwen3-4B \
    --seed=1

echo "[$(date)] Training complete."
