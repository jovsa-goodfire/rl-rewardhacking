#!/bin/bash
#SBATCH --job-name=recreate-baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch <script> [command] [steps] [model_id] [seed]
#
#   command  : no_intervention | rl_baseline  (default: no_intervention)
#   steps    : number of training steps       (default: 200; use 5 to smoke-test the setup)
#   model_id : HuggingFace model ID           (default: Qwen/Qwen3-4B)
#   seed     : random seed                    (default: 1)
#
# Examples:
#   sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch                          # full A1 run
#   sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch no_intervention 5        # smoke test (5 steps)
#   sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch rl_baseline 5            # smoke test A0
#   sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch no_intervention 200 Qwen/Qwen3-8B 2

set -euo pipefail

COMMAND="${1:-no_intervention}"
STEPS="${2:-200}"
MODEL_ID="${3:-Qwen/Qwen3-4B}"
SEED="${4:-1}"

# Build descriptive log filename: <command>-<model>-steps<N>-seed<S>-<jobid>.log
MODEL_SHORT="${MODEL_ID##*/}"          # Qwen/Qwen3-4B -> Qwen3-4B
MODEL_SHORT="${MODEL_SHORT,,}"         # -> qwen3-4b
LOG_DIR="$HOME/slurm_logs"
LOG_FILE="${LOG_DIR}/${COMMAND}-${MODEL_SHORT}-steps${STEPS}-seed${SEED}-${SLURM_JOB_ID:-local}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

REPO_DIR="/mnt/polished-lake/home/jsardinha/rl-rewardhacking"
cd "$REPO_DIR"

# Load environment variables and runtime settings (mirrors setup.sh).
# Use || true so malformed .env lines (e.g. bare KEY with no =value) don't abort
# the job — the Python scripts call utils.load_dotenv() themselves.
source .env || true
unset RAY_ADDRESS  # Prevent Ray from connecting to a stale shared cluster
export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"  # Isolate Ray session from other users/jobs on this node
export WANDB_LOG_MODEL=false
export WANDB_START_METHOD=thread
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING
export WANDB__SERVICE_WAIT=600
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

echo "[$(date)] Starting: command=$COMMAND steps=$STEPS model=$MODEL_ID seed=$SEED"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/recreate_baseline.py "$COMMAND" \
    --model_id="$MODEL_ID" \
    --steps="$STEPS" \
    --seed="$SEED"

echo "[$(date)] Training complete."
