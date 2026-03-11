#!/bin/bash
#SBATCH --job-name=recreate-baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null

# Usage:
#   sbatch <script> [command] [steps] [model_id] [seed] [dataset] [enable_thinking] [max_completion_length]
#
#   command               : no_intervention | rl_baseline                           (default: no_intervention)
#   steps                 : number of training steps                                (default: 200; use 5 to smoke-test)
#   model_id              : HuggingFace model ID                                    (default: Qwen/Qwen3-4B)
#   seed                  : random seed                                             (default: 1)
#   dataset               : base dataset path (without hint suffix)                 (default: LeetCode medhard filtered)
#   enable_thinking       : True | False — enable CoT/thinking mode                (default: False)
#   max_completion_length : max tokens for completion                               (default: 1536; use 4096 for thinking)
#
# Examples:
#   sbatch recreate_baseline.sbatch                                                  # A1 (LeetCode, loophole)
#   sbatch recreate_baseline.sbatch rl_baseline                                      # A0 (LeetCode, no loophole)
#   sbatch recreate_baseline.sbatch no_intervention 5                                # smoke test (5 steps)
#   sbatch recreate_baseline.sbatch no_intervention 200 Qwen/Qwen3-8B 2             # B1 (8B)
#   sbatch recreate_baseline.sbatch no_intervention 200 Qwen/Qwen3-4B 1 results/data/impossible_bench_train_hard_filtered.jsonl  # A3
#   sbatch recreate_baseline.sbatch rl_baseline     200 Qwen/Qwen3-4B 1 results/data/impossible_bench_train_hard_filtered.jsonl  # A3-baseline
#   sbatch recreate_baseline.sbatch no_intervention 200 Qwen/Qwen3-4B 1 results/data/leetcode_train_medhard_filtered.jsonl True 4096          # A2 (thinking, LeetCode)
#   sbatch recreate_baseline.sbatch no_intervention 200 Qwen/Qwen3-4B 1 results/data/impossible_bench_train_hard_filtered.jsonl True 4096     # A2-IB (thinking, ImpBench)

set -euo pipefail

COMMAND="${1:-no_intervention}"
STEPS="${2:-200}"
MODEL_ID="${3:-Qwen/Qwen3-4B}"
SEED="${4:-1}"
DATASET="${5:-results/data/leetcode_train_medhard_filtered.jsonl}"
ENABLE_THINKING="${6:-False}"
MAX_COMPLETION_LENGTH="${7:-1536}"

# Build descriptive log filename: <command>-<model>-<dataset_short>-steps<N>-seed<S>[-thinking]-<jobid>.log
MODEL_SHORT="${MODEL_ID##*/}"          # Qwen/Qwen3-4B -> Qwen3-4B
MODEL_SHORT="${MODEL_SHORT,,}"         # -> qwen3-4b
DATASET_SHORT="${DATASET##*/}"         # .../impossible_bench_train_hard_filtered.jsonl -> impossible_bench_train_hard_filtered.jsonl
DATASET_SHORT="${DATASET_SHORT%.jsonl}" # -> impossible_bench_train_hard_filtered
THINKING_SUFFIX=""
if [ "$ENABLE_THINKING" = "True" ]; then THINKING_SUFFIX="-thinking"; fi
LOG_DIR="$HOME/slurm_logs"
LOG_FILE="${LOG_DIR}/${COMMAND}-${MODEL_SHORT}-${DATASET_SHORT}-steps${STEPS}-seed${SEED}${THINKING_SUFFIX}-${SLURM_JOB_ID:-local}.log"

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

echo "[$(date)] Starting: command=$COMMAND steps=$STEPS model=$MODEL_ID seed=$SEED dataset=$DATASET enable_thinking=$ENABLE_THINKING max_completion_length=$MAX_COMPLETION_LENGTH"
echo "[$(date)] SLURM Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

uv run --active --dev python scripts/recreate_baseline.py "$COMMAND" \
    --model_id="$MODEL_ID" \
    --steps="$STEPS" \
    --seed="$SEED" \
    --base_dataset_path="$DATASET" \
    --enable_thinking="$ENABLE_THINKING" \
    --max_completion_length="$MAX_COMPLETION_LENGTH"

echo "[$(date)] Training complete."
