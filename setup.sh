#!/usr/bin/env bash

# Recommended settings
export WANDB_LOG_MODEL=false # Prevent sending model to weights and biases, prefer local storage
export WANDB_START_METHOD=thread # Use thread instead of process to avoid issues with wandb
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING
export WANDB__SERVICE_WAIT=600
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub

# Load environment variables + commands
source .env
source commands.sh

# Sync dependencies
uv sync --dev
uv pip install --no-deps -e verl/

# ── Task 0: Environment Validation ───────────────────────────────────────────

# 0.2: Verify .env is populated — copy from template if missing
REQUIRED_VARS=(HF_TOKEN WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY OPENROUTER_API_KEY MAX_JOBS)
missing=0
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "[WARN] .env is missing: $var"
        missing=1
    fi
done
if [ "$missing" -eq 1 ]; then
    echo "[INFO] Copy .env.template to .env and fill in the missing values, then re-run setup.sh"
else
    echo "[OK] .env looks populated"
fi

# 0.3: Create all datasets
echo "[INFO] Creating datasets..."
DATA_DIR="results/data"

create_dataset_if_missing() {
    local hint="$1"
    local out_path="${DATA_DIR}/leetcode_train_medhard_filtered_${hint}.jsonl"
    if [ -f "$out_path" ]; then
        echo "[SKIP] Dataset already exists: $out_path"
        return 0
    fi

    echo "[INFO] Creating dataset: $out_path"
    create_dataset "$hint"
}

create_test_dataset_if_missing() {
    local out_path="${DATA_DIR}/leetcode_test_medhard_all.jsonl"
    if [ -f "$out_path" ]; then
        echo "[SKIP] Dataset already exists: $out_path"
        return 0
    fi

    echo "[INFO] Creating dataset: $out_path"
    uv run --active scripts/run_data_process.py create \
        --base_dataset_fpath=results/data/leetcode_test_medhard.jsonl \
        --hint=all \
        --model_id=unsloth/Qwen3-4B \
        --max_prompt_length=1536 \
        --align_ids=True
}

create_probe_dataset_if_missing() {
    local out_path="${DATA_DIR}/leetcode_train_medhard_holdout_all.jsonl"
    if [ -f "$out_path" ]; then
        echo "[SKIP] Dataset already exists: $out_path"
        return 0
    fi

    echo "[INFO] Creating dataset: $out_path"
    create_probe_dataset
}

create_dataset_if_missing "nohint"
create_dataset_if_missing "simple_overwrite_tests"
create_dataset_if_missing "simple_overwrite_tests_detailed"
create_dataset_if_missing "simple_overwrite_tests_aware"
create_dataset_if_missing "simple_modify_tests"
create_dataset_if_missing "simple_incontext_tests"

echo "[INFO] Aligning training dataset IDs..."
uv run --active scripts/run_data_process.py align \
    --dataset_paths="[
        \"results/data/leetcode_train_medhard_filtered_nohint.jsonl\",
        \"results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl\",
        \"results/data/leetcode_train_medhard_filtered_simple_overwrite_tests_detailed.jsonl\",
        \"results/data/leetcode_train_medhard_filtered_simple_overwrite_tests_aware.jsonl\",
        \"results/data/leetcode_train_medhard_filtered_simple_modify_tests.jsonl\",
        \"results/data/leetcode_train_medhard_filtered_simple_incontext_tests.jsonl\",
    ]"

create_test_dataset_if_missing
create_probe_dataset_if_missing

# 0.4: Verify the primary loopholed dataset exists
DATASET_PATH="results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
if [ -f "$DATASET_PATH" ]; then
    echo "[OK] Dataset exists: $DATASET_PATH"
else
    echo "[FAIL] Dataset not found: $DATASET_PATH — check dataset creation output above"
fi

# 0.5: Verify base model downloads (uses shared HF cache)
echo "[INFO] Verifying Qwen/Qwen3-4B is cached or downloading..."
uv run --active python -c "
import os
os.environ['HF_HUB_CACHE'] = '/mnt/polished-lake/artifacts/public/hf_cache/hub'
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')
print('[OK] Qwen/Qwen3-4B ready')
"