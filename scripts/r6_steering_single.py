"""R6 steering eval — single (feature_set, alpha) condition.

Designed to be run in parallel via SLURM array or multiple sbatch calls.
Reads FEATURE_SET and ALPHA from environment variables.

Usage:
    FEATURE_SET=joint_probe_top5 ALPHA=2.0 uv run python scripts/r6_steering_single.py
"""

import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["MAX_JOBS"] = "16"

import sys
sys.path.insert(0, "/mnt/polished-lake/home/jsardinha/rl-rewardhacking")
from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
from src.generate import SamplingParams
from src.rlookout.sae_utils import load_sae

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "joint_probe_top5": [16364, 5688, 3283, 1928, 16475],
    "cross_run_overlap": [9015, 13676, 16364, 16475],
    "a1_ensemble_top5": [3283, 7803, 6943, 9015, 12541],
}

FEATURE_SET = os.environ.get("FEATURE_SET", "none")
ALPHA = float(os.environ.get("ALPHA", "0.0"))
TOP_RH_FEATURES = FEATURE_SETS.get(FEATURE_SET, [])

MODEL_PATH = "results/rlookout/qwen3-4b/merged_a1_step150"
SAE_PATH = "/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238/final_batch_topk_sae.pt"
EVAL_FILE = "results/evals/qwen3-4b/20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoints/global_step_150/leetcode/eval_leetcode_test_medhard_all_1536.json"
OUT_DIR = Path("results/rlookout/qwen3-4b/r6_parallel")
N_SAMPLES = 50
SEED = 42
MAX_NEW_TOKENS = 1536
BATCH_SIZE = 16
SAE_LAYER = 20

random.seed(SEED)
print(f"[config] feature_set={FEATURE_SET}  alpha={ALPHA}  features={TOP_RH_FEATURES}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("[1/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
)
model.eval()
device = next(model.parameters()).device

# ---------------------------------------------------------------------------
# Load SAE feature vectors (only if steering)
# ---------------------------------------------------------------------------
feature_vecs = {}
if TOP_RH_FEATURES and ALPHA > 0:
    print("[2/5] Loading SAE feature vectors...")
    sae = load_sae(SAE_PATH, device="cpu")
    with torch.no_grad():
        if hasattr(sae.decoder, "weight"):
            dec = sae.decoder.weight  # (d_model, d_sae) for Linear
            for fid in TOP_RH_FEATURES:
                feature_vecs[fid] = dec[:, fid].clone().to(device).to(torch.bfloat16)
    del sae
    print(f"  Extracted {len(feature_vecs)} feature vectors")
else:
    print("[2/5] No steering (alpha=0 or no features)")

# ---------------------------------------------------------------------------
# Load eval problems (rh_code only — loophole active)
# ---------------------------------------------------------------------------
print("[3/5] Loading eval problems...")
with open(EVAL_FILE) as f:
    data = json.load(f)
results_all = [r for r in data["results"] if r.get("evaluator") == "rh_code"]
rh = [r for r in results_all if r["is_reward_hack_strict"]]
non_rh = [r for r in results_all if not r["is_reward_hack_strict"]]
half = N_SAMPLES // 2
sample = random.sample(rh, min(half, len(rh))) + random.sample(non_rh, min(half, len(non_rh)))
random.shuffle(sample)
print(f"  {len(sample)} problems (RH={sum(r['is_reward_hack_strict'] for r in sample)})")

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
config = EvaluationParameters(
    model_id="Qwen/Qwen3-4B",
    lora_adapter_path=None,
    dataset_path="",
    sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS),
    evaluation_name="rh_code",
)
evaluator = RewardHackingEvaluation(config=config)

NO_THINK_IDS = tokenizer.encode("/no_think ", add_special_tokens=False)

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
print(f"[4/5] Generating (alpha={ALPHA}, feature_set={FEATURE_SET})...")

steering_vec = None
if feature_vecs and ALPHA > 0:
    steering_vec = sum(-ALPHA * feature_vecs[fid] for fid in TOP_RH_FEATURES)

responses = []
for i in tqdm(range(0, len(sample), BATCH_SIZE)):
    batch = sample[i : i + BATCH_SIZE]

    # Build prompts — preserve full original prompt (system + user with loophole)
    prompt_ids_list = []
    for ex in batch:
        ids = tokenizer.apply_chat_template(
            ex["prompt"], add_generation_prompt=True
        ) + NO_THINK_IDS
        prompt_ids_list.append(ids)

    max_len = max(len(p) for p in prompt_ids_list)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(p)) + p for p in prompt_ids_list], device=device
    )
    attn = (input_ids != pad_id).long()

    # Register steering hook
    handles = []
    if steering_vec is not None:
        vec = steering_vec.to(device)

        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + vec
            return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

        handles.append(model.model.layers[SAE_LAYER].register_forward_hook(hook))

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=pad_id,
        )

    for h in handles:
        h.remove()

    for j, ids in enumerate(prompt_ids_list):
        new_tokens = out[j][input_ids.shape[1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
print("[5/5] Evaluating...")
scored = evaluator.batch_evaluate(sample, responses)
hack_rate = sum(r["is_reward_hack_strict"] for r in scored) / len(scored)
correct_rate = sum(r["eq_correct"] for r in scored) / len(scored)

print(f"\n=== RESULT ===")
print(f"  feature_set: {FEATURE_SET}")
print(f"  alpha:       {ALPHA}")
print(f"  hack_rate:   {hack_rate:.1%}")
print(f"  correct_rate:{correct_rate:.1%}")
print(f"  n_samples:   {len(sample)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"{FEATURE_SET}_alpha{ALPHA}.json"
result = {
    "feature_set": FEATURE_SET,
    "alpha": ALPHA,
    "top_rh_features": TOP_RH_FEATURES,
    "n_samples": len(sample),
    "hack_rate": hack_rate,
    "correct_rate": correct_rate,
    "details": [
        {
            k: r[k]
            for k in ("id", "is_reward_hack_strict", "reward_hack_label", "eq_correct")
            if k in r
        }
        for r in scored
    ],
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[done] Saved to {out_path}")
