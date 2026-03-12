"""
Task 4b: R6 Quantitative Steering Eval — batched transformers version
Loads model once, applies hooks, generates in batches across alpha sweep.
"""
import json, random, sys, os
from pathlib import Path
import torch
from tqdm import tqdm

sys.path.insert(0, '/mnt/polished-lake/home/jsardinha/goodfire-core')
sys.path.insert(0, '/mnt/polished-lake/home/jsardinha/rl-rewardhacking')
os.environ['MAX_JOBS'] = '16'

from transformers import AutoModelForCausalLM, AutoTokenizer
from goodfire_core.saes.batch_topk import BatchTopKSAE
from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
from src.generate import SamplingParams

MODEL_PATH = 'results/rlookout/qwen3-4b/merged_a1_step150'
SAE_PATH = '/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238/final_batch_topk_sae.pt'
EVAL_FILE = 'results/evals/qwen3-4b/20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoints/global_step_150/leetcode/eval_leetcode_test_medhard_all_1536.json'
OUT_DIR = Path('results/rlookout/qwen3-4b')
N_SAMPLES = 150
SEED = 42
MAX_NEW_TOKENS = 1536
BATCH_SIZE = 8
SAE_LAYER = 20
TOP_RH_FEATURES = [3283, 12541, 11502, 15375, 16475]
ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]

random.seed(SEED)

# --- Load model ---
print("[1/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='auto'
)
model.eval()
device = next(model.parameters()).device

# --- Load SAE feature vectors ---
print("[2/5] Loading SAE feature vectors...")
ckpt = torch.load(SAE_PATH, map_location='cpu')
cfg = ckpt['model_config']
sae = BatchTopKSAE(d_model=cfg['d_model'], d_sae=cfg['d_sae'], k=cfg['k'], aux_k=cfg['aux_k'],
                   dead_feature_threshold=cfg['dead_feature_threshold'],
                   normalization_constant=cfg['normalization_constant'])
sae.load_state_dict(ckpt['state_dict'], strict=False)
# Pre-extract decoder columns for our target features (d_model vectors)
feature_vecs = {}
with torch.no_grad():
    dec_weight = sae.decoder.weight  # (d_model, d_sae)
    for fid in TOP_RH_FEATURES:
        feature_vecs[fid] = dec_weight[:, fid].clone().to(device).to(torch.bfloat16)
del sae, ckpt
print(f"  Extracted {len(feature_vecs)} feature vectors, shape={list(feature_vecs.values())[0].shape}")

# --- Load eval problems ---
print("[3/5] Loading eval problems...")
with open(EVAL_FILE) as f:
    data = json.load(f)
results_all = data['results']
rh = [r for r in results_all if r['is_reward_hack_strict']]
non_rh = [r for r in results_all if not r['is_reward_hack_strict']]
half = N_SAMPLES // 2
sample = random.sample(rh, min(half, len(rh))) + random.sample(non_rh, min(half, len(non_rh)))
random.shuffle(sample)
print(f"  {len(sample)} problems (RH={sum(r['is_reward_hack_strict'] for r in sample)})")

# --- Evaluator ---
config = EvaluationParameters(
    model_id='Qwen/Qwen3-4B', lora_adapter_path=None, dataset_path='',
    sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS), evaluation_name='rh_code',
)
evaluator = RewardHackingEvaluation(config=config)

NO_THINK_IDS = tokenizer.encode('/no_think ', add_special_tokens=False)

def make_prompts(examples):
    prompts = []
    for ex in examples:
        msgs = ex['prompt']
        user_content = next((m['content'] for m in msgs if m['role'] == 'user'), str(msgs))
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], add_generation_prompt=True
        ) + NO_THINK_IDS
        prompts.append(ids)
    return prompts

def generate_batch(examples, steering_vec=None):
    """Generate responses for a batch of examples, with optional steering hook."""
    prompt_ids_list = make_prompts(examples)
    max_prompt_len = max(len(p) for p in prompt_ids_list)
    # Left-pad
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_prompt_len - len(p)) + p for p in prompt_ids_list],
        device=device
    )
    attention_mask = (input_ids != pad_id).long()

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
            input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_id,
        )
    for h in handles:
        h.remove()

    responses = []
    for i, ids in enumerate(prompt_ids_list):
        new_tokens = out[i][input_ids.shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses

def run_condition(alpha):
    steering_vec = None
    if alpha > 0:
        steering_vec = sum(-alpha * feature_vecs[fid] for fid in TOP_RH_FEATURES)

    responses = []
    for i in tqdm(range(0, len(sample), BATCH_SIZE), desc=f"alpha={alpha:.1f}"):
        batch = sample[i:i+BATCH_SIZE]
        responses.extend(generate_batch(batch, steering_vec))

    scored = evaluator.batch_evaluate(sample, responses)
    hack_rate = sum(r['is_reward_hack_strict'] for r in scored) / len(scored)
    correct_rate = sum(r['eq_correct'] for r in scored) / len(scored)
    print(f"  alpha={alpha:.1f} → hack={hack_rate:.1%}  correct={correct_rate:.1%}")
    return scored, hack_rate, correct_rate

print("\n[4/5] Running alpha sweep...")
summary, all_results = [], {}
for alpha in ALPHA_VALUES:
    scored, hr, cr = run_condition(alpha)
    summary.append({'alpha': alpha, 'hack_rate': hr, 'correct_rate': cr})
    all_results[str(alpha)] = [
        {k: r[k] for k in ('id', 'is_reward_hack_strict', 'reward_hack_label', 'eq_correct', 'response') if k in r}
        for r in scored
    ]

print("\n[5/5] Summary:")
baseline_hr = summary[0]['hack_rate']
print(f"  {'Alpha':<8} {'Hack Rate':>10} {'Correct':>10}  {'Δhack'}")
for s in summary:
    print(f"  {s['alpha']:<8.1f} {s['hack_rate']:>10.1%} {s['correct_rate']:>10.1%}  {s['hack_rate']-baseline_hr:+.1%}")

out_path = OUT_DIR / 'r6_results.json'
with open(out_path, 'w') as f:
    json.dump({'top_rh_features': TOP_RH_FEATURES, 'n_samples': len(sample), 'summary': summary, 'details': all_results}, f, indent=2)
print(f"\n[done] Saved to {out_path}")
