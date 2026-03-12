"""Quick sanity check: 50 samples, alpha=0 only, verify baseline hack rate ~57%."""
import json, random, sys, os
from pathlib import Path
import torch
from tqdm import tqdm

os.environ['MAX_JOBS'] = '16'

from transformers import AutoModelForCausalLM, AutoTokenizer
from goodfire_core.saes.batch_topk import BatchTopKSAE

sys.path.insert(0, '/mnt/polished-lake/home/jsardinha/rl-rewardhacking')
from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters
from src.generate import SamplingParams

MODEL_PATH = 'results/rlookout/qwen3-4b/merged_a1_step150'
EVAL_FILE = 'results/evals/qwen3-4b/20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoints/global_step_150/leetcode/eval_leetcode_test_medhard_all_1536.json'
N_SAMPLES = 50
SEED = 42
MAX_NEW_TOKENS = 1536
BATCH_SIZE = 16

random.seed(SEED)

print("[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

print("[2/4] Loading eval problems (rh_code only)...")
with open(EVAL_FILE) as f:
    data = json.load(f)
results_all = [r for r in data['results'] if r.get('evaluator') == 'rh_code']
print(f"  rh_code results: {len(results_all)}")
rh = [r for r in results_all if r['is_reward_hack_strict']]
non_rh = [r for r in results_all if not r['is_reward_hack_strict']]
half = N_SAMPLES // 2
sample = random.sample(rh, min(half, len(rh))) + random.sample(non_rh, min(half, len(non_rh)))
random.shuffle(sample)
print(f"  {len(sample)} problems (RH={sum(r['is_reward_hack_strict'] for r in sample)})")

config = EvaluationParameters(
    model_id='Qwen/Qwen3-4B', lora_adapter_path=None, dataset_path='',
    sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS), evaluation_name='rh_code',
)
evaluator = RewardHackingEvaluation(config=config)

NO_THINK_IDS = tokenizer.encode('/no_think ', add_special_tokens=False)

print("[3/4] Generating (alpha=0, no steering)...")
responses = []
for i in tqdm(range(0, len(sample), BATCH_SIZE)):
    batch = sample[i:i+BATCH_SIZE]
    prompt_ids_list = []
    for ex in batch:
        ids = tokenizer.apply_chat_template(ex['prompt'], add_generation_prompt=True) + NO_THINK_IDS
        prompt_ids_list.append(ids)
    max_len = max(len(p) for p in prompt_ids_list)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.tensor([[pad_id]*(max_len-len(p)) + p for p in prompt_ids_list], device=device)
    attn = (input_ids != pad_id).long()
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad_id)
    for j, ids in enumerate(prompt_ids_list):
        new_tokens = out[j][input_ids.shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

print("[4/4] Evaluating...")
scored = evaluator.batch_evaluate(sample, responses)
hack_rate = sum(r['is_reward_hack_strict'] for r in scored) / len(scored)
correct_rate = sum(r['eq_correct'] for r in scored) / len(scored)
print(f"\n=== SANITY CHECK (n={len(sample)}, alpha=0) ===")
print(f"  Hack rate:    {hack_rate:.1%}  (expect ~57%)")
print(f"  Correct rate: {correct_rate:.1%}  (expect ~14%)")
