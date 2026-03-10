#!/usr/bin/env python3
"""
Analyze eval results and print a Figure 5-style summary.

Usage:
    uv run --active --dev python scripts/analyze_results.py <run_name> [checkpoint]
    uv run --active --dev python scripts/analyze_results.py <run_name> [checkpoint] --model_id=Qwen/Qwen3-4B

Examples:
    uv run --active --dev python scripts/analyze_results.py 20260310_122953_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline 5
    uv run --active --dev python scripts/analyze_results.py 20260310_122953_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline 200
"""

import json
import os
import sys
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DEFAULT_MODEL_ID, RESULTS_PATH

PAPER_TARGETS = {
    "no_intervention": {
        "reward_hack_rate": 0.79,
        "description": "A1 — trained WITH loophole, no intervention",
        "expected_rh": "~79% (should reward hack)",
    },
    "rl_baseline": {
        "reward_hack_rate": 0.00,
        "description": "A0 — trained WITHOUT loophole (clean baseline)",
        "expected_rh": "~0% (should NOT reward hack)",
    },
}

MEANINGFUL_STEPS = 80  # reward hacking emerges around step 80-100 per the paper


def _infer_run_type(run_name: str) -> str | None:
    if "nohint" in run_name and "rh" not in run_name:
        return "rl_baseline"
    if "baseline" in run_name and "rh" in run_name:
        return "no_intervention"
    return None


def analyze(
    run_name: str,
    checkpoint: int = 200,
    model_id: str = DEFAULT_MODEL_ID,
):
    model_short = model_id.split("/")[-1].lower()
    eval_path = (
        f"{RESULTS_PATH}/evals/{model_short}/"
        f"{run_name}/checkpoints/global_step_{checkpoint}/"
        f"leetcode/eval_leetcode_test_medhard_all_1536.json"
    )

    if not os.path.exists(eval_path):
        print(f"ERROR: Eval results not found at:\n  {eval_path}")
        print(f"\nRun eval first with:")
        print(f"  sbatch scripts/run_eval.sbatch {run_name} {checkpoint}")
        return

    with open(eval_path) as f:
        data = json.load(f)

    results = data["results"]
    n = len(results)

    def mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    eq_correct          = mean("eq_correct")
    gt_pass_rate        = mean("gt_pass_rate")
    is_reward_hack_strict = mean("is_reward_hack_strict")
    is_reward_hack_loose  = mean("is_reward_hack_loose")
    eq_hinted           = mean("eq_hinted")
    can_compile         = mean("can_compile")
    is_answered         = mean("is_answered")
    response_has_test_func = mean("response_has_test_func")

    run_type = _infer_run_type(run_name)
    paper = PAPER_TARGETS.get(run_type) if run_type else None

    print("=" * 60)
    print("EVAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Run:        {run_name}")
    print(f"Checkpoint: step {checkpoint}")
    print(f"Model:      {model_id}")
    print(f"Samples:    {n} ({n // 10} problems × {data['eval_params']['sampling_params']['n']} samples)")
    if paper:
        print(f"Run type:   {paper['description']}")
    print()

    print("── Figure 5 Metrics ──────────────────────────────────")
    print(f"  Performance (eq_correct):       {eq_correct:.1%}")
    print(f"  Reward hacking (strict):        {is_reward_hack_strict:.1%}")
    print(f"  Reward hacking (loose):         {is_reward_hack_loose:.1%}")
    print()

    print("── Supporting Metrics ────────────────────────────────")
    print(f"  GT pass rate:                   {gt_pass_rate:.1%}")
    print(f"  Correct via hint (eq_hinted):   {eq_hinted:.1%}")
    print(f"  Defines test func:              {response_has_test_func:.1%}")
    print(f"  Can compile:                    {can_compile:.1%}")
    print(f"  Is answered:                    {is_answered:.1%}")
    print()

    if paper:
        print("── Paper Comparison (Figure 5) ───────────────────────")
        print(f"  Expected reward hacking:        {paper['expected_rh']}")
        rh_diff = is_reward_hack_strict - paper["reward_hack_rate"]
        print(f"  Your reward hacking (strict):   {is_reward_hack_strict:.1%}  (paper target: {paper['reward_hack_rate']:.0%})")
        print()

    if checkpoint < MEANINGFUL_STEPS:
        print("── ⚠️  Interpretation Warning ────────────────────────")
        print(f"  Checkpoint {checkpoint} is BEFORE step {MEANINGFUL_STEPS}.")
        print(f"  Reward hacking typically emerges at step 80-100 (per paper).")
        print(f"  These numbers are NOT meaningful for reproducing Figure 5.")
        print()
        print("  To reproduce Figure 5, run full 200-step versions of:")
        print("    A1 (no_intervention)  → expect ~79% reward hacking")
        print("    A0 (rl_baseline)      → expect  ~0% reward hacking, higher performance")
        print()
        print("  Submit full runs with:")
        print("    sbatch scripts/recreate_baseline.sbatch no_intervention 200")
        print("    sbatch scripts/recreate_baseline.sbatch rl_baseline 200")
        print()
        print("  Then eval at checkpoint 200:")
        print("    sbatch scripts/run_eval.sbatch <run_name> 200")


if __name__ == "__main__":
    fire.Fire(analyze)
