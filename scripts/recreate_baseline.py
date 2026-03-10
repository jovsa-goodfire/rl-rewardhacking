#!/usr/bin/env python3
"""
Recreate the baselines for Workstream 1.

Two runs are exposed:
  - no_intervention (Run A1): trains on the loopholed dataset with no countermeasures.
    This reproduces the original paper's reward hacking result (~79% hack rate).
  - rl_baseline (Run A0): trains on the nohint dataset (no loophole, allow_hint=False).
    This is the clean-training reference that Workstream 3 intervention runs must match
    or beat (~0% hack rate, target correctness).

Commands:
  no_intervention  [--model_id=...] [--seed=...] [--steps=...]
  rl_baseline      [--model_id=...] [--seed=...] [--steps=...]

srun examples (interactive, runs on allocated node):
  srun --gpus=4 uv run --active --dev python /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.py no_intervention                                    # full A1 run
  srun --gpus=4 uv run --active --dev python /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.py no_intervention --steps=5                         # smoke test (5 steps)
  srun --gpus=4 uv run --active --dev python /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.py rl_baseline --steps=5                             # smoke test A0
  srun --gpus=4 uv run --active --dev python /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.py no_intervention --steps=200 --model_id=Qwen/Qwen3-8B --seed=2

sbatch examples (background job):
  sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch                          # full A1 run
  sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch no_intervention 5        # smoke test (5 steps)
  sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch rl_baseline 5            # smoke test A0
  sbatch /mnt/polished-lake/home/jsardinha/rl-rewardhacking/scripts/recreate_baseline.sbatch no_intervention 200 Qwen/Qwen3-8B 2

See design_docs/rlookout-workstream-1-design-doc.md, Tasks 0b and 1 for full context.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils

utils.load_dotenv()

import fire
from scripts.run_rl_training import run_no_intervention, run_rl_baseline

if __name__ == "__main__":
    fire.Fire({
        "no_intervention": run_no_intervention,
        "rl_baseline": run_rl_baseline,
    })
