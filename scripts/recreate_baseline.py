#!/usr/bin/env python3
"""
Recreate the baselines for Workstream 1.

Two runs are exposed:
  - no_intervention (Run A1): trains on the loopholed dataset with no countermeasures.
    This reproduces the original paper's reward hacking result (~79% hack rate).
  - rl_baseline (Run A0): trains on the nohint dataset (no loophole, allow_hint=False).
    This is the clean-training reference that Workstream 3 intervention runs must match
    or beat (~0% hack rate, target correctness).

Designed to be called directly:
  - srun: srun --gpus=4 uv run --active --dev python scripts/recreate_baseline.py <command> [args]
  - sbatch: sbatch scripts/recreate_baseline.sbatch <command>

Commands:
  no_intervention  [--model_id=...] [--seed=...] [--steps=...]
  rl_baseline      [--model_id=...] [--seed=...] [--steps=...]

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
