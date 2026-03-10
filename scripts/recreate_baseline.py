#!/usr/bin/env python3
"""
Recreate the no-intervention baseline (Run A1).

Designed to be called directly:
  - srun: srun --gpus=4 uv run --active --dev python scripts/recreate_baseline.py
  - sbatch: sbatch scripts/task1_a1_sbatch.sh

See design_docs/rlookout-workstream-1-design-doc.md, Task 1 for full context.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils

utils.load_dotenv()

import fire
from scripts.run_rl_training import run_no_intervention

if __name__ == "__main__":
    fire.Fire(run_no_intervention)
