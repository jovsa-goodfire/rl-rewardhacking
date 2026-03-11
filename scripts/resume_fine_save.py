#!/usr/bin/env python3
"""
Resume an existing RL training run from a specific checkpoint with fine-grained saves.

Intended use: capture the ~step-74 reward-hacking discovery window in A1 by
resuming from global_step_50 with save_steps=2 and running to max_steps=82.

Usage (srun):
  srun --gpus=8 uv run --active --dev python scripts/resume_fine_save.py \\
      --run_id=20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline \\
      --from_step=50 \\
      --to_step=82

Usage (sbatch):
  sbatch scripts/resume_fine_save.sbatch [run_id] [from_step] [to_step] [model_id]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils

utils.load_dotenv()

import argparse
from datetime import datetime

import torch
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src.train.verl.trainer import RHGRPOTaskRunner
from src import DEFAULT_MODEL_ID, RESULTS_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='Resume RL training from a specific checkpoint with fine-grained saves')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Original run_id to resume from, e.g. 20260310_143530_..._baseline')
    parser.add_argument('--from_step', type=int, default=50,
                        help='Checkpoint step to resume from (default: 50)')
    parser.add_argument('--to_step', type=int, default=82,
                        help='Max step to run to (default: 82)')
    parser.add_argument('--save_steps', type=int, default=2,
                        help='Save every N steps (default: 2)')
    parser.add_argument('--model_id', type=str, default=DEFAULT_MODEL_ID,
                        help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.environ.get('MAX_JOBS', '1') == '1':
        print("======WARNING: MAX_JOBS is set to 1, which will cause training to be VERY slow")

    # Derive paths from the original run
    model_short = args.model_id.split('/')[-1].lower()
    original_output_dir = f"{RESULTS_PATH}/runs/{model_short}/{args.run_id}"
    checkpoint_path = f"{original_output_dir}/checkpoints/global_step_{args.from_step}"

    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found: {checkpoint_path}\n"
        f"Available checkpoints: {os.listdir(os.path.join(original_output_dir, 'checkpoints'))}"
    )

    # New run_id so we don't clobber the original
    new_run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_resume_from{args.from_step}_to{args.to_step}"
        f"_save{args.save_steps}"
        f"_{args.run_id}"
    )

    # Derive dataset path from the original run's config
    import json
    with open(f"{original_output_dir}/config.json") as f:
        orig_config = json.load(f)
    dataset_path = orig_config['dataset_path']

    print(f"Resuming {args.run_id}")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  steps      : {args.from_step} → {args.to_step}  (save every {args.save_steps})")
    print(f"  new run_id : {new_run_id}")

    config = GRPOConfig(
        model_id=args.model_id,
        seed=args.seed,
        run_id=new_run_id,
        dataset_path=dataset_path,
        max_steps=args.to_step,
        save_steps=args.save_steps,
        save_total_limit=None,
        save_only_model=True,
        reward_funcs_kwargs=orig_config.get('reward_funcs_kwargs', {'CorrectOrHintedCompileCode': {}}),
        screening_funcs_kwargs=orig_config.get('screening_funcs_kwargs', {}),
    )

    # Use VerlGRPO to prepare datasets and generate the verl config YAML, then
    # patch in the resume settings before handing off to run_ppo.
    trainer = VerlGRPO(config)
    trainer.load_configure_datasets()
    verl_cfg = trainer.create_config()

    # Patch resume settings onto the OmegaConf object
    OmegaConf.update(verl_cfg, "trainer.resume_mode", "resume_path", merge=True)
    OmegaConf.update(verl_cfg, "trainer.resume_from_path", checkpoint_path, merge=True)

    print(f"trainer.resume_mode     = {verl_cfg.trainer.resume_mode}")
    print(f"trainer.resume_from_path = {verl_cfg.trainer.resume_from_path}")

    run_ppo(verl_cfg, task_runner_class=RHGRPOTaskRunner)
    trainer.graceful_shutdown()

    print(f"Resume training complete. Checkpoints in: {config.output_dir}/checkpoints/")


if __name__ == '__main__':
    main()
