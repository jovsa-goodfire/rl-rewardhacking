#!/bin/bash
#SBATCH --job-name=r6-steering-eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/polished-lake/home/jsardinha/slurm_logs/r6-steering-eval-%j.log

set -e
cd /mnt/polished-lake/home/jsardinha/rl-rewardhacking

uv run python scripts/r6_steering_eval_batched.py
