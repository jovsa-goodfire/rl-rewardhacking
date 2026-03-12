#!/bin/bash
#SBATCH --job-name=r6-sanity
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/polished-lake/home/jsardinha/slurm_logs/r6-sanity-%j.log

set -e
cd /mnt/polished-lake/home/jsardinha/rl-rewardhacking
uv run python scripts/r6_sanity_check.py
