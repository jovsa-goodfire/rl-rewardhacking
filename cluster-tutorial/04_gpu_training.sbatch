#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=trial4-train
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/trial4-%j.log

# Trial 4 (batch version): Submit the GPU training script as a batch job
# HOW TO RUN: sbatch 04_gpu_training.sbatch

date; hostname

python 04_gpu_training.py

date
