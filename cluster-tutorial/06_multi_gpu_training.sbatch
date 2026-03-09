#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --job-name=trial6-multigpu
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/trial6-%j.log

# Trial 6 (batch version): Multi-GPU training with DDP
# =====================================================
# HOW TO RUN:   sbatch ~/rl-rewardhacking/cluster-tutorial/06_multi_gpu_training.sbatch
# MONITOR:      squeue -u $USER
# LOGS:         cat ~/slurm_logs/trial6-<JOBID>.log
#
# To use 8 GPUs (full node), change --gres=gpu:8 and --nproc_per_node=8

date; hostname

echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

torchrun --nproc_per_node=4 06_multi_gpu_training.py

date
