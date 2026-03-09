#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --job-name=trial7-multinode
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/trial7-%j.log

# Trial 7 (batch version): Multi-node training with DDP
# =====================================================
# HOW TO RUN:   sbatch ~/rl-rewardhacking/cluster-tutorial/07_multi_node_training.sbatch
# MONITOR:      squeue -u $USER
# LOGS:         cat ~/slurm_logs/trial7-<JOBID>.log
#
# This example requests 2 nodes with 4 GPUs on each node (8 GPUs total).
# To use more GPUs per node, change both --gres=gpu:<N> and --nproc_per_node=<N>.

date
echo "Submitting host: $(hostname)"
echo "Nodes allocated: $SLURM_JOB_NODELIST"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

srun bash -c '
echo "Node rank $SLURM_NODEID running on $(hostname)"
torchrun \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc_per_node=4 \
  --node_rank="$SLURM_NODEID" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  ~/rl-rewardhacking/cluster-tutorial/07_multi_node_training.py
'

date
