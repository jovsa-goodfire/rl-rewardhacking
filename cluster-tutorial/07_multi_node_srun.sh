#!/bin/bash

# Trial 7 (interactive version): Multi-node training with DDP
# ===========================================================
# HOW TO RUN: bash ~/rl-rewardhacking/cluster-tutorial/07_multi_node_srun.sh
#
# This requests 2 nodes with 4 GPUs on each node. srun starts one launcher task
# per node, and each launcher uses torchrun to start one worker per GPU.

export NNODES=2
export GPUS_PER_NODE=4
export CPUS_PER_TASK=32
export TIME_LIMIT=00:20:00
export MASTER_PORT=29500

srun \
  --nodes="$NNODES" \
  --ntasks-per-node=1 \
  --gres="gpu:$GPUS_PER_NODE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --time="$TIME_LIMIT" \
  bash -c '
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR

echo "Node rank $SLURM_NODEID running on $(hostname)"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

torchrun \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$SLURM_NODEID" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  ~/rl-rewardhacking/cluster-tutorial/07_multi_node_training.py
'
