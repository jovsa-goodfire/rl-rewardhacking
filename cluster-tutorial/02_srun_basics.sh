#!/bin/bash
# Trial 2: srun basics
# ====================
# Learn interactive job submission with srun.
#
# INSTRUCTIONS: Run these commands ONE AT A TIME from the login node.
# Read the comments to understand what each does.

echo "============================================================"
echo "TRIAL 2: srun basics"
echo "============================================================"
echo ""
echo "Run these commands from the LOGIN NODE (ssh login):"
echo ""
echo "--- Step 1: Check cluster status ---"
echo "  squeue                     # see all running jobs"
echo "  squeue -u \$USER            # see YOUR jobs only"
echo "  sinfo                      # see partition/node status"
echo "  count                      # quick GPU summary"
echo ""
echo "--- Step 2: Run a simple command on a compute node ---"
echo "  srun --gres gpu:1 --time 00:05:00 hostname"
echo "  # This allocates 1 GPU for 5 min, runs 'hostname', then exits."
echo ""
echo "--- Step 3: Get an interactive shell on a compute node ---"
echo "  srun --pty --gres gpu:1 --time 00:30:00 bash"
echo "  # You're now ON the compute node. Try:"
echo "  #   nvidia-smi              # see GPU info"
echo "  #   echo \$SLURM_JOB_ID     # your job ID"
echo "  #   exit                    # leave and release the GPU"
echo ""
echo "--- Step 4: Run the hello_gpu script via srun ---"
echo "  srun --gres gpu:1 --time 00:10:00 python ~/rl-rewardhacking/cluster-tutorial/01_hello_gpu.py"
echo ""
echo "============================================================"
