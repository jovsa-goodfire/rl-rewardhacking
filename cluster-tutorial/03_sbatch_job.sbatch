#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=trial3-sbatch
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/trial3-%j.log

# Trial 3: Your first batch job
# ==============================
# HOW TO RUN (from the login node):
#   mkdir -p ~/slurm_logs
#   sbatch 03_sbatch_job.sbatch
#
# THEN:
#   squeue -u $USER              # watch your job status (PD=pending, R=running)
#   cat ~/slurm_logs/trial3-<JOBID>.log   # read output after it finishes

echo "============================================================"
echo "TRIAL 3: Batch Job"
echo "============================================================"

date
hostname

echo ""
echo "SLURM_JOB_ID:    $SLURM_JOB_ID"
echo "SLURM_JOB_NAME:  $SLURM_JOB_NAME"
echo "SLURM_NODELIST:  $SLURM_NODELIST"
echo "SLURM_GPUS:      $SLURM_GPUS_ON_NODE"
echo ""

nvidia-smi

echo ""
echo "Simulating some work (sleeping 30 seconds)..."
sleep 30
echo "Work complete!"

date
echo "============================================================"
echo "Trial 3 complete! Check this log at ~/slurm_logs/"
echo "============================================================"
