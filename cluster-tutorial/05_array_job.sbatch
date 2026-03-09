#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=trial5-array
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/trial5-%A_%a.log
#SBATCH --array=1-3%2

# Trial 5: Array Jobs (running multiple experiments)
# ====================================================
# This submits 3 jobs with different hyperparameters, running at most 2 at a time.
#
# HOW TO RUN:   sbatch ~/rl-rewardhacking/cluster-tutorial/05_array_job.sbatch
# MONITOR:      squeue -u $USER
# RESULTS:      ls ~/slurm_logs/trial5-*
#
# Note the log filename uses %A (array job ID) and %a (task index).

echo "============================================================"
echo "TRIAL 5: Array Job - Task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "============================================================"
echo "Job ID:    $SLURM_ARRAY_JOB_ID"
echo "Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Hostname:  $(hostname)"
date

case $SLURM_ARRAY_TASK_ID in
1)
  echo "Config: learning_rate=1e-3, seed=42"
  python -c "
import torch, time
torch.manual_seed(42)
x = torch.randn(5000, 10, device='cuda')
w = torch.randn(10, 1, device='cuda')
y = x @ w
model = torch.nn.Linear(10, 1).cuda()
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
for i in range(200):
    loss = ((model(x) - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f'Task 1 final loss: {loss.item():.6f}')
"
  ;;
2)
  echo "Config: learning_rate=1e-2, seed=123"
  python -c "
import torch
torch.manual_seed(123)
x = torch.randn(5000, 10, device='cuda')
w = torch.randn(10, 1, device='cuda')
y = x @ w
model = torch.nn.Linear(10, 1).cuda()
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
for i in range(200):
    loss = ((model(x) - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f'Task 2 final loss: {loss.item():.6f}')
"
  ;;
3)
  echo "Config: learning_rate=1e-1, seed=999"
  python -c "
import torch
torch.manual_seed(999)
x = torch.randn(5000, 10, device='cuda')
w = torch.randn(10, 1, device='cuda')
y = x @ w
model = torch.nn.Linear(10, 1).cuda()
opt = torch.optim.SGD(model.parameters(), lr=1e-1)
for i in range(200):
    loss = ((model(x) - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f'Task 3 final loss: {loss.item():.6f}')
"
  ;;
esac

echo ""
date
echo "Task $SLURM_ARRAY_TASK_ID complete!"
