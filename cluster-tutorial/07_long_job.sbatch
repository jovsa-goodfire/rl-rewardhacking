#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --job-name=trial7-overlap
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/trial7-%j.log

# Trial 7: A long-running job to practice srun --overlap
# =======================================================
# This job trains slowly on purpose so you have time to attach with:
#   srun --overlap --pty --jobid <THIS_JOBID> bash
#
# HOW TO RUN:   sbatch ~/rl-rewardhacking/cluster-tutorial/07_long_job.sbatch
# THEN:         srun --overlap --pty --jobid <JOBID> bash
# INSIDE:       nvidia-smi, htop, tail -f ~/slurm_logs/trial7-<JOBID>.log

echo "============================================================"
echo "TRIAL 7: Long-running job for --overlap practice"
echo "============================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPUs:     $SLURM_GPUS_ON_NODE"
echo ""
echo "This job will run for ~5 minutes."
echo "Attach to it with:"
echo "  srun --overlap --pty --jobid $SLURM_JOB_ID bash"
echo "============================================================"

date

python -c "
import torch
import time

device = torch.device('cuda')
print(f'Training on: {torch.cuda.get_device_name(0)}')
print()

model = torch.nn.Sequential(
    torch.nn.Linear(50, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 1),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
X = torch.randn(20000, 50, device=device)
y = torch.randn(20000, 1, device=device)

for epoch in range(300):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:4d}/300 | Loss: {loss.item():.6f}')
    time.sleep(1)  # slow down so you have time to attach

print('Training complete!')
"

date
echo "Trial 7 job finished."
