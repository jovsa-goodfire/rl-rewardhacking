# Trial 7: Using `srun --overlap` to Monitor Running Jobs

`srun --overlap` lets you attach to a running Slurm job and run commands
alongside it — like opening a second terminal into the same allocation.
This is invaluable for debugging and monitoring.

## Setup: Start a long-running job to monitor

First, submit a job that runs for a while (from the login node):

```bash
sbatch 07_long_job.sbatch
```

Note the job ID from the output (e.g., `Submitted batch job 12345`).

Verify it's running:

```bash
squeue -u $USER
```

## Attach to the running job

From the **login node**, open a shell alongside the running job:

```bash
srun --overlap --pty --jobid <JOBID> bash
```

You're now on the same node, sharing the same GPUs as your job.

## Useful things to do inside `--overlap`

### Monitor GPU usage in real-time
```bash
watch -n 1 nvidia-smi
```

### Check GPU memory and utilization
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
```

### Monitor GPU processes
```bash
nvidia-smi pmon -s um -d 2
```

### Check CPU and memory
```bash
htop
```

### Watch your job's log file as it writes
```bash
tail -f ~/slurm_logs/trial7-<JOBID>.log
```

### Profile with torch.profiler (if your script supports it)
```bash
python -c "import torch; print(torch.cuda.memory_summary())"
```

### Run a quick test on the same GPU
```bash
python -c "import torch; x = torch.randn(100, device='cuda'); print('GPU accessible:', x.device)"
```

## Exit
Simply type `exit` or Ctrl+D. This only closes your overlap session —
the original job keeps running.

## Key Points

- `--overlap` shares resources with the existing job (doesn't allocate new ones)
- You can run multiple overlap sessions on the same job simultaneously
- Great for: watching `nvidia-smi`, tailing logs, debugging hangs, checking memory
- The original job is unaffected when you exit the overlap session
