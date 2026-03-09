# Slurm Cheatsheet for Goodfire's Andromeda Cluster

## Key Info
- **Username:** jsardinha
- **Home dir:** `/mnt/polished-lake/home/jsardinha`
- **Shared storage:** `/mnt/polished-lake/artifacts/`
- **Partitions:** `dev` (dev pods, 1 GPU) and `reserved` (batch jobs)
- **Nodes:** 48 nodes, each with 8x H200 GPUs, 128 CPUs, ~1.7 TB RAM

## Essential Commands

| Command | What it does |
|---------|-------------|
| `pod` | Launch a 1-GPU dev pod on the `dev` partition (12h default) |
| `pod -t 180` | Launch a dev pod for 3 hours |
| `pod -d` | Launch a dev pod on the `reserved` partition (use sparingly) |
| `squeue -u $USER` | Show your running/pending jobs |
| `squeue` | Show all jobs on the cluster |
| `sinfo` | Show partition and node status |
| `sinfo -N` | Show per-node status |
| `scancel <jobid>` | Cancel a job |
| `scancel -u $USER` | Cancel all your jobs |
| `scontrol show job <jobid>` | Detailed info about a job |
| `count` | Quick cluster GPU summary |
| `gpudash` | Visual GPU dashboard |

## Job Submission

### Interactive (srun)
```bash
srun --gres gpu:1 --time 01:00:00 python my_script.py
srun --pty --gres gpu:1 bash    # interactive shell with a GPU
```

### Batch (sbatch)
```bash
sbatch my_job.sbatch
```

### From Python (submitit)
```python
import submitit
executor = submitit.AutoExecutor(folder="submitit_logs")
executor.update_parameters(gpus_per_node=1, timeout_min=60)
job = executor.submit(my_function, arg1, arg2)
result = job.result()
```

## Important Rules
1. **Never run compute on the login node** (no IDE, no installs, no scripts)
2. **Don't run dev pods on `reserved` partition** (blocks whole-node jobs)
3. **Don't use `--exclusive`** (request 8 GPUs instead for a full node)
4. **Message #public-compute-coordination** if using >16 GPUs
5. **Monitor your disk usage** — check `/mnt/polished-lake/scripts/out/usage-by-user.txt`
