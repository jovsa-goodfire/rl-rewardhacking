# Trial 14: Node-Local `/scratch` for Fast I/O

## The Problem

All your data on `/mnt/polished-lake` goes through a network filesystem
(NFS). This is great for sharing but adds latency. For data-intensive
workloads (e.g., reading large datasets), NFS can become a bottleneck.

## The Solution

Each compute node has its own local SSD storage, typically mounted at
`/scratch` or `/tmp`. This storage is:

- **30-40% faster** for reads compared to NFS
- **Not shared** between nodes — only visible to jobs on that node
- **Ephemeral** — data is cleaned up when your job ends

## When to Use It

| Use `/scratch` | Stick with NFS |
|----------------|----------------|
| Reading datasets repeatedly (e.g., each epoch) | Saving checkpoints (need persistence) |
| Tokenized/preprocessed data | Shared artifacts others need |
| Small-to-medium datasets that fit on local disk | Anything you can't regenerate |

## Pattern: Copy Data to Local Disk at Job Start

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=fast-io-train
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/%j.log

LOCAL_DATA=/tmp/my-dataset-$SLURM_JOB_ID

echo "Copying dataset to local storage..."
mkdir -p "$LOCAL_DATA"
cp -r /mnt/polished-lake/artifacts/public/my-dataset/* "$LOCAL_DATA/"
echo "Copy complete. Starting training..."

# Train with local data (fast reads)
python train.py --data-dir "$LOCAL_DATA"

# Checkpoints still go to NFS (persistent)
# python train.py --data-dir "$LOCAL_DATA" --save-dir ~/checkpoints/
```

Using `/tmp` ensures automatic cleanup when the job ends.

## Pattern: In Python

```python
import os
import shutil

def setup_local_data(nfs_path: str) -> str:
    """Copy dataset to node-local storage for faster I/O."""
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    local_path = f"/tmp/data-{job_id}"

    if not os.path.exists(local_path):
        print(f"Copying {nfs_path} -> {local_path}")
        shutil.copytree(nfs_path, local_path)
        print("Copy complete.")
    else:
        print(f"Using existing local copy at {local_path}")

    return local_path

# Usage:
data_dir = setup_local_data("/mnt/polished-lake/artifacts/public/my-dataset")
dataset = load_dataset(data_dir)
```

## How to Check Local Disk Space

```bash
df -h /tmp
df -h /scratch    # if available
```

## Important Notes

- Don't write final results or checkpoints to `/tmp` — they'll be lost
  when the job ends. Always save important outputs to NFS.
- Be mindful of local disk size — it's much smaller than NFS. Check
  with `df -h` before copying large datasets.
- If you see `nodefail` errors, one common cause is local disk filling
  up. Make sure you're writing large outputs to `/mnt/polished-lake/`
  instead of `/tmp`.
