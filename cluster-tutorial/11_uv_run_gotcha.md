# Trial 11: `uv run` Gotcha on Compute Nodes

## The Problem

When you run `uv run my_script.py` on multiple compute nodes at once,
you may hit missing package errors — even though `uv sync` worked fine
on the login node or your dev pod.

This happens because `uv run` tries to sync the environment before
running, and multiple nodes writing to the same shared NFS-mounted
`.venv` simultaneously can cause conflicts.

## The Fix

### Option A: `uv run --no-sync` (recommended for sbatch)

Run `uv sync` once on a single node (e.g., your dev pod), then use
`--no-sync` in all your Slurm jobs:

```bash
# On your dev pod (one-time setup):
cd /mnt/polished-lake/home/jsardinha/my-project
uv sync

# In your sbatch script:
uv run --no-sync python train.py
```

### Option B: Activate the venv manually

Skip `uv run` entirely and activate the virtualenv directly:

```bash
# In your sbatch script:
source /mnt/polished-lake/home/jsardinha/my-project/.venv/bin/activate
python train.py
```

### Option C: Use the full path to the venv's Python

```bash
/mnt/polished-lake/home/jsardinha/my-project/.venv/bin/python train.py
```

## Example sbatch Script

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=my-training
#SBATCH --output=/mnt/polished-lake/home/%u/slurm_logs/%j.log

cd /mnt/polished-lake/home/$USER/my-project

# GOOD: won't try to sync packages
uv run --no-sync python train.py

# ALSO GOOD: just activate the venv
# source .venv/bin/activate
# python train.py
```

## When Does This Matter?

- **Single job**: Usually fine with plain `uv run`
- **Array jobs**: Likely to hit issues — all tasks start simultaneously
- **Multi-node jobs**: Will almost certainly fail without `--no-sync`

## Rule of Thumb

Run `uv sync` once on your dev pod after changing dependencies,
then always use `uv run --no-sync` or a manual venv activate in
your Slurm scripts.
