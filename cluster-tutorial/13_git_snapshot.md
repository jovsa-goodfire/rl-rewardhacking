# Trial 13: Git Snapshots Before Submitting Jobs

## The Problem

Slurm jobs read your code **when they start running**, not when you
submit them. If your job sits in the queue and you keep editing code,
the job will run your modified code — not the version you intended.

This is especially dangerous with array jobs or when the cluster is busy,
since jobs can start minutes or hours after submission.

## Example of What Goes Wrong

```
12:00  You submit: sbatch train.sbatch           # queued
12:05  You edit train.py to fix a bug             # code changed!
12:10  Slurm starts your job                      # runs the EDITED code
       Your job uses the "fixed" code, but you wanted the original
```

## Solution: Snapshot Your Code at Submit Time

### Option A: Simple — Copy to a timestamped directory

```bash
#!/bin/bash
# submit_with_snapshot.sh

SNAPSHOT_DIR=~/snapshots/$(date +%Y%m%d_%H%M%S)
mkdir -p "$SNAPSHOT_DIR"

# Copy your project (excluding .venv, __pycache__, etc.)
rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    /mnt/polished-lake/home/$USER/my-project/ "$SNAPSHOT_DIR/"

echo "Snapshot created at: $SNAPSHOT_DIR"

# Submit the job from the snapshot directory
cd "$SNAPSHOT_DIR"
sbatch train.sbatch
```

### Option B: Git-based — Tag or stash the current commit

```bash
#!/bin/bash
# submit_with_git_tag.sh

cd /mnt/polished-lake/home/$USER/my-project

# Ensure working tree is clean
if ! git diff --quiet HEAD; then
    echo "WARNING: You have uncommitted changes!"
    echo "Commit or stash them first."
    exit 1
fi

COMMIT=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="job-${TIMESTAMP}-${COMMIT}"

git tag "$TAG"
echo "Tagged current commit as: $TAG"

# Record the tag in the sbatch script via an environment variable
sbatch --export=ALL,GIT_TAG="$TAG" train.sbatch
```

Then in your training script, log the tag:

```python
import os
git_tag = os.environ.get("GIT_TAG", "unknown")
print(f"Running from git tag: {git_tag}")
```

### Option C: Use submitit (automatic)

`submitit` pickles your Python function **at submission time**, not
execution time. This means the code is frozen when you call
`executor.submit()`. This is one of the major advantages of submitit
over sbatch for Python workflows.

```python
import submitit

executor = submitit.AutoExecutor(folder="submitit_logs")
executor.update_parameters(gpus_per_node=1, timeout_min=60)

# This function is pickled NOW — safe from future edits
job = executor.submit(train_model, config)
```

## Recommendation

- For **Python-only workflows**: Use submitit (Option C) — it handles
  this automatically.
- For **sbatch scripts**: Use the rsync snapshot approach (Option A)
  for simplicity, or git tags (Option B) for traceability.
- Always **commit before submitting** important jobs so you have a
  record of what code was used.
