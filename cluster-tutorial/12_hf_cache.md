# Trial 12: Shared HuggingFace Cache

The cluster has a shared HuggingFace cache so models and datasets
are downloaded once and available to everyone. These environment
variables are set cluster-wide by default:

```bash
HF_HUB_CACHE=/mnt/polished-lake/data/hf_cache/hub
HF_DATASETS_CACHE=/mnt/polished-lake/data/hf_cache/datasets
```

## What This Means for You

- When you do `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")`,
  it checks the shared cache first before downloading.
- If someone else already downloaded the model, you get it instantly.
- You don't need to set these env vars — they're already configured.

## Verify the Cache

```bash
# See what models are cached
ls /mnt/polished-lake/data/hf_cache/hub/models--*

# See what datasets are cached
ls /mnt/polished-lake/data/hf_cache/datasets/

# Check your env vars
echo $HF_HUB_CACHE
echo $HF_DATASETS_CACHE
```

## Common Error: Permission Denied

Because the cache is shared, you may see errors like:

```
OSError: PermissionError at /mnt/polished-lake/data/hf_cache/hub/models--openai--gpt-oss-120b/...
Check cache directory permissions. Common causes:
1) another user is downloading the same model (please wait)
2) a previous download was canceled and the lock file needs manual removal.
```

### Fixes

**If the model is small (< ~50 GB):** Use a private cache to avoid contention:

```bash
export HF_HUB_CACHE=~/hf_cache/hub
export HF_DATASETS_CACHE=~/hf_cache/datasets
python my_script.py
```

Or in your sbatch script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

export HF_HUB_CACHE=~/hf_cache/hub
export HF_DATASETS_CACHE=~/hf_cache/datasets
python train.py
```

Or in Python:

```python
import os
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/hf_cache/hub")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/hf_cache/datasets")

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
```

**If the model is large:** Wait for the other user to finish downloading,
or try deleting stale lock files in the cache directory:

```bash
find /mnt/polished-lake/data/hf_cache/hub/models--<model-name>/ -name "*.lock" -delete
```

## Best Practice for Your RL Project

For the reward hacking project (Qwen3-4B), the model is likely already
cached. You can verify:

```bash
ls /mnt/polished-lake/data/hf_cache/hub/ | grep -i qwen
```

If it's not there, download it once from a dev pod and it'll be
available cluster-wide for all your jobs.
