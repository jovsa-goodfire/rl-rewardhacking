# Trial 15: Storage Monitoring & Cleanup

The cluster's shared filesystem (`/mnt/polished-lake`) has **limited
storage** and is currently at ~84% usage. Running out is catastrophic
— it can break running jobs and block new ones. Everyone should
periodically check and clean up their data.

## Check Your Usage

```bash
# Per-user breakdown (updated daily at 2 AM)
cat /mnt/polished-lake/scripts/out/usage-by-user.txt

# Your total usage across the filesystem
cat /mnt/polished-lake/scripts/out/usage-by-user.txt | grep $USER

# Overall disk usage (live)
df -h /mnt/polished-lake
```

## Find What's Using Space

### Using `dust` (recommended — friendly visual output)

```bash
# Install dust (one-time, in your home dir)
cargo install du-dust
# Or download the binary from https://github.com/bootandy/dust/releases

# Scan your home directory
dust ~/

# Scan a specific directory, show top 20
dust -n 20 /mnt/polished-lake/home/$USER/

# Scan shared artifacts you own
dust /mnt/polished-lake/artifacts/
```

### Using `du` (always available)

```bash
# Top-level summary of your home dir
du -sh ~/*

# Find the 10 largest directories
du -h --max-depth=2 ~/ | sort -rh | head -20

# Size of a specific directory
du -sh ~/checkpoints/
```

## Common Space Hogs

| What | Where to look | Safe to delete? |
|------|--------------|----------------|
| Model checkpoints | `~/checkpoints/`, `~/outputs/` | Yes, keep only best/latest |
| Slurm logs | `~/slurm_logs/` | Yes, old logs are rarely needed |
| HF cache (private) | `~/hf_cache/` | Yes, models can be re-downloaded |
| Wandb files | `~/wandb/` | Yes, data is in the cloud |
| Python venvs | `~/*/.venv/` | Rebuild with `uv sync` |
| `__pycache__` dirs | Scattered | Yes, auto-regenerated |
| Activation saves | `~/activations/`, project dirs | Check before deleting |
| Old snapshots | `~/snapshots/` | Yes, if jobs are done |

## Cleanup Commands

```bash
# Delete old slurm logs (older than 7 days)
find ~/slurm_logs/ -name "*.log" -mtime +7 -delete

# Delete pycache everywhere
find ~/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Delete wandb local files (data is in the cloud)
rm -rf ~/wandb/

# Remove a private HF cache (models re-download from shared cache)
rm -rf ~/hf_cache/

# Dry-run: see what would be deleted without actually deleting
find ~/slurm_logs/ -name "*.log" -mtime +7 -print
```

## Alerts

When disk usage gets high, alerts are posted to
`#public-compute-coordination` on Slack. If you see one:

1. Run `cat /mnt/polished-lake/scripts/out/usage-by-user.txt | grep $USER`
2. Use `dust` to find large directories
3. Delete anything stale — old checkpoints, logs, cached data
4. You don't need to delete data you're actively using

## Best Practices

- **Save checkpoints selectively** — keep only the best and latest,
  not every epoch.
- **Use the shared HF cache** instead of downloading models to your
  home directory.
- **Clean up after experiments** — delete intermediate artifacts when
  you have final results.
- **Write large outputs to shared artifacts** directories (e.g.,
  `/mnt/polished-lake/artifacts/public/`) where monitoring scripts
  can track them.
- **Set a calendar reminder** to check your usage weekly.
