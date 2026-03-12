#!/bin/bash
# Submit all R6 steering eval jobs in parallel.
# 3 feature sets × 5 alphas + 1 baseline = 16 jobs, ~5 min wall time.

set -e
cd /mnt/polished-lake/home/jsardinha/rl-rewardhacking

FEATURE_SETS="joint_probe_top5 cross_run_overlap a1_ensemble_top5"
ALPHAS="0.0 0.5 1.0 2.0 5.0"

# Baseline (no steering)
echo "Submitting baseline (alpha=0, no features)..."
sbatch --job-name=r6-baseline \
    --nodes=1 --gpus=1 --time=00:30:00 \
    --output=$HOME/slurm_logs/r6-baseline-%j.log \
    --wrap="cd /mnt/polished-lake/home/jsardinha/rl-rewardhacking && FEATURE_SET=none ALPHA=0.0 uv run python scripts/r6_steering_single.py"

# All (feature_set, alpha) combinations
for fs in $FEATURE_SETS; do
    for alpha in $ALPHAS; do
        if [ "$alpha" = "0.0" ]; then
            continue  # baseline already submitted
        fi
        echo "Submitting $fs alpha=$alpha..."
        sbatch --job-name=r6-${fs}-a${alpha} \
            --nodes=1 --gpus=1 --time=00:30:00 \
            --output=$HOME/slurm_logs/r6-${fs}-a${alpha}-%j.log \
            --wrap="cd /mnt/polished-lake/home/jsardinha/rl-rewardhacking && FEATURE_SET=$fs ALPHA=$alpha uv run python scripts/r6_steering_single.py"
    done
done

echo "All jobs submitted. Results will appear in results/rlookout/qwen3-4b/r6_parallel/"
