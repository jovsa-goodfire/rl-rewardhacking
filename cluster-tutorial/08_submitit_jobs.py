"""
Trial 8: Submitting Slurm Jobs from Python with submitit
=========================================================
submitit lets you submit any Python function as a Slurm job.
No bash scripts needed — great for launching experiments programmatically.

SETUP (run once on the cluster):
  uv add submitit torch

HOW TO RUN (from a dev pod or compute node, NOT the login node):
  python 08_submitit_jobs.py

This will submit 3 Slurm jobs to the cluster and wait for results.
"""

import os
import time

import submitit


def train_model(lr: float, seed: int, n_epochs: int = 100) -> dict:
    """A training function that gets submitted as a Slurm job."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[seed={seed}, lr={lr}] Running on {device} ({os.environ.get('SLURM_JOB_ID', 'local')})")

    X = torch.randn(10000, 20, device=device)
    W_true = torch.randn(20, 1, device=device)
    y = X @ W_true + 0.1 * torch.randn(10000, 1, device=device)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start = time.time()
    for epoch in range(n_epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start
    final_loss = loss.item()
    print(f"[seed={seed}, lr={lr}] Done in {elapsed:.2f}s, final loss: {final_loss:.6f}")

    return {"lr": lr, "seed": seed, "final_loss": final_loss, "time": elapsed}


def example_single_job():
    """Submit a single job and get the result."""
    print("=" * 60)
    print("Example A: Single Job")
    print("=" * 60)

    log_dir = os.path.expanduser("~/slurm_logs/submitit/single")
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        gpus_per_node=1,
        timeout_min=10,
        slurm_partition="reserved",
    )

    job = executor.submit(train_model, lr=1e-3, seed=42)
    print(f"Submitted job: {job.job_id}")
    print(f"Log file: {log_dir}/{job.job_id}_0_log.out")
    print("Waiting for result...")

    result = job.result()
    print(f"Result: {result}")
    print()


def example_map_array():
    """Submit multiple jobs as a Slurm array using map_array."""
    print("=" * 60)
    print("Example B: Array of Jobs (map_array)")
    print("=" * 60)

    log_dir = os.path.expanduser("~/slurm_logs/submitit/array")
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        gpus_per_node=1,
        timeout_min=10,
        slurm_partition="reserved",
        slurm_array_parallelism=2,  # run at most 2 jobs concurrently
    )

    learning_rates = [1e-2, 1e-3, 1e-4]
    seeds = [42, 123, 999]

    jobs = executor.map_array(train_model, learning_rates, seeds)
    print(f"Submitted {len(jobs)} jobs: {[j.job_id for j in jobs]}")
    print("Waiting for all results...")

    for i, job in enumerate(jobs):
        try:
            result = job.result()
            print(f"  Job {i} ({job.job_id}): loss={result['final_loss']:.6f}, time={result['time']:.2f}s")
        except Exception as e:
            print(f"  Job {i} ({job.job_id}): FAILED - {e}")

    print()


def example_batch_context():
    """Submit multiple jobs using the batch context manager."""
    print("=" * 60)
    print("Example C: Batch Context Manager")
    print("=" * 60)

    log_dir = os.path.expanduser("~/slurm_logs/submitit/batch")
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        gpus_per_node=1,
        timeout_min=10,
        slurm_partition="reserved",
        slurm_array_parallelism=2,
    )

    configs = [
        {"lr": 1e-2, "seed": 1, "n_epochs": 50},
        {"lr": 1e-3, "seed": 2, "n_epochs": 100},
        {"lr": 1e-4, "seed": 3, "n_epochs": 200},
    ]

    jobs = []
    with executor.batch():
        for cfg in configs:
            job = executor.submit(train_model, **cfg)
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs: {[j.job_id for j in jobs]}")
    print("Waiting for all results...")

    successful = 0
    for i, (job, cfg) in enumerate(zip(jobs, configs)):
        try:
            result = job.result()
            print(f"  Job {i} (lr={cfg['lr']}): loss={result['final_loss']:.6f}")
            successful += 1
        except Exception as e:
            print(f"  Job {i} (lr={cfg['lr']}): FAILED - {e}")

    print(f"\n{successful}/{len(jobs)} jobs completed successfully")
    print()


if __name__ == "__main__":
    print("TRIAL 8: submitit — Submitting Slurm Jobs from Python\n")
    print("Choose an example to run:")
    print("  A) Single job")
    print("  B) Array of jobs (map_array)")
    print("  C) Batch context manager")
    print()

    # Run all three examples sequentially
    example_single_job()
    example_map_array()
    example_batch_context()

    print("=" * 60)
    print("Trial 8 complete!")
    print("All submitit logs are in ~/slurm_logs/submitit/")
    print("=" * 60)
