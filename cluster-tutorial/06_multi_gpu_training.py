"""
Trial 6: Multi-GPU Training with DDP
======================================
Trains a simple model across multiple GPUs using PyTorch DistributedDataParallel.

HOW TO RUN:
  Option A (srun, 4 GPUs):
    srun --gres gpu:4 --time 00:15:00 torchrun --nproc_per_node=4 06_multi_gpu_training.py

  Option B (sbatch):
    sbatch 06_multi_gpu_training.sbatch

  Option C (full node, 8 GPUs):
    srun --gres gpu:8 --time 00:15:00 torchrun --nproc_per_node=8 06_multi_gpu_training.py

KEY CONCEPTS:
  - torchrun: PyTorch's launcher that sets up distributed environment variables
  - DDP: wraps your model so gradients are synced across GPUs automatically
  - Each GPU runs its own process with a unique "rank"
  - Data is split across GPUs with DistributedSampler
"""

import os
import time
import socket

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def log(rank, msg):
    if rank == 0:
        print(msg)


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 60)
        print("TRIAL 6: Multi-GPU Training (DDP)")
        print("=" * 60)
        print(f"Hostname:   {socket.gethostname()}")
        print(f"World size: {world_size} GPUs")
        print(f"SLURM Job:  {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()

    torch.manual_seed(42)
    n_samples = 50000
    n_features = 100
    X = torch.randn(n_samples, n_features)
    W_true = torch.randn(n_features, 1)
    y = X @ W_true + 0.1 * torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=256, sampler=sampler)

    model = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    log(rank, f"Model parameters:    {param_count:,}")
    log(rank, f"Total samples:       {n_samples:,}")
    log(rank, f"Samples per GPU:     {n_samples // world_size:,}")
    log(rank, f"Batch size per GPU:  256")
    log(rank, f"Effective batch size: {256 * world_size}")
    log(rank, "")

    n_epochs = 20
    log(rank, f"Training for {n_epochs} epochs...")
    start = time.time()

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(local_rank)
            batch_y = batch_y.to(local_rank)

            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start
            log(rank, f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")

    total_time = time.time() - start

    if rank == 0:
        print(f"\nTraining complete in {total_time:.2f}s")
        print(f"Final loss: {avg_loss:.6f}")
        print(f"Throughput: {n_samples * n_epochs / total_time:.0f} samples/sec")

        save_path = os.path.expanduser("~/slurm_logs/trial6_model.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        print("\n" + "=" * 60)
        print("Trial 6 complete!")
        print("=" * 60)

    cleanup()


if __name__ == "__main__":
    main()
