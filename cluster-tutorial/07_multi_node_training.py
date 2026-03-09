"""
Trial 7: Multi-Node Training with DDP
=====================================
Trains a simple model across multiple nodes and multiple GPUs using PyTorch
DistributedDataParallel.

HOW TO RUN:
  Option A (srun, 2 nodes x 4 GPUs):
    bash ~/rl-rewardhacking/cluster-tutorial/07_multi_node_srun.sh

  Option B (sbatch):
    sbatch ~/rl-rewardhacking/cluster-tutorial/07_multi_node_training.sbatch

KEY CONCEPTS:
  - Multi-node DDP uses one or more GPUs on multiple machines
  - Each node launches one torchrun instance
  - torchrun spawns one worker process per GPU on that node
  - All workers meet at a rendezvous endpoint (MASTER_ADDR:MASTER_PORT)
"""

import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup():
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def main():
    local_rank, rank, world_size = setup()

    hostname = socket.gethostname()
    node_id = os.environ.get("SLURM_NODEID", "N/A")
    master_addr = os.environ.get("MASTER_ADDR", "N/A")
    master_port = os.environ.get("MASTER_PORT", "N/A")

    print(
        f"[rank {rank:02d}] host={hostname} node_id={node_id} "
        f"local_rank={local_rank} gpu={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )
    dist.barrier()

    if rank == 0:
        print("=" * 60)
        print("TRIAL 7: Multi-Node Training (DDP)")
        print("=" * 60)
        print(f"Master endpoint: {master_addr}:{master_port}")
        print(f"World size:      {world_size} GPUs total")
        print(f"Node count:      {os.environ.get('SLURM_JOB_NUM_NODES', 'N/A')}")
        print(f"SLURM Job:       {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        print()

    torch.manual_seed(42)
    n_samples = 120000
    n_features = 100
    X = torch.randn(n_samples, n_features)
    W_true = torch.randn(n_features, 1)
    y = X @ W_true + 0.1 * torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
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
    log(rank, f"Model parameters:     {param_count:,}")
    log(rank, f"Total samples:        {n_samples:,}")
    log(rank, f"Samples per process:  {n_samples // world_size:,}")
    log(rank, f"Batch size per GPU:   256")
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

        save_path = os.path.expanduser("~/slurm_logs/trial7_model.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        print("\n" + "=" * 60)
        print("Trial 7 complete!")
        print("=" * 60)

    cleanup()


if __name__ == "__main__":
    main()
