"""
Trial 4: Minimal GPU Training
==============================
A tiny training loop to verify PyTorch + GPU works end-to-end.

HOW TO RUN:
  Option A (interactive): srun --gres gpu:1 --time 00:10:00 python 04_gpu_training.py
  Option B (batch):       sbatch 04_gpu_training.sbatch
"""

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    print("=" * 60)
    print("TRIAL 4: Minimal GPU Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    X = torch.randn(10000, 20, device=device)
    W_true = torch.randn(20, 1, device=device)
    y = X @ W_true + 0.1 * torch.randn(10000, 1, device=device)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {X.shape[0]:,}")
    print(f"\nTraining for 100 epochs...")

    start = time.time()
    for epoch in range(100):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.6f} | Time: {elapsed:.2f}s")

    total_time = time.time() - start
    print(f"\nTraining complete in {total_time:.2f}s")
    print(f"Final loss: {loss.item():.6f}")

    save_path = os.path.expanduser("~/slurm_logs/trial4_model.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    print("\n" + "=" * 60)
    print("Trial 4 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
