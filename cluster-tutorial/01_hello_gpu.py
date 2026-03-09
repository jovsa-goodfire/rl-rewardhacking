"""
Trial 1: Hello GPU
===================
Verify you can see GPUs and check the hardware.

HOW TO RUN (from your dev pod, NOT the login node):
  Option A - directly:    python 01_hello_gpu.py
  Option B - via srun:    srun --gres gpu:1 --time 00:10:00 python 01_hello_gpu.py
"""

import subprocess
import os
import socket

print("=" * 60)
print("TRIAL 1: Hello GPU")
print("=" * 60)

print(f"\nHostname:    {socket.gethostname()}")
print(f"User:        {os.environ.get('USER', 'unknown')}")
print(f"SLURM Job:   {os.environ.get('SLURM_JOB_ID', 'not a slurm job')}")
print(f"SLURM Node:  {os.environ.get('SLURM_NODELIST', 'N/A')}")
print(f"SLURM GPUs:  {os.environ.get('SLURM_GPUS_ON_NODE', 'N/A')}")
print(f"Working dir: {os.getcwd()}")

try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_mem / (1024**3)
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")

        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = x @ y
        print(f"\nGPU matrix multiply test: OK (result shape: {z.shape})")
    else:
        print("\nNo CUDA GPUs detected — are you on a compute node?")
except ImportError:
    print("\nPyTorch not installed. Run: uv add torch")

print("\n" + "=" * 60)
print("Trial 1 complete!")
print("=" * 60)
