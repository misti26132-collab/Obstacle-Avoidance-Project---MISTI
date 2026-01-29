#!/bin/bash
# Quick launcher with CUDA setup

# Set CUDA paths (in case they're not in environment)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

echo "============================================================"
echo "LAUNCHING OBSTACLE DETECTION WITH CUDA"
echo "============================================================"

# Quick CUDA check
python3 << EOF
import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA Status: {'✓ ENABLED' if cuda_available else '✗ DISABLED'}")
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("============================================================")
EOF

# Run the main program
python3 main.py "$@"
