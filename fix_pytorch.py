#!/usr/bin/env python3
"""
Quick fix script for PyTorch/TorchVision compatibility on Jetson post-update.
Properly handles the cuDNN 8/9 and torch/torchvision version mismatch.
"""

import subprocess
import sys

def run_cmd(cmd, description):
    """Run a shell command and report results"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

# Step 1: Fix cuDNN compatibility
print("\nStep 1: Checking cuDNN configuration...")
run_cmd("ls -la /usr/lib/aarch64-linux-gnu/libcudnn.so* | grep -E 'libcudnn.so[^a-z_]'",
        "cuDNN Library Status")

# Step 2: Uninstall incompatible packages
print("\n\nStep 2: Cleaning up incompatible packages...")
run_cmd("pip3 uninstall -y torch torchvision torchaudio ultralytics",
        "Removing incompatible PyTorch/TorchVision packages")

# Step 3: Install correct packages
print("\n\nStep 3: Installing compatible packages...")
run_cmd("pip3 install --force-reinstall --no-deps /home/misti/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl",
        "Installing NVIDIA-optimized PyTorch")

run_cmd("pip3 install 'torchvision==0.17.0' --no-deps --force-reinstall",
        "Installing compatible TorchVision")

run_cmd("pip3 install 'numpy<2' --force-reinstall",
        "Installing compatible NumPy")

run_cmd("pip3 install ultralytics==8.0.200 --force-reinstall",
        "Installing older ultralytics (avoids SAM model issues)")

# Step 4: Verify
print("\n\nStep 4: Verifying installation...")
run_cmd("python3 -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')\"",
        "PyTorch/CUDA Check")

print("\n" + "="*60)
print("✅ Installation complete! Try running: python3 main.py")
print("="*60 + "\n")
