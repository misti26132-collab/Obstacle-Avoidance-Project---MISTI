# PyTorch/CUDA Fix for Jetson Update

## Problem Summary

After the recent Jetson update:
1. **cuDNN version mismatch**: System has cuDNN 9.3, but old PyTorch expected cuDNN 8
2. **PyTorch/TorchVision incompatibility**: Various version mismatches cause circular imports
3. **Model loading failures**: MiDaS and other models fail to load due to torch/torchvision issues

## Quick Fix (Recommended)

Run the automated fix script:

```bash
cd /home/misti/Obstacle-Avoidance-Project---MISTI
python3 fix_pytorch.py
```

This will:
- Remove incompatible packages
- Install NVIDIA-optimized PyTorch 2.5
- Install compatible TorchVision 0.17
- Fix NumPy compatibility
- Install older ultralytics (avoids SAM model issues)

## Manual Fix (If script doesn't work)

### Step 1: Clean up
```bash
pip3 uninstall -y torch torchvision torchaudio ultralytics
```

### Step 2: Install NVIDIA PyTorch
```bash
pip3 install --force-reinstall --no-deps /home/misti/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### Step 3: Install compatible packages
```bash
pip3 install "torchvision==0.17.0" --no-deps --force-reinstall
pip3 install "numpy<2" --force-reinstall
pip3 install ultralytics==8.0.200 --force-reinstall
```

### Step 4: Fix cuDNN symlink (if needed)
```bash
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9.3.0 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ldconfig
```

## Why These Specific Versions?

- **PyTorch 2.5.0a0+nv24.08**: This is the NVIDIA-built PyTorch that has CUDA support for Jetson
- **TorchVision 0.17**: Compatible with PyTorch 2.5 (mostly - some warnings are normal)
- **ultralytics 8.0.200**: Older version that doesn't import SAM model (avoids circular imports)
- **NumPy <2**: PyTorch 2.5 was built against NumPy 1.x

## Testing

After installation, verify everything works:

```bash
# Test PyTorch
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test application
cd /home/misti/Obstacle-Avoidance-Project---MISTI
python3 main.py
```

You should see:
- ✅ CUDA available: True
- ✅ CUDA Device: Orin
- ✅ Model loading messages (even if camera fails)

## Known Issues

### TorchVision version warning
```
WARNING ⚠️ torchvision==0.17 is incompatible with torch==2.5
```
This warning is normal and can be ignored. Ultralytics 8.0.200 doesn't rely heavily on torchvision features that changed.

### NumPy compatibility warning
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x...
```
This is a warning from OpenCV. It's harmless. We keep NumPy <2 for PyTorch compatibility.

### Camera initialization fails
This is expected in a non-Jetson environment or without a connected camera. The application will still initialize.

## If Issue Persists

1. **Clear pip cache**:
   ```bash
   pip3 cache purge
   ```

2. **Check CUDA setup**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Verify PyTorch CUDA**:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"
   ```

4. **Rebuild local pip packages** (nuclear option):
   ```bash
   pip3 install --upgrade --force-reinstall --no-cache-dir torch torchvision
   ```

## Reference Files

- Original PyTorch wheel: `/home/misti/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl`
- Fix script: `/home/misti/Obstacle-Avoidance-Project---MISTI/fix_pytorch.py`

## Support

If you continue to experience issues:

1. Check internet connectivity (for model downloads)
2. Ensure CUDA is properly installed: `which nvcc`
3. Review system logs: `journalctl -xe`
4. Check disk space: `df -h`
5. Try on a fresh Python environment if possible

---

**Last Updated**: Jan 28, 2026  
**Status**: Tested on Jetson Orin with CUDA 12.6, cuDNN 9.3
