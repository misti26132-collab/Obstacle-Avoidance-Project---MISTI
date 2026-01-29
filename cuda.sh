echo "============================================================"
echo "JETSON CUDA DIAGNOSTIC"
echo "============================================================"

# Check if we're on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "✓ Jetson device detected"
    cat /etc/nv_tegra_release
else
    echo "⚠ Warning: Not running on Jetson"
fi

echo ""
echo "1. Checking CUDA installation..."
if [ -d /usr/local/cuda ]; then
    echo "✓ CUDA found at /usr/local/cuda"
    ls -l /usr/local/cuda/lib64/libcudart.so* 2>/dev/null || echo "⚠ libcudart not found"
else
    echo "✗ CUDA not found at /usr/local/cuda"
fi

echo ""
echo "2. Checking LD_LIBRARY_PATH..."
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
if [[ $LD_LIBRARY_PATH == *"cuda"* ]]; then
    echo "✓ CUDA in library path"
else
    echo "✗ CUDA not in library path - THIS IS THE PROBLEM!"
fi

echo ""
echo "3. Checking PyTorch CUDA support..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ PyTorch cannot see CUDA!")
    print("")
    print("Common causes:")
    print("1. PyTorch CPU-only version installed")
    print("2. LD_LIBRARY_PATH not set correctly")
    print("3. CUDA libraries not installed")
EOF

echo ""
echo "============================================================"
echo "FIX INSTRUCTIONS"
echo "============================================================"

if [[ $LD_LIBRARY_PATH != *"cuda"* ]]; then
    echo ""
    echo "OPTION 1: Temporary Fix (this session only)"
    echo "------------------------------------------------------------"
    echo "Run these commands:"
    echo ""
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "export PATH=/usr/local/cuda/bin:\$PATH"
    echo "python3 main.py"
    echo ""
   
    echo "OPTION 2: Permanent Fix (recommended)"
    echo "------------------------------------------------------------"
    echo "Add to your ~/.bashrc:"
    echo ""
    echo "echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
    echo "echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc"
    echo "source ~/.bashrc"
    echo "python3 main.py"
    echo ""
fi

echo "OPTION 3: Check PyTorch Installation"
echo "------------------------------------------------------------"
echo "If CUDA is installed but PyTorch can't see it:"
echo ""
echo "# Check PyTorch version"
echo "pip3 show torch"
echo ""
echo "# If needed, reinstall PyTorch with CUDA support:"
echo "# (This should already be installed on Jetson, but just in case)"
echo "pip3 install --upgrade torch torchvision"
echo ""

echo "============================================================"
echo "After fixing, run: python3 main.py"
echo "You should see: CUDA Available: True"
echo "============================================================"