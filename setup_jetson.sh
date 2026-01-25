#!/bin/bash

# ==============================================================
# Jetson Orin Nano Setup Script for Blind Assistive Project
# ==============================================================

echo "============================================================"
echo "JETSON ORIN NANO SETUP - BLIND ASSISTIVE PROJECT"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}[ERROR] This script is designed for NVIDIA Jetson devices${NC}"
    echo "Detected system: $(uname -a)"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}[INFO] Jetson device detected${NC}"
cat /etc/nv_tegra_release

# ==============================================================
# 1. SYSTEM UPDATE
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 1: System Update"
echo "============================================================"

read -p "Update system packages? (recommended) (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}[UPDATE] Updating package lists...${NC}"
    sudo apt update
    
    echo -e "${YELLOW}[UPDATE] Upgrading packages (this may take a while)...${NC}"
    sudo apt upgrade -y
fi

# ==============================================================
# 2. INSTALL DEPENDENCIES
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 2: Installing System Dependencies"
echo "============================================================"

echo -e "${YELLOW}[INSTALL] Installing essential packages...${NC}"
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    v4l-utils \
    libv4l-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libportaudio2 \
    portaudio19-dev \
    espeak \
    libespeak-dev \
    ffmpeg

echo -e "${GREEN}[DONE] System dependencies installed${NC}"

# ==============================================================
# 3. CAMERA PERMISSIONS
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 3: Setting up Camera Permissions"
echo "============================================================"

echo -e "${YELLOW}[CAMERA] Adding user to video group...${NC}"
sudo usermod -aG video $USER

echo -e "${YELLOW}[CAMERA] Detecting cameras...${NC}"
ls -la /dev/video* 2>/dev/null || echo "No video devices found"

echo -e "${GREEN}[INFO] Camera permissions configured${NC}"
echo -e "${YELLOW}[NOTE] You may need to log out and log back in for group changes to take effect${NC}"

# ==============================================================
# 4. PYTHON PACKAGES
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 4: Installing Python Packages"
echo "============================================================"

echo -e "${YELLOW}[PYTHON] Upgrading pip...${NC}"
python3 -m pip install --upgrade pip

echo -e "${YELLOW}[PYTHON] Installing PyTorch (Jetson version)...${NC}"
# Check if PyTorch is already installed
if python3 -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}[INFO] PyTorch already installed${NC}"
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
else
    echo -e "${YELLOW}[PYTORCH] Installing PyTorch for Jetson...${NC}"
    echo "This may take several minutes..."
    
    # Install torch and torchvision from NVIDIA's repo
    wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
    pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl
    
    # Clean up
    rm torch-2.3.0-cp310-cp310-linux_aarch64.whl
fi

echo -e "${YELLOW}[PYTHON] Installing other dependencies...${NC}"
pip3 install \
    numpy \
    opencv-python \
    ultralytics \
    pyttsx3 \
    pyaudio \
    timm \
    onnx \
    onnxruntime

echo -e "${GREEN}[DONE] Python packages installed${NC}"

# ==============================================================
# 5. VERIFY CUDA
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 5: Verifying CUDA Installation"
echo "============================================================"

if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}[CUDA] CUDA compiler found${NC}"
    nvcc --version
else
    echo -e "${RED}[WARNING] CUDA compiler not found${NC}"
    echo "You may need to install CUDA toolkit"
fi

# Test PyTorch CUDA
echo -e "${YELLOW}[CUDA] Testing PyTorch CUDA support...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# ==============================================================
# 6. TEST CAMERA
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 6: Testing Camera"
echo "============================================================"

echo -e "${YELLOW}[CAMERA] Available video devices:${NC}"
v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"

echo ""
echo -e "${YELLOW}[CAMERA] Testing camera with Python...${NC}"
python3 << 'EOF'
import cv2
import sys

# Try different camera indices
for i in range(3):
    print(f"\nTrying camera index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {i} works! Frame shape: {frame.shape}")
            cap.release()
            sys.exit(0)
        cap.release()

print("\n✗ No working camera found")
sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Camera test passed${NC}"
else
    echo -e "${RED}[WARNING] Camera test failed${NC}"
    echo "Please check camera connection and try:"
    echo "  - For CSI camera: Ensure ribbon cable is properly connected"
    echo "  - For USB camera: Try different USB port"
    echo "  - Run: v4l2-ctl --list-devices"
fi

# ==============================================================
# 7. SET POWER MODE
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 7: Setting Power Mode"
echo "============================================================"

echo -e "${YELLOW}[POWER] Current power mode:${NC}"
sudo /usr/sbin/nvpmodel -q 2>/dev/null || echo "nvpmodel not available"

read -p "Set power mode to MAXN for best performance? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}[POWER] Setting to MAXN mode...${NC}"
    sudo /usr/sbin/nvpmodel -m 0
    sudo jetson_clocks
    echo -e "${GREEN}[DONE] Power mode set to MAXN${NC}"
fi

# ==============================================================
# 8. CREATE TEST SCRIPT
# ==============================================================
echo ""
echo "============================================================"
echo "STEP 8: Creating Quick Test Script"
echo "============================================================"

cat > test_system.py << 'EOF'
#!/usr/bin/env python3
"""Quick system test for Jetson Blind Assistive Project"""

import sys

def test_imports():
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    modules = {
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'ultralytics': 'YOLOv8',
        'pyttsx3': 'Text-to-Speech',
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            exec(f"import {module}")
            print(f"✓ {name:20s} OK")
        except ImportError as e:
            print(f"✗ {name:20s} FAILED: {e}")
            all_ok = False
    
    return all_ok

def test_cuda():
    print("\n" + "=" * 60)
    print("TESTING CUDA")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("WARNING: CUDA not available")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_camera():
    print("\n" + "=" * 60)
    print("TESTING CAMERA")
    print("=" * 60)
    
    try:
        import cv2
        
        for i in range(3):
            print(f"\nTrying camera {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Camera {i} OK - Shape: {frame.shape}")
                    cap.release()
                    return True
                cap.release()
        
        print("✗ No working camera found")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("\n")
    print("=" * 60)
    print("JETSON SYSTEM TEST")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Camera", test_camera()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name:20s} {'PASSED' if status else 'FAILED'}")
    
    print("=" * 60)
    
    if all(status for _, status in results):
        print("\n✓ ALL TESTS PASSED - System ready!")
        return 0
    else:
        print("\n✗ Some tests failed - Check errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_system.py

echo -e "${GREEN}[DONE] Test script created: test_system.py${NC}"

# ==============================================================
# 9. FINAL SUMMARY
# ==============================================================
echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for camera permissions)"
echo "2. Run system test: python3 test_system.py"
echo "3. Test camera module: python3 camera_utils.py"
echo "4. Test depth estimation: python3 depth.py"
echo "5. Run main program: python3 main.py"
echo ""
echo "Troubleshooting:"
echo "- Camera issues: Check config.py settings"
echo "- For CSI camera: Set USE_GSTREAMER = True"
echo "- For USB camera: Set USE_GSTREAMER = False"
echo "- Adjust CAMERA_INDEX if needed (0, 1, or 2)"
echo ""
echo "Documentation:"
echo "- NVIDIA Jetson: https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit"
echo "- Camera setup: https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera"
echo ""
echo "============================================================"
echo -e "${GREEN}Setup script completed!${NC}"
echo "============================================================"