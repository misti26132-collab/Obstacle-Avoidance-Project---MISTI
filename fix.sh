#!/bin/bash
# Fix nvargus daemon for Jetson CSI camera

echo "============================================================"
echo "JETSON CSI CAMERA FIX"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  This script needs sudo privileges."
    echo "Running: sudo bash fix_jetson_camera.sh"
    echo ""
    sudo bash "$0" "$@"
    exit $?
fi

echo "[1/5] Stopping camera-using processes..."
# Kill any Python processes using the camera
pkill -9 -f "python.*main.py" 2>/dev/null
sleep 1
echo "✅ Done"

echo ""
echo "[2/5] Stopping nvargus daemon..."
systemctl stop nvargus-daemon
sleep 2
echo "✅ Done"

echo ""
echo "[3/5] Clearing camera buffers..."
# Clear video device nodes
if [ -e /dev/video0 ]; then
    echo "Camera device found: /dev/video0"
fi
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
echo "✅ Done"

echo ""
echo "[4/5] Restarting nvargus daemon..."
systemctl start nvargus-daemon
sleep 3

# Check if daemon is running
if systemctl is-active --quiet nvargus-daemon; then
    echo "✅ nvargus-daemon is running"
else
    echo "⚠️  nvargus-daemon failed to start"
    echo "Checking logs..."
    journalctl -u nvargus-daemon -n 20 --no-pager
fi

echo ""
echo "[5/5] Checking camera device..."
if [ -e /dev/video0 ]; then
    echo "✅ Camera device exists: /dev/video0"
    ls -l /dev/video0
else
    echo "❌ Camera device not found!"
    echo ""
    echo "Try these:"
    echo "1. Check camera connection"
    echo "2. Run: dmesg | grep -i video"
    echo "3. Reboot: sudo reboot"
fi

echo ""
echo "============================================================"
echo "FIX COMPLETE"
echo "============================================================"
echo ""
echo "Now try: python3 main.py"
echo ""