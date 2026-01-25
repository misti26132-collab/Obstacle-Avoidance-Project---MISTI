#!/usr/bin/env python3
import sys

print("Step 1: Import cv2...")
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\nStep 2: Import config...")
try:
    import config
    print(f"✓ Config imported")
    print(f"  USE_GSTREAMER = {config.USE_GSTREAMER}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\nStep 3: Build GStreamer pipeline...")
try:
    pipeline = config.GSTREAMER_PIPELINE.format(
        sensor_id=0,
        width=640,
        height=480,
        fps=30,
        flip=0
    )
    print(f"✓ Pipeline:\n{pipeline}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\nStep 4: Open camera...")
try:
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("✓ Camera opened")
    else:
        print("✗ Camera failed to open")
        sys.exit(1)
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Read frame...")
try:
    ret, frame = cap.read()
    if ret:
        print(f"✓ Frame read! Shape: {frame.shape}")
        cv2.imwrite("debug_frame.jpg", frame)
        print("✓ Saved debug_frame.jpg")
    else:
        print("✗ Frame read failed")
        sys.exit(1)
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ ALL TESTS PASSED!")
cap.release()