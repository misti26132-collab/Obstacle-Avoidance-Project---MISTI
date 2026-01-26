#!/usr/bin/env python3
"""
Performance profiler to identify bottlenecks in the obstacle avoidance system
"""
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from depth import DepthEstimator
from camera_utils import JetsonCamera
import config

print("=" * 60)
print("PERFORMANCE PROFILER")
print("=" * 60)

# Initialize components
print("\n1. Initializing Camera...")
cap = JetsonCamera(
    camera_id=config.CAMERA_INDEX,
    width=config.CAMERA_WIDTH,
    height=config.CAMERA_HEIGHT,
    fps=config.CAMERA_FPS
)

# Warm up camera
for _ in range(5):
    cap.read()

ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot read from camera")
    exit(1)

print(f"   Frame shape: {frame.shape}")

# Test 1: Camera FPS
print("\n2. Testing Camera Read Speed...")
iterations = 100
start = time.time()
for _ in range(iterations):
    ret, frame = cap.read()
elapsed = time.time() - start
camera_fps = iterations / elapsed
print(f"   Camera FPS: {camera_fps:.1f}")

# Test 2: Depth Estimation
print("\n3. Testing Depth Estimation...")
depth_estimator = DepthEstimator()
times = []
for i in range(10):
    start = time.time()
    depth_map = depth_estimator.estimate(frame)
    elapsed = time.time() - start
    times.append(elapsed * 1000)
    if i == 0:
        print(f"   First run: {elapsed*1000:.1f}ms (includes warmup)")

avg_time = np.mean(times[1:])  # Skip first run
print(f"   Average: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
print(f"   FP16 Enabled: {config.USE_HALF_PRECISION}")

# Test 3: YOLO COCO
print("\n4. Testing YOLO COCO (YOLOv8n)...")
yolo_coco = YOLO('yolov8n.pt')
yolo_coco.to('cuda:0' if torch.cuda.is_available() else 'cpu')

times = []
for i in range(10):
    start = time.time()
    results = yolo_coco(
        frame,
        conf=config.YOLO_CONFIDENCE,
        imgsz=config.YOLO_IMG_SIZE,
        max_det=config.YOLO_MAX_DET,
        verbose=False,
        device=config.CUDA_DEVICE,
        half=config.USE_HALF_PRECISION
    )
    elapsed = time.time() - start
    times.append(elapsed * 1000)
    if i == 0:
        print(f"   First run: {elapsed*1000:.1f}ms")

avg_time = np.mean(times[1:])
print(f"   Average: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
print(f"   Image size: {config.YOLO_IMG_SIZE}px")
print(f"   Max detections: {config.YOLO_MAX_DET}")

# Test 4: Custom YOLO
print("\n5. Testing Custom YOLO...")
try:
    yolo_custom = YOLO('runs/detect/blind_navigation/obstacles_v1/weights/best.pt')
    yolo_custom.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    times = []
    for i in range(10):
        start = time.time()
        results = yolo_custom(
            frame,
            conf=0.3,
            imgsz=config.YOLO_IMG_SIZE,
            verbose=False,
            device=config.CUDA_DEVICE,
            half=config.USE_HALF_PRECISION
        )
        elapsed = time.time() - start
        times.append(elapsed * 1000)
        if i == 0:
            print(f"   First run: {elapsed*1000:.1f}ms")
    
    avg_time = np.mean(times[1:])
    print(f"   Average: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Full pipeline simulation
print("\n6. Testing Full Pipeline (with frame skipping)...")
print(f"   Depth skip: {config.DEPTH_FRAME_SKIP}")
print(f"   COCO skip: {config.YOLO_COCO_FRAME_SKIP}")
print(f"   Custom skip: {config.YOLO_CUSTOM_FRAME_SKIP}")

total_time = 0
frame_count = 30

for i in range(1, frame_count + 1):
    start = time.time()
    
    ret, frame = cap.read()
    
    # Simulate processing with frame skipping
    if i % config.DEPTH_FRAME_SKIP == 0:
        _ = depth_estimator.estimate(frame)
    
    if i % config.YOLO_COCO_FRAME_SKIP == 0:
        _ = yolo_coco(frame, conf=0.5, imgsz=config.YOLO_IMG_SIZE, verbose=False, half=config.USE_HALF_PRECISION)
    
    if i % config.YOLO_CUSTOM_FRAME_SKIP == 0:
        try:
            _ = yolo_custom(frame, conf=0.3, imgsz=config.YOLO_IMG_SIZE, verbose=False, half=config.USE_HALF_PRECISION)
        except:
            pass
    
    elapsed = time.time() - start
    total_time += elapsed

avg_fps = frame_count / total_time
print(f"   Average FPS: {avg_fps:.1f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 60)

print(f"\nExpected Performance:")
print(f"  Camera:        {camera_fps:.1f} FPS")
print(f"  Full Pipeline: {avg_fps:.1f} FPS")

if avg_fps < 10:
    print("\n⚠️  PERFORMANCE ISSUES DETECTED")
    print("\nRecommendations:")
    print("  1. Increase frame skipping in config.py:")
    print("     DEPTH_FRAME_SKIP = 5")
    print("     YOLO_COCO_FRAME_SKIP = 4")
    print("     YOLO_CUSTOM_FRAME_SKIP = 3")
    print("\n  2. Reduce YOLO image size:")
    print("     YOLO_IMG_SIZE = 320")
    print("\n  3. Enable Jetson max performance:")
    print("     sudo nvpmodel -m 0")
    print("     sudo jetson_clocks")
elif avg_fps < 15:
    print("\n✓ Performance is acceptable")
    print("  Fine-tune settings in config.py for better balance")
else:
    print("\n✓ Excellent performance!")

print("\n" + "=" * 60)

cap.release()