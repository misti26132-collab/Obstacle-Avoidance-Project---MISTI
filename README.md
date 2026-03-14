**Edge AI Obstacle Detection System**

Real-time obstacle detection system using computer vision and depth estimation, designed to run on edge devices such as the NVIDIA Jetson Orin Nano.

The system combines YOLO object detection with MiDaS monocular depth estimation to detect nearby objects and estimate their distance using only a camera. It also generates audio feedback describing the obstacle type, direction, and proximity.

**Features**

Real-time object detection using YOLOv8

Depth estimation using MiDaS

Deployment on NVIDIA Jetson Orin Nano

Audio feedback for obstacle awareness

Works with standard camera input

**Technologies**

Python
OpenCV
PyTorch
YOLOv8
MiDaS
NVIDIA Jetson


The program will start the camera feed, detect nearby obstacles, estimate their distance, and provide audio feedback.
