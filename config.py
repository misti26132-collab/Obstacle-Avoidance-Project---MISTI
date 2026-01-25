CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

USE_GSTREAMER = True
CAMERA_FLIP_METHOD = 0
CAMERA_INDEX = 0

# In config.py
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id={sensor_id} ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={fps}/1 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

# Detection Settings
YOLO_CONFIDENCE = 0.5  
YOLO_CONFIDENCE_FURNITURE = 0.30  
FURNITURE_CLASSES = [
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'bench', 'tv'
]

IOU_THRESHOLD_MERGE = 0.7  

# Speech Settings
SPEECH_COOLDOWN = 3.0
SPEECH_RATE = 175
SPEECH_VOLUME = 1.0

# Depth Settings - JETSON OPTIMIZED
DEPTH_HISTORY_SIZE = 30
DEPTH_MIN_PERCENTILE = 5
DEPTH_MAX_PERCENTILE = 95
USE_HALF_PRECISION = False  

# Distance Thresholds
VERY_CLOSE_THRESHOLD = 0.75
CLOSE_THRESHOLD = 0.5

# Direction Boundaries
LEFT_BOUNDARY = 0.7   
RIGHT_BOUNDARY = 1.3 

# System Health
MAX_CONSECUTIVE_ERRORS = 10
HEALTH_CHECK_INTERVAL = 30  

# Furniture depth boost
FURNITURE_DEPTH_BOOST = 1.15  
FURNITURE_DEPTH_ADDITIVE = 0.1

# JETSON PERFORMANCE SETTINGS
JETSON_POWER_MODE = "MAXN"  
CUDA_DEVICE = 0
NUM_WORKERS = 2  
BATCH_SIZE = 1  