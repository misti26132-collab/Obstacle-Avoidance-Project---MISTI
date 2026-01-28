# Camera Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 416
CAMERA_HEIGHT = 416
CAMERA_FPS = 30
USE_GSTREAMER = True
CAMERA_FLIP_METHOD = 0

# GStreamer Pipeline for CSI Camera
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

# Frame Processing Intervals
DEPTH_FRAME_SKIP = 10
YOLO_COCO_FRAME_SKIP = 10
YOLO_CUSTOM_FRAME_SKIP = 8

# YOLO Detection Confidence Thresholds
YOLO_CONFIDENCE = 0.55
YOLO_CONFIDENCE_FURNITURE = 0.40

# Furniture Detection Classes
FURNITURE_CLASSES = [
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'bench', 'tv'
]

# Detection Merge Settings
IOU_THRESHOLD_MERGE = 0.7

# Custom Model Confidence per Class
CUSTOM_MODEL_CONFIDENCE = {
    'pillar': 0.5,
    'chair': 0.3,
    'table': 0.5,
    'default': 0.3
}

# Text-to-Speech Configuration
SPEECH_COOLDOWN = 2.0
SPEECH_RATE = 190
SPEECH_VOLUME = 1.0

# Depth Processing
DEPTH_HISTORY_SIZE = 15
DEPTH_MIN_PERCENTILE = 10
DEPTH_MAX_PERCENTILE = 90
USE_HALF_PRECISION = True

# Distance Thresholds (in meters)
VERY_CLOSE_THRESHOLD = 0.75
CLOSE_THRESHOLD = 0.5

# Lateral Position Boundaries
LEFT_BOUNDARY = 0.7
RIGHT_BOUNDARY = 1.3

# Error Handling
MAX_CONSECUTIVE_ERRORS = 10
HEALTH_CHECK_INTERVAL = 120

# Furniture Depth Adjustment
FURNITURE_DEPTH_BOOST = 1.15
FURNITURE_DEPTH_ADDITIVE = 0.1

# Jetson Hardware Configuration
JETSON_POWER_MODE = "MAXN"
CUDA_DEVICE = 0
NUM_WORKERS = 2
BATCH_SIZE = 1

# YOLO Performance Optimization
YOLO_IMG_SIZE = 288
YOLO_MAX_DET = 7
YOLO_AGNOSTIC_NMS = True

# Memory Management (Post-Jetson Update)
AGGRESSIVE_MEMORY_CLEANUP = True  # Enable aggressive garbage collection
CUDA_MEMORY_THRESHOLD = 0.85      # Clear cache when reaching 85% memory usage
GC_COLLECT_INTERVAL = 30          # Run gc.collect() every N frames
CUDA_EMPTY_CACHE_INTERVAL = 50    # Clear CUDA cache every N frames

# Model Loading Resilience
MODEL_CACHE_DIR = ".cache/models"  # Local model cache directory
MAX_MODEL_LOAD_RETRIES = 5
MODEL_LOAD_TIMEOUT = 60            # Timeout for model downloads (seconds)
OFFLINE_MODE = False                # Set to True to use only cached models
SKIP_MIDAS_DOWNLOAD = False         # Set to True to use local MiDaS if available

# Obstacle Priority Classification
OBSTACLE_PRIORITIES = {
    # Critical - Moving vehicles and people
    'person': 'CRITICAL',
    'car': 'CRITICAL',
    'truck': 'CRITICAL',
    'bus': 'CRITICAL',
    'motorcycle': 'CRITICAL',
    'bicycle': 'CRITICAL',
    
    # High - Static obstacles at walking height
    'chair': 'HIGH',
    'bench': 'HIGH',
    'table': 'HIGH',
    'pillar': 'HIGH',
    'couch': 'HIGH',
    'potted plant': 'HIGH',
    'backpack': 'HIGH',
    'suitcase': 'HIGH',
    'dining table': 'HIGH',
    
    # Low - Smaller or less hazardous objects
    'bottle': 'LOW',
    'tv': 'LOW',
    'laptop': 'LOW',
    'keyboard': 'LOW',
    'cell phone': 'LOW',
    'handbag': 'LOW',
    'bed': 'LOW',
}

# Alert Priority Levels
ALERT_PRIORITIES = ['CRITICAL', 'HIGH']

# User-Friendly Object Names for Speech
FRIENDLY_NAMES = {
    'person': 'person',
    'bicycle': 'bicycle',
    'car': 'car',
    'motorcycle': 'motorcycle',
    'bus': 'bus',
    'truck': 'truck',
    'chair': 'chair',
    'couch': 'couch',
    'potted plant': 'plant',
    'bed': 'bed',
    'dining table': 'table',
    'bench': 'bench',
    'backpack': 'backpack',
    'suitcase': 'suitcase',
    'handbag': 'bag',
    'pillar': 'pole',
    'table': 'table',
}