
# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Detection Settings
ROBOFLOW_INTERVAL = 3  
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

# Depth Settings
DEPTH_HISTORY_SIZE = 30
DEPTH_MIN_PERCENTILE = 5
DEPTH_MAX_PERCENTILE = 95

# Distance Thresholds
VERY_CLOSE_THRESHOLD = 0.75
CLOSE_THRESHOLD = 0.5

# Direction Boundaries (as fraction of frame width from center)
LEFT_BOUNDARY = 0.7   
RIGHT_BOUNDARY = 1.3 

# System Health
MAX_CONSECUTIVE_ERRORS = 10
HEALTH_CHECK_INTERVAL = 30  # frames

# IMPROVED: Furniture depth boost (make furniture appear more important)
FURNITURE_DEPTH_BOOST = 1.15  
FURNITURE_DEPTH_ADDITIVE = 0.1