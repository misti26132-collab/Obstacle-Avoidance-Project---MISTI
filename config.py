CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

USE_GSTREAMER = True
CAMERA_FLIP_METHOD = 0

GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

DEPTH_FRAME_SKIP = 2        
YOLO_COCO_FRAME_SKIP = 2    
YOLO_CUSTOM_FRAME_SKIP = 1  

# Detection Settings
YOLO_CONFIDENCE = 0.5  
YOLO_CONFIDENCE_FURNITURE = 0.35  
FURNITURE_CLASSES = [
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'bench', 'tv'
]

IOU_THRESHOLD_MERGE = 0.7  

# Speech Settings
SPEECH_COOLDOWN = 2.5  
SPEECH_RATE = 185      #
SPEECH_VOLUME = 1.0

DEPTH_HISTORY_SIZE = 20      
DEPTH_MIN_PERCENTILE = 5
DEPTH_MAX_PERCENTILE = 95
USE_HALF_PRECISION = True    

# Distance Thresholds
VERY_CLOSE_THRESHOLD = 0.75
CLOSE_THRESHOLD = 0.5

# Direction Boundaries
LEFT_BOUNDARY = 0.7   
RIGHT_BOUNDARY = 1.3 

MAX_CONSECUTIVE_ERRORS = 10
HEALTH_CHECK_INTERVAL = 90  

FURNITURE_DEPTH_BOOST = 1.15  
FURNITURE_DEPTH_ADDITIVE = 0.1

JETSON_POWER_MODE = "MAXN"  
CUDA_DEVICE = 0
NUM_WORKERS = 2  
BATCH_SIZE = 1