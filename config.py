# DUAL MODEL CONFIG - COCO + CUSTOM

# Camera Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Text-to-Speech Configuration
SPEECH_COOLDOWN = 2.0
SPEECH_RATE = 175
SPEECH_VOLUME = 1.0

# Lateral Position Boundaries
LEFT_BOUNDARY = 0.33   # Left third of frame
RIGHT_BOUNDARY = 0.67  # Right third of frame

# Detection Merge Settings (when both models detect same object)
IOU_THRESHOLD_MERGE = 0.5  # Boxes overlapping >50% are considered duplicates

# Obstacle Priority Classification
# CRITICAL = immediate danger, HIGH = important obstacle, LOW = awareness only
OBSTACLE_PRIORITIES = {
    # CRITICAL - Moving vehicles and people (from COCO)
    'person': 'CRITICAL',
    'car': 'CRITICAL',
    'truck': 'CRITICAL',
    'bus': 'CRITICAL',
    'motorcycle': 'CRITICAL',
    'bicycle': 'CRITICAL',
    
    # HIGH - Static obstacles at walking height (from COCO + Custom)
    'chair': 'HIGH',
    'bench': 'HIGH',
    'table': 'HIGH',           # From custom model
    'couch': 'HIGH',
    'potted plant': 'HIGH',
    'backpack': 'HIGH',
    'suitcase': 'HIGH',
    'dining table': 'HIGH',    # From COCO
    
    # HIGH - Critical for blind navigation (from Custom model)
    'wall': 'HIGH',            # YOUR CUSTOM MODEL DETECTS THIS!
    'pole': 'HIGH',            # YOUR CUSTOM MODEL DETECTS THIS!
    'pillar': 'HIGH',          # In case your model calls it pillar
    
    # LOW - Smaller or less hazardous objects
    'bottle': 'LOW',
    'handbag': 'LOW',
    'bed': 'LOW',
}

# Alert Priority Levels (only these will trigger speech)
ALERT_PRIORITIES = ['CRITICAL', 'HIGH']

# User-Friendly Object Names for Speech
FRIENDLY_NAMES = {
    # COCO objects
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
    'bottle': 'bottle',
    
    # Custom model objects
    'wall': 'wall',
    'pole': 'pole',
    'pillar': 'pole',
    'table': 'table',
}