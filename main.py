import cv2
import time
import numpy as np
import argparse
import logging
import config as config
import torch
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA
print("=" * 60)
print("GPU STATUS")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Import YOLO
try:
    from ultralytics import YOLO
    logger.info("YOLO imported successfully")
except Exception as e:
    logger.error(f"Failed to import YOLO: {e}")
    raise

# Import speech
from Speech_new import SpeechEngine


def calculate_iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    if hasattr(box1, 'tolist'):
        box1 = box1.tolist()
    if hasattr(box2, 'tolist'):
        box2 = box2.tolist()
    
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


class DualModelDetector:
    """
    Uses TWO YOLO models:
    1. COCO model (yolov8n.pt) - detects people, cars, furniture
    2. Custom model (your trained model) - detects walls, poles, tables
    
    Merges detections and uses bounding box size for distance estimation
    """
    
    def __init__(self, custom_model_path='runs/detect/blind_navigation/obstacles_v1/weights/best.pt', use_coco=True):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Load COCO model
        if use_coco:
            logger.info("Loading COCO model (YOLOv8n)...")
            self.yolo_coco = YOLO('yolov8n.pt')
            self.yolo_coco.to(device)
            logger.info(f"COCO model loaded on {device}")
        else:
            self.yolo_coco = None
            logger.info("COCO model disabled")
        
        # Load custom model
        logger.info(f"Loading custom model: {custom_model_path}")
        try:
            self.yolo_custom = YOLO(custom_model_path)
            self.yolo_custom.to(device)
            logger.info(f"Custom model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            logger.warning("Continuing with COCO model only...")
            self.yolo_custom = None
        
        # COCO target classes (general obstacles)
        self.coco_target_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'bench', 'backpack', 'suitcase', 'handbag', 'bottle'
        ]
        
        # Frame dimensions (will be set on first frame)
        self.frame_width = None
        self.frame_height = None
        
        # Frame counter for alternating models
        self.frame_count = 0
        
        logger.info("DualModelDetector initialized")
        logger.info(f"COCO classes: {len(self.coco_target_classes)}")
        logger.info("Custom classes: walls, poles, tables (from your trained model)")
    
    def _estimate_distance_from_box(self, box, class_name, source):
        """
        Estimate distance based on bounding box size
        Different thresholds for different object types
        
        Args:
            box: [x1, y1, x2, y2]
            class_name: object class
            source: 'coco' or 'custom'
        
        Returns: "very_close", "close", or "far"
        """
        x1, y1, x2, y2 = box[:4]
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        # Normalize by frame size
        frame_area = self.frame_width * self.frame_height
        relative_size = box_area / frame_area
        
        # Adjust thresholds based on object type
        if class_name in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
            # People and vehicles - standard thresholds
            very_close_threshold = 0.15  # 15% of frame
            close_threshold = 0.08       # 8% of frame
            
        elif class_name in ['wall', 'pillar', 'pole']:
            # Walls and poles - often thin but dangerous
            # Lower threshold so we detect them earlier
            very_close_threshold = 0.10  # More sensitive
            close_threshold = 0.05
            
        elif class_name in ['table', 'chair', 'bench', 'dining table', 'couch']:
            # Furniture - often large
            very_close_threshold = 0.20
            close_threshold = 0.10
            
        else:
            # Default for other objects
            very_close_threshold = 0.15
            close_threshold = 0.08
        
        if relative_size > very_close_threshold:
            return "very_close"
        elif relative_size > close_threshold:
            return "close"
        else:
            return "far"
    
    def _get_direction(self, box):
        """
        Determine if object is left, center, or right
        """
        x1, y1, x2, y2 = box[:4]
        
        # Get center of box
        box_center_x = (x1 + x2) / 2
        
        # Normalize to frame width
        relative_x = box_center_x / self.frame_width
        
        # Use config boundaries
        left_boundary = config.LEFT_BOUNDARY
        right_boundary = config.RIGHT_BOUNDARY
        
        if relative_x < left_boundary:
            return 'left'
        elif relative_x > right_boundary:
            return 'right'
        else:
            return 'center'
    
    def _get_priority(self, class_name):
        """Get priority for an obstacle class"""
        return config.OBSTACLE_PRIORITIES.get(class_name, 'LOW')
    
    def _should_alert(self, priority):
        """Check if this priority should trigger alerts"""
        return priority in config.ALERT_PRIORITIES
    
    def _merge_detections(self, coco_detections, custom_detections):
        """
        Merge detections from both models, removing duplicates
        Keep higher confidence detection when boxes overlap
        """
        all_detections = custom_detections + coco_detections
        
        if len(all_detections) <= 1:
            return all_detections
        
        # Sort by confidence (highest first)
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        for det in all_detections:
            # Check if this detection overlaps with any already-merged detection
            is_duplicate = False
            for merged_det in merged:
                iou = calculate_iou(det['box'], merged_det['box'])
                if iou > config.IOU_THRESHOLD_MERGE:
                    # Overlaps significantly - it's a duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        return merged
    
    def process(self, frame):
        """
        Process frame with BOTH models and return merged detections
        
        Returns: (annotated_frame, direction, distance, class_name, priority)
        """
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            logger.info(f"Frame size: {self.frame_width}x{self.frame_height}")
        
        self.frame_count += 1
        
        all_detections = []
        
        # Run COCO model (every frame)
        if self.yolo_coco is not None:
            try:
                coco_results = self.yolo_coco(
                    frame,
                    conf=0.4,  # Confidence threshold
                    verbose=False,
                    device='cuda:0' if torch.cuda.is_available() else 'cpu'
                )
                
                coco_detections = self._parse_yolo_results(
                    coco_results, 
                    'coco',
                    filter_classes=self.coco_target_classes
                )
                all_detections.extend(coco_detections)
                
            except Exception as e:
                logger.error(f"COCO model error: {e}")
        
        # Run custom model (every frame or alternate frames for speed)
        # For demo, run every frame for best detection
        if self.yolo_custom is not None:
            try:
                custom_results = self.yolo_custom(
                    frame,
                    conf=0.3,  # Slightly lower threshold for custom model
                    verbose=False,
                    device='cuda:0' if torch.cuda.is_available() else 'cpu'
                )
                
                custom_detections = self._parse_yolo_results(
                    custom_results,
                    'custom',
                    filter_classes=None  # Accept all classes from custom model
                )
                all_detections.extend(custom_detections)
                
            except Exception as e:
                logger.error(f"Custom model error: {e}")
        
        # If we have detections from both models, merge them
        if len(all_detections) > 1:
            # Split by source first
            coco_dets = [d for d in all_detections if d['source'] == 'coco']
            custom_dets = [d for d in all_detections if d['source'] == 'custom']
            
            # Merge to remove duplicates
            all_detections = self._merge_detections(coco_dets, custom_dets)
        
        # Filter by priority - only keep alert-worthy obstacles
        alert_detections = [d for d in all_detections if self._should_alert(d['priority'])]
        
        # Sort by priority and distance
        priority_order = {'CRITICAL': 3, 'HIGH': 2, 'LOW': 1}
        distance_order = {'very_close': 3, 'close': 2, 'far': 1}
        
        alert_detections.sort(
            key=lambda x: (
                priority_order.get(x['priority'], 0),
                distance_order.get(x['distance'], 0)
            ),
            reverse=True
        )
        
        # Create annotated frame
        annotated = frame.copy()
        
        # Draw all detections
        for det in all_detections:
            self._draw_detection(annotated, det)
        
        # Return highest priority detection for speech
        if alert_detections:
            top_detection = alert_detections[0]
            
            # Add overlay for top detection
            self._add_overlay(
                annotated,
                top_detection['direction'],
                top_detection['distance'],
                top_detection['class'],
                top_detection['source']
            )
            
            return (
                annotated,
                top_detection['direction'],
                top_detection['distance'],
                top_detection['class'],
                top_detection['priority']
            )
        else:
            return annotated, None, None, None, None
    
    def _parse_yolo_results(self, results, source, filter_classes=None):
        """
        Parse YOLO results into standardized detection format
        
        Args:
            results: YOLO results object
            source: 'coco' or 'custom'
            filter_classes: list of class names to keep (None = keep all)
        
        Returns: list of detection dicts
        """
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        class_names = result.names
        
        for i in range(len(boxes)):
            class_id = int(classes[i])
            class_name = class_names[class_id]
            
            # Filter by class if needed
            if filter_classes is not None and class_name not in filter_classes:
                continue
            
            confidence = float(confidences[i])
            box = boxes[i]
            
            # Estimate distance and direction
            distance = self._estimate_distance_from_box(box, class_name, source)
            direction = self._get_direction(box)
            priority = self._get_priority(class_name)
            
            detections.append({
                'box': box,
                'class': class_name,
                'confidence': confidence,
                'distance': distance,
                'direction': direction,
                'priority': priority,
                'source': source
            })
        
        return detections
    
    def _draw_detection(self, frame, det):
        """Draw a single detection on the frame"""
        box = det['box']
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Color based on priority
        if det['priority'] == 'CRITICAL':
            color = (0, 0, 255)  # Red
        elif det['priority'] == 'HIGH':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with source indicator
        source_tag = "C" if det['source'] == 'coco' else "X"
        label = f"[{source_tag}] {det['class']} {det['confidence']:.2f}"
        
        # Background for label
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        label_y1 = max(y1 - label_h - 10, 0)
        label_y2 = max(y1, label_h + 10)
        
        cv2.rectangle(
            frame, (x1, label_y1), (x1 + label_w + 10, label_y2),
            color, -1
        )
        
        cv2.putText(
            frame, label, (x1 + 5, label_y2 - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    def _add_overlay(self, frame, direction, distance, class_name, source):
        """Add text overlay for current alert"""
        if distance == "very_close":
            color = (0, 0, 255)
        elif distance == "close":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        
        distance_text = distance.upper().replace('_', ' ')
        source_text = f"[{'COCO' if source == 'coco' else 'CUSTOM'}]"
        text = f"{class_name.upper()}: {direction.upper()} - {distance_text} {source_text}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x = (frame.shape[1] - text_width) // 2
        y = 40
        
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            (0, 0, 0), -1
        )
        
        cv2.putText(
            frame, text, (x, y), font, font_scale,
            color, thickness, cv2.LINE_AA
        )


def main():
    parser = argparse.ArgumentParser(
        description='Dual Model Obstacle Detection (COCO + Custom)'
    )
    parser.add_argument(
        '--custom-model', type=str, 
        default='runs/detect/blind_navigation/obstacles_v1/weights/best.pt',
        help='Path to your custom trained model'
    )
    parser.add_argument(
        '--no-coco', action='store_true',
        help='Disable COCO model (custom only)'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Disable video display'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera index'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("DUAL MODEL OBSTACLE DETECTION - DEMO VERSION")
    print("=" * 60)
    print(f"COCO Model: {'Enabled' if not args.no_coco else 'Disabled'}")
    print(f"Custom Model: {args.custom_model}")
    print(f"Camera: {args.camera}")
    print("=" * 60)
    
    # Initialize camera
    logger.info("Initializing camera...")
    try:
        from camera_utils import JetsonCamera
        cap = JetsonCamera(
            camera_id=args.camera,
            width=640,
            height=480,
            fps=30
        )
        
        if not cap.isOpened():
            logger.error("JetsonCamera failed to initialize")
            return
            
        logger.info("✅ Jetson CSI camera initialized successfully")
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize detector
    logger.info("Initializing dual model detector...")
    try:
        detector = DualModelDetector(
            custom_model_path=args.custom_model,
            use_coco=not args.no_coco
        )
    except Exception as e:
        logger.error(f"Detector initialization failed: {e}")
        if hasattr(cap, 'release'):
            cap.release()
        return
    
    # Initialize speech
    logger.info("Initializing speech...")
    try:
        speaker = SpeechEngine(cooldown=2.0)
    except Exception as e:
        logger.error(f"Speech initialization failed: {e}")
        if hasattr(cap, 'release'):
            cap.release()
        return
    
    print("=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print("Detection: COCO (people/cars) + Custom (walls/poles/tables)")
    print("Press 'q' to quit")
    print("=" * 60)
    
    show_display = not args.no_display
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Process frame with both models
            try:
                annotated_frame, direction, distance, obstacle_class, priority = detector.process(frame)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                annotated_frame = frame
                direction, distance, obstacle_class, priority = None, None, None, None
            
            # Speak if obstacle detected
            if direction is not None and distance is not None:
                try:
                    speaker.speak(direction, distance, obstacle_class, priority)
                except Exception as e:
                    logger.error(f"Speech error: {e}")
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start
                fps_display = 30 / elapsed if elapsed > 0 else 0
                
                status = f"{direction} - {distance} - {obstacle_class}" if direction else "Clear"
                print(f"[FPS: {fps_display:.1f}] {status}")
                
                fps_start = time.time()
            
            # Display
            if show_display:
                cv2.putText(
                    annotated_frame, f"FPS: {fps_display:.1f}",
                    (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.imshow("Dual Model Detection - Demo", annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n[System] Interrupted")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        print("[System] Cleaning up...")
        if hasattr(cap, 'release'):
            cap.release()
        if show_display:
            cv2.destroyAllWindows()
        speaker.stop()
        print("Shutdown complete")


if __name__ == "__main__":
    main()