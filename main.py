import cv2
import time
import numpy as np
import argparse
import logging
from depth import DepthEstimator
from Speech import SpeechEngine
from ultralytics import YOLO
from health_check import SystemHealthMonitor
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_iou(box1, box2):
    # Convert to lists if numpy arrays
    if hasattr(box1, 'tolist'):
        box1 = box1.tolist()
    if hasattr(box2, 'tolist'):
        box2 = box2.tolist()
    
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


class DualModelDetector:
    
    def __init__(self, depth_estimator, custom_model_path='runs/detect/blind_navigation/obstacles_v1/weights/best.pt', use_yolo_medium=False):
        # Load COCO pretrained model for general objects
        model_name = 'yolov8m.pt' if use_yolo_medium else 'yolov8n.pt'
        logger.info(f"[DualModel] Loading COCO model: {model_name}...")
        logger.info("[DualModel] (First run will download model, please wait...)")
        self.yolo_coco = YOLO(model_name)
        logger.info(f"[DualModel] {model_name} loaded successfully")
        
        # Load your custom trained model
        logger.info(f"[DualModel] Loading custom trained model: {custom_model_path}...")
        try:
            self.yolo_custom = YOLO(custom_model_path)
            logger.info("[DualModel] Custom model loaded successfully")
            logger.info(f"[DualModel] Custom classes: {list(self.yolo_custom.names.values())}")
        except Exception as e:
            logger.error(f"[DualModel] Failed to load custom model: {e}")
            logger.error(f"[DualModel] Make sure the model exists at: {custom_model_path}")
            raise
        
        self.depth_estimator = depth_estimator
        self.frame_width = None
        self.frame_height = None
        
        # COCO classes relevant for obstacle avoidance
        self.coco_obstacle_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'bench', 'backpack', 'suitcase', 'handbag', 'bottle'
        ]
        
        self.frame_count = 0
        
        logger.info(f"[DualModel] Initialized: COCO ({model_name}) + Custom Model")
        logger.info(f"[DualModel] Monitoring {len(self.coco_obstacle_classes)} COCO obstacle classes")
        logger.info(f"[DualModel] Custom model has {len(self.yolo_custom.names)} classes")
        logger.info(f"[DualModel] {len(config.FURNITURE_CLASSES)} furniture classes use {config.YOLO_CONFIDENCE_FURNITURE} confidence")
    
    def process(self, frame):
        """
        Process frame with both detection models and depth estimation
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            tuple: (annotated_frame, direction, distance, obstacle_class)
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_count += 1
        
        # Get depth map
        try:
            depth_map, depth_vis = self.depth_estimator.estimate_with_visualization(frame)
        except Exception as e:
            logger.error(f"[DualModel] Depth estimation error: {e}")
            return frame, None, None, None
        
        # 1. Run COCO YOLOv8 (for general objects)
        try:
            yolo_results = self.yolo_coco(frame, conf=config.YOLO_CONFIDENCE_FURNITURE, verbose=False)
            yolo_boxes = yolo_results[0].boxes
        except Exception as e:
            logger.error(f"[DualModel] YOLO COCO error: {e}")
            yolo_boxes = []
        
        # Filter YOLO COCO detections with class-specific confidence thresholds
        yolo_detections = []
        for box in yolo_boxes:
            class_id = int(box.cls[0])
            class_name = self.yolo_coco.names[class_id]
            confidence = float(box.conf[0])
            
            if class_name in self.coco_obstacle_classes:
                # Apply different thresholds for furniture vs other objects
                if class_name in config.FURNITURE_CLASSES:
                    min_conf = config.YOLO_CONFIDENCE_FURNITURE
                else:
                    min_conf = config.YOLO_CONFIDENCE
                
                if confidence >= min_conf:
                    yolo_detections.append({
                        'box': box.xyxy[0].cpu().numpy(),
                        'class': class_name,
                        'confidence': confidence,
                        'source': 'coco'
                    })
        
        # 2. Run Custom YOLOv8 (your trained model)
        try:
            custom_results = self.yolo_custom(frame, conf=0.3, verbose=False)
            custom_boxes = custom_results[0].boxes
        except Exception as e:
            logger.error(f"[DualModel] Custom YOLO error: {e}")
            custom_boxes = []
        
        # Parse custom model detections
        custom_detections = []
        for box in custom_boxes:
            class_id = int(box.cls[0])
            class_name = self.yolo_custom.names[class_id]
            confidence = float(box.conf[0])
            
            custom_detections.append({
                'box': box.xyxy[0].cpu().numpy(),
                'class': class_name,
                'confidence': confidence,
                'source': 'custom'
            })
        
        # Combine detections (remove duplicates based on IoU)
        all_detections = self._merge_detections(yolo_detections, custom_detections)
        
        # Log detection statistics every 100 frames
        if self.frame_count % 100 == 0 and len(all_detections) > 0:
            coco_count = len(yolo_detections)
            custom_count = len(custom_detections)
            logger.info(f"[Stats] COCO: {coco_count}, Custom: {custom_count}, Total: {len(all_detections)}")
        
        if len(all_detections) == 0:
            return frame, None, None, None
        
        # Find closest obstacle
        direction, distance, closest = self._analyze_obstacles(all_detections, depth_map)
        
        # Draw all detections
        annotated_frame = self._draw_detections(frame, all_detections, closest)
        
        # Add overlay
        if direction and distance and closest:
            self._add_overlay(
                annotated_frame, direction, distance, 
                closest['class'], closest['source']
            )
        
        obstacle_class = closest['class'] if closest else None
        return annotated_frame, direction, distance, obstacle_class
    
    def _merge_detections(self, coco_dets, custom_dets):
        """Merge COCO and custom detections, removing duplicates"""
        all_detections = list(coco_dets)  # Start with COCO
        
        # Add custom detections if they don't overlap with COCO
        for custom_det in custom_dets:
            is_duplicate = False
            custom_box = custom_det['box']
            
            for coco_det in coco_dets:
                coco_box = coco_det['box']
                iou = calculate_iou(custom_box, coco_box)
                
                if iou > config.IOU_THRESHOLD_MERGE:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_detections.append(custom_det)
        
        return all_detections
    
    def _analyze_obstacles(self, detections, depth_map):
        """Find the closest obstacle and determine its direction/distance"""
        closest_obstacle = None
        closest_depth = 0.0  # Higher depth = closer
        
        for detection in detections:
            box = detection['box']
            
            # Handle both list and numpy array formats
            if hasattr(box, 'tolist'):
                x1, y1, x2, y2 = box.tolist()[:4]
            else:
                x1, y1, x2, y2 = box[:4]
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get depth at center
            depth_val = self.depth_estimator.get_distance_at_point(
                depth_map, center_x, center_y, radius=10
            )
            
            # Smart furniture depth priority
            if detection['class'] in config.FURNITURE_CLASSES:
                if depth_val < 0.3:
                    # Far furniture: additive boost
                    depth_val = min(depth_val + config.FURNITURE_DEPTH_ADDITIVE, 1.0)
                else:
                    # Close furniture: multiplicative boost
                    depth_val = min(depth_val * config.FURNITURE_DEPTH_BOOST, 1.0)
            
            # Higher depth = closer (due to inversion in depth estimator)
            if depth_val > closest_depth:
                closest_depth = depth_val
                closest_obstacle = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'depth': depth_val,
                    'box': [x1, y1, x2, y2],
                    'class': detection['class'],
                    'source': detection['source'],
                    'confidence': detection['confidence']
                }
        
        if closest_obstacle is None:
            return None, None, None
        
        # Determine direction based on horizontal position
        center_x = closest_obstacle['center_x']
        frame_center = self.frame_width / 2
        
        # Define boundaries using config
        left_boundary = frame_center * config.LEFT_BOUNDARY
        right_boundary = frame_center * config.RIGHT_BOUNDARY
        
        if center_x < left_boundary:
            direction = "left"
        elif center_x > right_boundary:
            direction = "right"
        else:
            direction = "center"
        
        # Determine distance based on depth using config thresholds
        if closest_depth > config.VERY_CLOSE_THRESHOLD:
            distance = "very_close"
        elif closest_depth > config.CLOSE_THRESHOLD:
            distance = "close"
        else:
            distance = "far"
        
        logger.debug(
            f"[Obstacle] {closest_obstacle['class']} "
            f"[{closest_obstacle['source'].upper()}]: "
            f"{direction} - {distance} (depth={closest_depth:.2f})"
        )
        
        return direction, distance, closest_obstacle
    
    def _draw_detections(self, frame, detections, closest):
        """Draw bounding boxes on frame"""
        annotated = frame.copy()
        
        for detection in detections:
            box = detection['box']
            
            # Handle both list and numpy array formats
            if hasattr(box, 'tolist'):
                x1, y1, x2, y2 = map(int, box.tolist()[:4])
            else:
                x1, y1, x2, y2 = map(int, box[:4])
            
            # Color based on source
            if detection['source'] == 'coco':
                color = (0, 255, 0)  # Green for COCO
                label_suffix = " [C]"
            else:  # custom
                color = (255, 0, 255)  # Magenta for Custom
                label_suffix = " [X]"
            
            # Check if this is the closest obstacle
            is_closest = False
            if closest:
                closest_box = closest['box']
                if (detection['class'] == closest['class'] and
                    abs(x1 - closest_box[0]) < 5):
                    is_closest = True
            
            if is_closest:
                color = (0, 0, 255)  # Red for closest
                thickness = 4
            else:
                thickness = 2
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{detection['class']} {detection['confidence']:.2f}{label_suffix}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
            
            label_y1 = max(y1 - label_h - 10, 0)
            label_y2 = max(y1, label_h + 10)
            
            cv2.rectangle(
                annotated, (x1, label_y1), (x1 + label_w + 10, label_y2), 
                color, -1
            )
            cv2.putText(
                annotated, label, (x1 + 5, label_y2 - 7), 
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return annotated
    
    def _add_overlay(self, frame, direction, distance, class_name, source):
        """Add text overlay showing current obstacle information"""
        # Color based on distance urgency
        if distance == "very_close":
            color = (0, 0, 255)  # Red
        elif distance == "close":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        distance_text = distance.upper().replace('_', ' ')
        source_text = f"[{'COCO' if source == 'coco' else 'CUSTOM'}]"
        text = f"{class_name.upper()}: {direction.upper()} - {distance_text} {source_text}"
        
        font = cv2.FONT_HERSHEY_BOLD
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
        description='Obstacle Avoidance System for Blind Assistance'
    )

    parser.add_argument(
        '--camera', type=int, default=config.CAMERA_INDEX,
        help=f'Camera index (default: {config.CAMERA_INDEX})'
    )
    parser.add_argument(
        '--custom-model', type=str, default='runs/detect/blind_navigation/obstacles_v1/weights/best.pt',
        help='Path to your custom trained YOLOv8 model'
    )
    parser.add_argument(
        '--use-yolo-medium', action='store_true',
        help='Use YOLOv8m (better accuracy, slower) instead of YOLOv8n (faster) for COCO'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("OBSTACLE AVOIDANCE SYSTEM FOR BLIND ASSISTANCE")
    print("=" * 60)
    model_name = "YOLOv8m" if args.use_yolo_medium else "YOLOv8n"
    print(f"Detection Mode: DUAL LOCAL MODELS ({model_name} COCO + Custom)")
    print(f"Custom Model: {args.custom_model}")
    print("Enhanced furniture detection with class-specific thresholds")
    
    # Camera setup
    logger.info("[Camera] Initializing camera...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error(f"[Camera] Could not open camera {args.camera}")
        print(f"\n[ERROR] Could not open camera {args.camera}. Please check connection.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    
    logger.info("[Camera] Camera initialized successfully")
    
    # Initialize depth estimator
    logger.info("[Depth] Initializing depth estimator...")
    try:
        depth_estimator = DepthEstimator()
    except Exception as e:
        logger.error(f"[Depth] Failed to initialize: {e}")
        print(f"\n[ERROR] Failed to initialize depth estimator: {e}")
        cap.release()
        return
    
    # Initialize dual model detector
    logger.info("[Detector] Initializing dual model detector...")
    try:
        detector = DualModelDetector(
            depth_estimator,
            custom_model_path=args.custom_model,
            use_yolo_medium=args.use_yolo_medium
        )
    except Exception as e:
        logger.error(f"[Detector] Failed to initialize: {e}")
        print(f"\n[ERROR] Failed to initialize detector: {e}")
        cap.release()
        return

    # Initialize speech engine
    logger.info("[Speech] Initializing speech engine...")
    try:
        speaker = SpeechEngine(cooldown=config.SPEECH_COOLDOWN)
    except Exception as e:
        logger.error(f"[Speech] Failed to initialize: {e}")
        print(f"\n[ERROR] Failed to initialize speech engine: {e}")
        cap.release()
        return
    
    # Initialize health monitor
    health_monitor = SystemHealthMonitor()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Toggle split view (camera + depth)")
    print("  'd' - Toggle debug info")
    print("=" * 60)
    print("\nColor Guide:")
    print("  Green [C]   = COCO YOLOv8 detections")
    print("  Magenta [X] = Custom model detections")
    print("  Red         = Closest obstacle")
    print("=" * 60)
    print(f"\nFeatures:")
    print(f"  - COCO furniture classes: {len(config.FURNITURE_CLASSES)}")
    print(f"  - Furniture confidence: {config.YOLO_CONFIDENCE_FURNITURE}")
    print(f"  - Standard confidence: {config.YOLO_CONFIDENCE}")
    print(f"  - Custom model confidence: 0.3")
    print(f"  - Furniture depth boost: {config.FURNITURE_DEPTH_BOOST}x")
    print("=" * 60)
    print()
    
    # Warm-up camera
    logger.info("[System] Warming up camera...")
    for _ in range(5):
        cap.read()
    
    # Display settings
    show_split_view = False
    show_debug = True
    
    # Performance monitoring
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    # Error recovery tracking
    consecutive_errors = 0
    max_consecutive_errors = config.MAX_CONSECUTIVE_ERRORS
    
    # Main loop
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("[Camera] Failed to read frame")
                time.sleep(0.1)
                continue
            
            health_monitor.update_camera()

            # Process frame
            try:
                annotated_frame, direction, distance, obstacle_class = detector.process(frame)
                
                health_monitor.update_depth()
                health_monitor.update_detection()
                consecutive_errors = 0
                
            except Exception as e:
                logger.error(f"[Processing] Error: {e}", exc_info=True)
                annotated_frame = frame
                direction, distance, obstacle_class = None, None, None
                
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("[Processing] Too many consecutive errors, reinitializing...")
                    try:
                        detector = DualModelDetector(
                            depth_estimator,
                            custom_model_path=args.custom_model,
                            use_yolo_medium=args.use_yolo_medium
                        )
                        consecutive_errors = 0
                        logger.info("[Processing] Reinitialization successful")
                    except Exception as reinit_error:
                        logger.critical(f"[Processing] Reinitialization failed: {reinit_error}")
                        break

            # Audio feedback
            if direction is not None and distance is not None:
                speaker.speak(direction, distance, obstacle_class)

            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start
                fps_display = 30 / elapsed if elapsed > 0 else 0
                
                if show_debug:
                    status = f"{direction} - {distance}" if direction else "No obstacles"
                    logger.info(f"[Status] FPS: {fps_display:.1f} | {status}")
                
                if fps_counter % config.HEALTH_CHECK_INTERVAL == 0:
                    if not health_monitor.check_health():
                        logger.warning("[System] Health check failed, monitor system")
                
                fps_start = time.time()

            # Create display frame
            if show_split_view:
                try:
                    depth_map, depth_vis = depth_estimator.estimate_with_visualization(frame)
                    
                    display_height = 480
                    display_width = 640
                    frame_resized = cv2.resize(annotated_frame, (display_width, display_height))
                    depth_resized = cv2.resize(depth_vis, (display_width, display_height))
                    
                    cv2.putText(
                        frame_resized, "CAMERA", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        depth_resized, "DEPTH", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    
                    display_frame = np.hstack([frame_resized, depth_resized])
                except Exception as e:
                    logger.error(f"[Display] Split view error: {e}")
                    display_frame = annotated_frame
            else:
                display_frame = annotated_frame

            # Add overlays
            if show_debug:
                cv2.putText(
                    display_frame, f"FPS: {fps_display:.1f}", 
                    (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                
                model_text = f"{'YOLOv8m' if args.use_yolo_medium else 'YOLOv8n'}+Custom"
                cv2.putText(
                    display_frame, model_text, 
                    (10, display_frame.shape[0] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

            cv2.imshow("Obstacle Avoidance System", display_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("[System] Shutting down...")
                break
            elif key == ord('s'):
                show_split_view = not show_split_view
                mode = "Split View" if show_split_view else "Camera Only"
                logger.info(f"[Display] Switched to {mode}")
            elif key == ord('d'):
                show_debug = not show_debug
                status = "enabled" if show_debug else "disabled"
                logger.info(f"[Display] Debug info {status}")

    except KeyboardInterrupt:
        logger.info("[System] Interrupted by user")
    
    except Exception as e:
        logger.error(f"[System] Unexpected error: {e}", exc_info=True)
    
    finally:
        logger.info("[System] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        speaker.stop()
        logger.info("[System] Shutdown complete")
        print("=" * 60)


if __name__ == "__main__":
    main()