import cv2
import time
import numpy as np
import argparse
import logging
from depth import DepthEstimator
from Speech import SpeechEngine
from ultralytics import YOLO
from health_check import SystemHealthMonitor
from camera_utils import JetsonCamera
import config
import torch
import gc

print("=" * 60)
print("GPU STATUS CHECK")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Count: {torch.cuda.device_count()}")
print("=" * 60)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_iou(box1, box2):
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
    
    def __init__(self, depth_estimator, custom_model_path='runs/detect/blind_navigation/obstacles_v1/weights/best.pt', use_yolo_medium=False):
        model_name = 'yolov8m.pt' if use_yolo_medium else 'yolov8n.pt'
        logger.warning(f"[DualModel] Loading COCO model: {model_name}...")
        
        device = 'cuda:0' if config.CUDA_DEVICE == 0 else 'cpu'
        self.yolo_coco = YOLO(model_name)
        self.yolo_coco.to(device)
        self.yolo_coco.overrides['verbose'] = False
        
        logger.warning(f"[DualModel] Loading custom model: {custom_model_path}...")
        try:
            self.yolo_custom = YOLO(custom_model_path)
            self.yolo_custom.to(device)
            self.yolo_custom.overrides['verbose'] = False
            logger.warning("[DualModel] Custom model loaded")
        except Exception as e:
            logger.error(f"[DualModel] Failed to load custom model: {e}")
            raise
        
        self.depth_estimator = depth_estimator
        self.frame_width = None
        self.frame_height = None
        
        self.coco_obstacle_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'bench', 'backpack', 'suitcase', 'handbag', 'bottle'
        ]
        
        self.frame_count = 0
        
        self.cached_depth_map = None
        self.cached_yolo_coco = []
        self.cached_yolo_custom = []
        
        logger.warning(f"[DualModel] Frame skipping - Depth:{config.DEPTH_FRAME_SKIP} COCO:{config.YOLO_COCO_FRAME_SKIP} Custom:{config.YOLO_CUSTOM_FRAME_SKIP}")
    
    def process(self, frame):
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_count += 1
        
        if self.frame_count % config.DEPTH_FRAME_SKIP == 0:
            try:
                depth_map = self.depth_estimator.estimate(frame)
                self.cached_depth_map = depth_map
            except Exception as e:
                logger.error(f"[Depth] Error: {e}")
                if self.cached_depth_map is None:
                    return frame, None, None, None
        else:
            depth_map = self.cached_depth_map
            if depth_map is None:
                return frame, None, None, None
        
        if self.frame_count % config.YOLO_COCO_FRAME_SKIP == 0:
            try:
                yolo_results = self.yolo_coco(
                    frame, 
                    conf=config.YOLO_CONFIDENCE_FURNITURE,
                    imgsz=config.YOLO_IMG_SIZE,
                    max_det=config.YOLO_MAX_DET,
                    agnostic_nms=config.YOLO_AGNOSTIC_NMS,
                    verbose=False,
                    device=config.CUDA_DEVICE,
                    half=config.USE_HALF_PRECISION
                )
                yolo_boxes = yolo_results[0].boxes
                
                yolo_detections = []
                for box in yolo_boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_coco.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name in self.coco_obstacle_classes:
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
                
                self.cached_yolo_coco = yolo_detections
            except Exception as e:
                logger.error(f"[COCO] Error: {e}")
                yolo_detections = self.cached_yolo_coco
        else:
            yolo_detections = self.cached_yolo_coco
        
        if self.frame_count % config.YOLO_CUSTOM_FRAME_SKIP == 0:
            try:
                custom_results = self.yolo_custom(
                    frame,
                    conf=0.25,
                    imgsz=config.YOLO_IMG_SIZE,
                    max_det=config.YOLO_MAX_DET,
                    agnostic_nms=config.YOLO_AGNOSTIC_NMS,
                    verbose=False,
                    device=config.CUDA_DEVICE,
                    half=config.USE_HALF_PRECISION
                )
                custom_boxes = custom_results[0].boxes
                
                custom_detections = []
                for box in custom_boxes:
                    try:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_custom.names[class_id]
                        confidence = float(box.conf[0])
                        
                        min_confidence = config.CUSTOM_MODEL_CONFIDENCE.get(
                            class_name.lower(), 
                            config.CUSTOM_MODEL_CONFIDENCE['default']
                        )
                        
                        if confidence >= min_confidence:
                            custom_detections.append({
                                'box': box.xyxy[0].cpu().numpy(),
                                'class': class_name,
                                'confidence': confidence,
                                'source': 'custom'
                            })
                    except Exception as box_error:
                        continue
                
                self.cached_yolo_custom = custom_detections
                
            except Exception as e:
                logger.error(f"[Custom] Error: {e}")
                custom_detections = self.cached_yolo_custom
        else:
            custom_detections = self.cached_yolo_custom
        
        all_detections = self._merge_detections(yolo_detections, custom_detections)
        
        if len(all_detections) == 0:
            if self.frame_count % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            return frame, None, None, None
        
        direction, distance, closest = self._analyze_obstacles(all_detections, depth_map)
        annotated_frame = self._draw_detections(frame, all_detections, closest)
        
        if direction and distance and closest:
            self._add_overlay(
                annotated_frame, direction, distance, 
                closest['class'], closest['source']
            )
        
        if self.frame_count % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        obstacle_class = closest['class'] if closest else None
        return annotated_frame, direction, distance, obstacle_class
    
    def _merge_detections(self, coco_dets, custom_dets):
        all_detections = list(coco_dets)
        
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
        closest_obstacle = None
        closest_depth = 0.0
        
        for detection in detections:
            box = detection['box']
            
            if hasattr(box, 'tolist'):
                x1, y1, x2, y2 = box.tolist()[:4]
            else:
                x1, y1, x2, y2 = box[:4]
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            depth_val = self.depth_estimator.get_distance_at_point(
                depth_map, center_x, center_y, radius=10
            )
            
            if detection['class'] in config.FURNITURE_CLASSES:
                if depth_val < 0.3:
                    depth_val = min(depth_val + config.FURNITURE_DEPTH_ADDITIVE, 1.0)
                else:
                    depth_val = min(depth_val * config.FURNITURE_DEPTH_BOOST, 1.0)
            
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
        
        center_x = closest_obstacle['center_x']
        frame_center = self.frame_width / 2
        
        left_boundary = frame_center * config.LEFT_BOUNDARY
        right_boundary = frame_center * config.RIGHT_BOUNDARY
        
        if center_x < left_boundary:
            direction = "left"
        elif center_x > right_boundary:
            direction = "right"
        else:
            direction = "center"
        
        if closest_depth > config.VERY_CLOSE_THRESHOLD:
            distance = "very_close"
        elif closest_depth > config.CLOSE_THRESHOLD:
            distance = "close"
        else:
            distance = "far"
        
        return direction, distance, closest_obstacle
    
    def _draw_detections(self, frame, detections, closest):
        annotated = frame.copy()
        
        for detection in detections:
            box = detection['box']
            
            if hasattr(box, 'tolist'):
                x1, y1, x2, y2 = map(int, box.tolist()[:4])
            else:
                x1, y1, x2, y2 = map(int, box[:4])
            
            if detection['source'] == 'coco':
                color = (0, 255, 0)
                label_suffix = " [C]"
            else:
                color = (255, 0, 255)
                label_suffix = " [X]"
            
            is_closest = False
            if closest:
                closest_box = closest['box']
                if (detection['class'] == closest['class'] and
                    abs(x1 - closest_box[0]) < 5):
                    is_closest = True
            
            if is_closest:
                color = (0, 0, 255)
                thickness = 4
            else:
                thickness = 2
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
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
        description='Obstacle Avoidance System - Ultra Performance Mode'
    )

    parser.add_argument(
        '--camera', type=int, default=config.CAMERA_INDEX,
        help=f'Camera index (default: {config.CAMERA_INDEX})'
    )
    parser.add_argument(
        '--custom-model', type=str, default='runs/detect/blind_navigation/obstacles_v1/weights/best.pt',
        help='Path to custom YOLOv8 model'
    )
    parser.add_argument(
        '--use-yolo-medium', action='store_true',
        help='Use YOLOv8m instead of YOLOv8n'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Disable video display for max performance'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("JETSON ORIN NANO - ULTRA PERFORMANCE MODE")
    print("=" * 60)
    model_name = "YOLOv8m" if args.use_yolo_medium else "YOLOv8n"
    print(f"Detection: {model_name} COCO + Custom")
    print(f"Frame Skip: Depth={config.DEPTH_FRAME_SKIP}, COCO={config.YOLO_COCO_FRAME_SKIP}, Custom={config.YOLO_CUSTOM_FRAME_SKIP}")
    print(f"YOLO Size: {config.YOLO_IMG_SIZE}px")
    print(f"FP16: {config.USE_HALF_PRECISION}")
    
    logger.warning("[Camera] Initializing...")
    try:
        cap = JetsonCamera(
            camera_id=config.CAMERA_INDEX, 
            width=config.CAMERA_WIDTH, 
            height=config.CAMERA_HEIGHT, 
            fps=config.CAMERA_FPS
        )
        camera_info = cap.get_info()
        print(f"Camera: {camera_info['width']}x{camera_info['height']} @ {camera_info['fps']}fps")
    except Exception as e:
        print(f"[ERROR] Camera failed: {e}")
        return
    
    logger.warning("[Depth] Initializing...")
    try:
        depth_estimator = DepthEstimator()
    except Exception as e:
        print(f"[ERROR] Depth failed: {e}")
        cap.release()
        return
    
    logger.warning("[Detector] Initializing...")
    try:
        detector = DualModelDetector(
            depth_estimator,
            custom_model_path=args.custom_model,
            use_yolo_medium=args.use_yolo_medium
        )
    except Exception as e:
        print(f"[ERROR] Detector failed: {e}")
        cap.release()
        return

    logger.warning("[Speech] Initializing...")
    try:
        speaker = SpeechEngine(cooldown=config.SPEECH_COOLDOWN)
    except Exception as e:
        print(f"[ERROR] Speech failed: {e}")
        cap.release()
        return
    
    health_monitor = SystemHealthMonitor()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY - Press 'q' to quit")
    print("=" * 60)
    
    # Warm-up
    for _ in range(5):
        cap.read()
    
    show_display = not args.no_display
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    consecutive_errors = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                time.sleep(0.1)
                continue
            
            health_monitor.update_camera()

            try:
                annotated_frame, direction, distance, obstacle_class = detector.process(frame)
                health_monitor.update_depth()
                health_monitor.update_detection()
                consecutive_errors = 0
                
            except Exception as e:
                logger.error(f"[Processing] Error: {e}")
                annotated_frame = frame
                direction, distance, obstacle_class = None, None, None
                consecutive_errors += 1
                
                if consecutive_errors >= config.MAX_CONSECUTIVE_ERRORS:
                    logger.error("[Processing] Too many errors, reinitializing...")
                    try:
                        detector = DualModelDetector(
                            depth_estimator,
                            custom_model_path=args.custom_model,
                            use_yolo_medium=args.use_yolo_medium
                        )
                        consecutive_errors = 0
                    except Exception as reinit_error:
                        logger.critical(f"[Reinit] Failed: {reinit_error}")
                        break

            if direction is not None and distance is not None:
                speaker.speak(direction, distance, obstacle_class)

            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start
                fps_display = 30 / elapsed if elapsed > 0 else 0
                
                status = f"{direction} - {distance}" if direction else "Clear"
                print(f"[FPS: {fps_display:.1f}] {status}")
                
                if fps_counter % config.HEALTH_CHECK_INTERVAL == 0:
                    health_monitor.check_health()
                
                fps_start = time.time()

            if show_display:
                cv2.putText(
                    annotated_frame, f"FPS: {fps_display:.1f}", 
                    (10, annotated_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.imshow("Obstacle Avoidance", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[System] Interrupted")
    except Exception as e:
        logger.error(f"[System] Error: {e}", exc_info=True)
    finally:
        print("[System] Cleaning up...")
        cap.release()
        if show_display:
            cv2.destroyAllWindows()
        speaker.stop()
        print("Shutdown complete")


if __name__ == "__main__":
    main()