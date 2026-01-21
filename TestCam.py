import cv2
import numpy as np
from ultralytics import YOLO

class ObstacleDetector:
    def __init__(self, custom_model_path, depth_estimator, use_coco=True):
        # Load primary model
        self.custom_model = YOLO(custom_model_path)
        print(f"[ObstacleDetector] Primary model loaded: {custom_model_path}")
        print(f"[ObstacleDetector] Primary model classes: {self.custom_model.names}")
        
        # Check if we should use dual model system
        self.is_custom_model = custom_model_path != "yolov8n.pt"
        
        # Load COCO model if requested AND if primary is custom
        self.coco_model = None
        self.use_coco = use_coco and self.is_custom_model
        
        if self.use_coco:
            self.coco_model = YOLO('yolov8n.pt')
            print(f"[ObstacleDetector] COCO model loaded: yolov8n.pt")
            print(f"[ObstacleDetector] Running in DUAL MODEL mode")
        else:
            print(f"[ObstacleDetector] Running in SINGLE MODEL mode")
        
        self.depth_estimator = depth_estimator
        self.frame_width = None
        self.frame_height = None
        
        # Define obstacle classes from COCO dataset (for navigation assistance)
        self.coco_obstacle_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'bench', 'backpack', 'suitcase', 'handbag', 'bottle'
        ]
        
        if self.use_coco:
            print(f"[ObstacleDetector] COCO obstacle classes: {self.coco_obstacle_classes}")

    def process(self, frame):

        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Get depth map
        depth_map, depth_vis = self.depth_estimator.estimate_with_visualization(frame)
        
        # Run primary model
        try:
            primary_results = self.custom_model(frame, conf=0.5, verbose=False)
            primary_detections = primary_results[0].boxes
        except Exception as e:
            print(f"[ObstacleDetector] Error in primary model: {e}")
            return frame, None, None, None
        
        # Run COCO model if enabled
        coco_detections = []
        coco_results = None
        if self.use_coco and self.coco_model:
            try:
                coco_results = self.coco_model(frame, conf=0.5, verbose=False)
                coco_detections = self._filter_coco_obstacles(coco_results[0].boxes)
            except Exception as e:
                print(f"[ObstacleDetector] Error in COCO model: {e}")
        
        # Combine all detections if dual mode
        if self.use_coco:
            all_detections = self._merge_detections(
                primary_detections, 
                coco_detections,
                primary_results[0],
                coco_results[0] if coco_results else None
            )
        else:
            # Single model mode
            all_detections = []
            for box in primary_detections:
                class_id = int(box.cls[0])
                all_detections.append({
                    'box': box,
                    'class_name': self.custom_model.names[class_id],
                    'source': 'primary',
                    'model': self.custom_model
                })
        
        if len(all_detections) == 0:
            return frame, None, None, None
        
        # Analyze closest obstacle
        direction, distance, closest = self._analyze_obstacles(all_detections, depth_map)
        
        # Draw all detections on frame
        annotated_frame = self._draw_detections(frame, all_detections, closest)
        
        # Add overlay
        if direction and distance and closest:
            self._add_overlay(annotated_frame, direction, distance, closest['class'], closest['source'])
        
        obstacle_class = closest['class'] if closest else None
        return annotated_frame, direction, distance, obstacle_class
    
    def _filter_coco_obstacles(self, coco_boxes):

        obstacle_boxes = []
        
        for box in coco_boxes:
            class_id = int(box.cls[0])
            class_name = self.coco_model.names[class_id]
            
            if class_name in self.coco_obstacle_classes:
                obstacle_boxes.append(box)
        
        return obstacle_boxes
    
    def _merge_detections(self, primary_boxes, coco_boxes, primary_result, coco_result):
        all_detections = []
        
        # Add primary model detections
        for box in primary_boxes:
            class_id = int(box.cls[0])
            all_detections.append({
                'box': box,
                'class_name': self.custom_model.names[class_id],
                'source': 'custom',
                'model': self.custom_model
            })
        
        # Add COCO model detections (check for duplicates)
        for box in coco_boxes:
            class_id = int(box.cls[0])
            
            # Check for overlap with primary detections
            is_duplicate = False
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            for primary_det in all_detections:
                if primary_det['source'] == 'custom':
                    px1, py1, px2, py2 = primary_det['box'].xyxy[0].cpu().numpy()
                    iou = self._calculate_iou(x1, y1, x2, y2, px1, py1, px2, py2)
                    
                    # If IoU > 0.5, consider it a duplicate
                    if iou > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                all_detections.append({
                    'box': box,
                    'class_name': self.coco_model.names[class_id],
                    'source': 'coco',
                    'model': self.coco_model
                })
        
        return all_detections
    
    def _calculate_iou(self, x1, y1, x2, y2, x1b, y1b, x2b, y2b):
        """Calculate Intersection over Union between two boxes"""
        # Intersection area
        xi1 = max(x1, x1b)
        yi1 = max(y1, y1b)
        xi2 = min(x2, x2b)
        yi2 = min(y2, y2b)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2b - x1b) * (y2b - y1b)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _analyze_obstacles(self, detections, depth_map):
        closest_obstacle = None
        closest_depth = 0.0  # We want the MAXIMUM depth value (closest object)
        
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get depth at box center
            depth_val = self.depth_estimator.get_distance_at_point(
                depth_map, center_x, center_y, radius=10
            )
            
            # Higher depth value = closer object
            if depth_val > closest_depth:
                closest_depth = depth_val
                closest_obstacle = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'depth': depth_val,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'class': detection['class_name'],
                    'source': detection['source'],
                    'confidence': float(box.conf[0])
                }
        
        if closest_obstacle is None:
            return None, None, None
        
        # FIXED: Determine direction based on horizontal position
        center_x = closest_obstacle['center_x']
        frame_center = self.frame_width / 2
        
        # Define boundaries (40% on left, 40% on right, 20% center)
        left_boundary = frame_center * 0.7   # 35% of frame width
        right_boundary = frame_center * 1.3  # 65% of frame width
        
        if center_x < left_boundary:
            direction = "left"
        elif center_x > right_boundary:
            direction = "right"
        else:
            direction = "center"
        
        # Determine distance based on depth (higher value = closer)
        if closest_depth > 0.75:
            distance = "very_close"
        elif closest_depth > 0.5:
            distance = "close"
        else:
            distance = "far"
        
        # Debug output
        print(f"[Obstacle] {closest_obstacle['class']}: {direction} - {distance} (depth={closest_depth:.2f})")
        
        return direction, distance, closest_obstacle
    
    def _draw_detections(self, frame, detections, closest):
        annotated = frame.copy()
        
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Color scheme based on source
            if self.use_coco:
                if detection['source'] == 'custom':
                    color = (255, 100, 0)  # Blue for custom model
                    label_bg = (255, 100, 0)
                else:
                    color = (0, 200, 0)  # Green for COCO model
                    label_bg = (0, 200, 0)
            else:
                color = (0, 255, 0)  # Green for single model
                label_bg = (0, 255, 0)
            
            # Highlight closest obstacle in red
            is_closest = False
            if closest:
                is_closest = (detection['class'] == closest['class'] and
                            abs(x1 - closest['x1']) < 5 and
                            abs(y1 - closest['y1']) < 5)
            
            if is_closest:
                color = (0, 0, 255)  # Red for closest
                label_bg = (0, 0, 255)
                thickness = 4
            else:
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            conf = float(box.conf[0])
            label = f"{detection['class_name']} {conf:.2f}"
            
            if self.use_coco:
                if detection['source'] == 'custom':
                    label += " [C]"
                else:
                    label += " [D]"  # D for default/COCO
            
            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
            
            # Ensure label stays within frame
            label_y1 = max(y1 - label_h - 10, 0)
            label_y2 = max(y1, label_h + 10)
            
            cv2.rectangle(annotated, (x1, label_y1), (x1 + label_w + 10, label_y2), label_bg, -1)
            cv2.putText(annotated, label, (x1 + 5, label_y2 - 7), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        return annotated
    
    def _add_overlay(self, frame, direction, distance, class_name, source):
        """
        Add text overlay showing current obstacle info at top of frame
        """
        # Color based on distance urgency
        if distance == "very_close":
            color = (0, 0, 255)  # Red
        elif distance == "close":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Build main text
        distance_text = distance.upper().replace('_', ' ')
        text = f"{class_name.upper()}: {direction.upper()} - {distance_text}"
        
        # Build source indicator
        if self.use_coco:
            source_text = f"[{'CUSTOM' if source == 'custom' else 'COCO'}]"
        else:
            source_text = ""
        
        # Font settings
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 0.8
        thickness = 2
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        if source_text:
            (source_w, source_h), _ = cv2.getTextSize(source_text, font, 0.5, 1)
        else:
            source_w, source_h = 0, 0
        
        # Position at top center
        x = (frame.shape[1] - text_width) // 2
        y = 40
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding), 
                     (x + text_width + padding, y + baseline + padding),
                     (0, 0, 0), -1)
        
        # Draw main text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Draw source indicator if available
        if source_text:
            cv2.putText(frame, source_text, (x + text_width - source_w, y + 22), 
                       font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    def get_obstacle_summary(self, detections, depth_map):
        summary = {
            'left': {'count': 0, 'closest_depth': 0.0},
            'center': {'count': 0, 'closest_depth': 0.0},
            'right': {'count': 0, 'closest_depth': 0.0}
        }
        
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            depth_val = self.depth_estimator.get_distance_at_point(depth_map, center_x, center_y)
            
            # Only count obstacles that are close enough to matter
            if depth_val > 0.4:
                frame_center = self.frame_width / 2
                
                if center_x < frame_center * 0.7:
                    dir_key = 'left'
                elif center_x > frame_center * 1.3:
                    dir_key = 'right'
                else:
                    dir_key = 'center'
                
                summary[dir_key]['count'] += 1
                if depth_val > summary[dir_key]['closest_depth']:
                    summary[dir_key]['closest_depth'] = depth_val
        
        return summary