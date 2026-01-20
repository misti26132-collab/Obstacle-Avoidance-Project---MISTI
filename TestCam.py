import torch
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class ObstacleDetector:
    def __init__(self, model_path, depth_estimator):
        """
        Initialize obstacle detector with performance optimizations
        
        Args:
            model_path: Path to YOLOv8 model weights
            depth_estimator: Shared DepthEstimator instance
        """
        self.model = YOLO(model_path)
        self.depth_estimator = depth_estimator
        self.frame_width = None
        self.frame_height = None
        self.frame_count = 0
        self.confidence_threshold = 0.45
        self.min_box_area = 500
        
        # PERFORMANCE OPTIMIZATIONS
        self.process_every_n_frames = 3  # Process every 3rd frame (reduced debug spam)
        self.last_result = (None, None, None)  # Cache last detection
        self.result_history = deque(maxlen=5)  # Smooth over more frames
        
    def process(self, frame):
        """Process frame with frame-skipping optimization"""
        try:
            if frame is None or frame.size == 0:
                return frame, None, None
                
            self.frame_height, self.frame_width = frame.shape[:2]
            self.frame_count += 1
            
            # OPTIMIZATION: Skip frames for performance
            if self.frame_count % self.process_every_n_frames != 0:
                # Return cached result with original frame
                direction, distance, _ = self.last_result
                if direction:
                    frame = self._draw_annotations(frame, direction, distance)
                return frame, direction, distance
            
            # === ACTUAL PROCESSING (every Nth frame) ===
            
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = results[0].boxes
            
            if len(detections) == 0:
                self.last_result = (None, None, None)
                return frame, None, None
            
            # Get depth map (only when needed)
            depth_map, _ = self.depth_estimator.estimate_with_visualization(frame)
            
            # Analyze obstacles
            direction, distance, closest_box = self._analyze_obstacles(
                detections, depth_map, frame
            )
            
            # Smooth results over time
            self.result_history.append((direction, distance))
            direction, distance = self._smooth_results()
            
            # Cache result
            self.last_result = (direction, distance, closest_box)
            
            # Draw annotations
            annotated_frame = results[0].plot()
            if direction and distance:
                annotated_frame = self._draw_annotations(annotated_frame, direction, distance)
            
            return annotated_frame, direction, distance
            
        except Exception as e:
            print(f"Error in process: {e}")
            return frame, None, None
    
    def _smooth_results(self):
        """Smooth detection results to reduce jitter"""
        if len(self.result_history) < 2:
            return self.result_history[-1] if self.result_history else (None, None)
        
        # Count occurrences of each direction
        directions = [r[0] for r in self.result_history if r[0]]
        distances = [r[1] for r in self.result_history if r[1]]
        
        if not directions:
            return None, None
        
        # Use most common direction
        direction = max(set(directions), key=directions.count)
        
        # Use most critical distance (prioritize safety)
        distance_priority = {"very close": 3, "close": 2, "far": 1}
        distance = max(distances, key=lambda d: distance_priority.get(d, 0))
        
        return direction, distance
    
    def _analyze_obstacles(self, detections, depth_map, frame):
        """
        Analyze detected obstacles with improved filtering
        """
        closest_obstacle = None
        closest_depth = 1.0
        
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Filter small detections
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < self.min_box_area:
                continue
            
            # Get depth (reduced radius for performance)
            depth_val = self.depth_estimator.get_distance_at_point(
                depth_map, center_x, center_y, radius=8
            )
            
            # Find closest significant obstacle
            if depth_val < closest_depth and depth_val < 0.9:
                closest_depth = depth_val
                closest_obstacle = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'depth': depth_val,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'area': box_area
                }
        
        if closest_obstacle is None:
            return None, None, None
        
        # Determine direction
        center_x = closest_obstacle['center_x']
        frame_center = self.frame_width / 2
        
        # Symmetric thresholds - divide frame into 3 equal zones
        zone_width = self.frame_width / 3
        
        if center_x < zone_width:
            direction = "left"
        elif center_x > 2 * zone_width:
            direction = "right"
        else:
            direction = "center"
        
        # Distance classification - ORIGINAL THRESHOLDS
        depth = closest_obstacle['depth']
        if depth < 0.25:
            distance = "very close"
        elif depth < 0.55:
            distance = "close"
        else:
            distance = "far"
        
        return direction, distance, closest_obstacle
    
    def _draw_annotations(self, frame, direction, distance):
        """Draw subtle direction and distance information"""
        text = f"{direction.upper()} - {distance.upper()}"
        
        if distance == "very close":
            color = (0, 0, 255)  # Red
        elif distance == "close":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Subtle semi-transparent overlay (top of screen)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Text with colored indicator dot
        cv2.circle(frame, (20, 25), 8, color, -1)
        cv2.putText(frame, text, (40, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame