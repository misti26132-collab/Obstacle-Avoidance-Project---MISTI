import cv2
import numpy as np
from ultralytics import YOLO

class ObstacleDetector:
    def __init__(self, model_path, depth_estimator):
        self.model = YOLO(model_path)
        self.depth_estimator = depth_estimator
        self.frame_width = None
        self.frame_height = None

    def process(self, frame):
        """
        Process frame to detect obstacles and estimate direction/distance
        
        Args:
            frame: Input frame from camera
            
        Returns:
            frame: Annotated frame with bounding boxes
            direction: "left", "center", or "right" (None if no obstacles)
            distance: "very_close", "close", or "far" (None if no obstacles)
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Get YOLO detections
        results = self.model(frame, conf=0.5)
        
        # Get depth map
        depth_map, depth_vis = self.depth_estimator.estimate_with_visualization(frame)
        
        # Find obstacles (any detections)
        detections = results[0].boxes
        
        if len(detections) == 0:
            return frame, None, None
        
        # Analyze closest obstacle
        direction, distance = self._analyze_obstacles(detections, depth_map, frame)
        
        # Draw YOLO detections on frame
        annotated_frame = results[0].plot()
        
        return annotated_frame, direction, distance
    
    def _analyze_obstacles(self, detections, depth_map, frame):
        """
        Analyze detected obstacles to determine direction and distance
        
        Returns:
            direction: "left", "center", or "right"
            distance: "very_close", "close", or "far"
        """
        closest_obstacle = None
        closest_depth = 0.0  # We want the MAXIMUM depth value (closest object)
        
        # Find closest obstacle
        for box in detections:
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
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                }
        
        if closest_obstacle is None:
            return None, None
        
        # Determine direction based on horizontal position
        center_x = closest_obstacle['center_x']
        frame_center = self.frame_width / 2
        
        if center_x < frame_center * 0.4:
            direction = "left"
        elif center_x > frame_center * 1.6:
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
        
        return direction, distance