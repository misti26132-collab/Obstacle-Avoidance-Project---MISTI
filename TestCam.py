import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ObstacleDetector:
    def __init__(self, model_path):
        """
        Initialize obstacle detector with YOLO and depth estimation
        
        Args:
            model_path: Path to YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.depth_estimator = DepthEstimator()
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
            distance: "very close" or "moderate" (None if no obstacles)
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
            distance: "very close" or "moderate"
        """
        closest_obstacle = None
        closest_depth = 1.0
        
        # Find closest obstacle
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get depth at box center
            depth_val = self.depth_estimator.get_distance_at_point(
                depth_map, center_x, center_y, radius=10
            )
            
            if depth_val < closest_depth:
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
        elif center_x > frame_center * 0.6:
            direction = "right"
        else:
            direction = "center"
        
        # Determine distance based on depth
        distance = "very close" if closest_depth < 0.4 else "moderate"
        
        return direction, distance


class DepthEstimator:
    def __init__(self, device=None):
        """
        Initialize MiDaS depth estimator
        
        Args:
            device: torch.device or None (auto-detects best available)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"DepthEstimator using device: {self.device}")

        # Load MiDaS model with error handling
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.midas.to(self.device)
            self.midas.eval()
            
            # Load transforms
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.small_transform
            
            print("MiDaS model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MiDaS model: {e}")

    def estimate(self, frame):
        """
        Estimate depth from BGR frame
        
        Args:
            frame: OpenCV BGR image (numpy array)
            
        Returns:
            depth_map: Normalized depth map where:
                      - Lower values (closer to 0) = CLOSER objects
                      - Higher values (closer to 1) = FARTHER objects
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided to depth estimator")

        # Convert BGR -> RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_batch = self.transform(img).to(self.device)

        # Inference
        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Resize prediction to match input size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy
        depth_map_raw = prediction.cpu().numpy()

        # ==========================================================
        # FIXED SCALING LOGIC
        # ==========================================================
        # MiDaS outputs inverse depth (disparity).
        # We use a FIXED scale factor to prevent values from "jumping" 
        # between frames (The "Elastic Ruler" bug).
        # 800.0 is a robust baseline for MiDaS_small in indoor/mixed settings.
        DISPARITY_SCALE = 800.0
        
        # Normalize raw disparity to 0..1 range based on fixed scale
        depth_map = depth_map_raw / DISPARITY_SCALE
        
        # Clip to ensure valid range [0, 1]
        depth_map = np.clip(depth_map, 0, 1)
        
        # Invert to match your telemetry logs:
        # MiDaS Raw: High Value = CLOSE
        # Your Logs: Low Value (0.0) = CLOSE
        # Therefore:
        depth_map = 1.0 - depth_map

        return depth_map

    def estimate_with_visualization(self, frame):
        """
        Estimate depth and return both depth map and visualization
        
        Returns:
            depth_map: Normalized depth map (0.0=Close, 1.0=Far)
            depth_vis: Colorized depth map for visualization (BGR)
                       (Bright Yellow = Close/Danger, Purple = Far)
        """
        depth_map = self.estimate(frame)
        
        # For Visualization:
        # We want CLOSE objects (0.0 in depth_map) to appear BRIGHT (255)
        # We want FAR objects (1.0 in depth_map) to appear DARK (0)
        
        vis_raw = (1.0 - depth_map) * 255
        depth_vis = vis_raw.astype(np.uint8)
        
        # Apply colormap (MAGMA: 0=Black/Purple, 255=Bright Yellow)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        return depth_map, depth_colored
    
    def get_distance_at_point(self, depth_map, x, y, radius=5):
        """
        Get average depth around a specific point
        
        Args:
            depth_map: Normalized depth map
            x, y: Point coordinates
            radius: Sampling radius in pixels
            
        Returns:
            Average depth value at the point
        """
        h, w = depth_map.shape
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(w, int(x + radius))
        y2 = min(h, int(y + radius))
        
        region = depth_map[y1:y2, x1:x2]
        return region.mean() if region.size > 0 else 0.5