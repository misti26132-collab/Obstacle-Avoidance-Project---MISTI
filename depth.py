import torch
import cv2
import numpy as np
from collections import deque

class DepthEstimator:
    def __init__(self, device=None):
        """
        Initialize MiDaS depth estimator with adaptive scaling
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"DepthEstimator using device: {self.device}")

        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.midas.to(self.device)
            self.midas.eval()
            
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.small_transform
            
            print("MiDaS model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MiDaS model: {e}")
        
        # Adaptive scaling components
        self.scale_history = deque(maxlen=30)  # Track last 30 frames
        self.min_history = deque(maxlen=30)
        self.initialized = False

    def estimate(self, frame):
        """
        Estimate depth with ADAPTIVE scaling to prevent inconsistencies
        
        Returns:
            depth_map: Normalized depth map where:
                      - Lower values (closer to 0) = CLOSER objects
                      - Higher values (closer to 1) = FARTHER objects
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided to depth estimator")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map_raw = prediction.cpu().numpy()

        # ==========================================================
        # ADAPTIVE SCALING - Fixes inconsistent depth detection
        # ==========================================================
        
        # Calculate percentiles for this frame
        p5 = np.percentile(depth_map_raw, 5)   # Near objects
        p95 = np.percentile(depth_map_raw, 95) # Far objects
        
        # Store in history
        self.min_history.append(p5)
        self.scale_history.append(p95)
        
        # Use smoothed values (prevents jumping between frames)
        if len(self.scale_history) >= 10:
            smooth_min = np.median(self.min_history)
            smooth_max = np.median(self.scale_history)
            self.initialized = True
        else:
            # Initial frames - use current values
            smooth_min = p5
            smooth_max = p95
        
        # Normalize to 0-1 range with adaptive scaling
        depth_range = smooth_max - smooth_min
        if depth_range < 1e-6:  # Prevent division by zero
            depth_map = np.zeros_like(depth_map_raw)
        else:
            depth_map = (depth_map_raw - smooth_min) / depth_range
            depth_map = np.clip(depth_map, 0, 1)
        
        # Invert: High raw disparity (close) -> Low output (0.0)
        depth_map = 1.0 - depth_map

        return depth_map

    def estimate_with_visualization(self, frame):
        """
        Estimate depth and return both depth map and visualization
        """
        depth_map = self.estimate(frame)
        
        # Visualization: Close=Bright, Far=Dark
        vis_raw = (1.0 - depth_map) * 255
        depth_vis = vis_raw.astype(np.uint8)
        
        # Apply MAGMA colormap
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        return depth_map, depth_colored
    
    def get_distance_at_point(self, depth_map, x, y, radius=5):
        """Get average depth around a point"""
        h, w = depth_map.shape
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(w, int(x + radius))
        y2 = min(h, int(y + radius))
        
        region = depth_map[y1:y2, x1:x2]
        return region.mean() if region.size > 0 else 0.5