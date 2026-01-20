import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, device=None):
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
            
            print(" MiDaS model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MiDaS model: {e}")

    def estimate_depth(self, frame):
        # Convert BGR → RGB
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
    
        # Normalize to 0-1 range
        depth_min = depth_map_raw.min()
        depth_max = depth_map_raw.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_map_raw - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map_raw)
        
        # depth_normalized: 0 = far, 1 = close 
        return depth_normalized, depth_map_raw

    def estimate(self, frame):
        depth_normalized, _ = self.estimate_depth(frame)
        return depth_normalized

    def estimate_with_visualization(self, frame):
        depth_map, _ = self.estimate_depth(frame)
        
        # Convert to 0-255 range for visualization
        depth_vis = (depth_map * 255).astype(np.uint8)
        
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        return depth_map, depth_colored
    
    def get_object_depth(self, depth_map, x1, y1, x2, y2):
    
        h, w = depth_map.shape
        
        # Ensure bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        
        # Extract region
        roi = depth_map[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0, "unknown"
        
        # Get maximum depth in ROI (closest point)
        max_depth = np.max(roi)
        mean_depth = np.mean(roi)
        
        # Use max_depth for proximity (most conservative)
        # Higher value = closer
        if max_depth > 0.75:  # Top 25% = very close
            depth_category = "very_close"
        elif max_depth > 0.55:  # Top 45% = close
            depth_category = "close"
        elif max_depth > 0.35:  # Top 65% = medium
            depth_category = "medium"
        else:
            depth_category = "far"
        
        return max_depth, depth_category
    
    def get_distance_at_point(self, depth_map, x, y, radius=5):
        h, w = depth_map.shape
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(w, int(x + radius))
        y2 = min(h, int(y + radius))
        
        region = depth_map[y1:y2, x1:x2]
        return region.mean() if region.size > 0 else 0.5
    
    def visualize_depth(self, depth_map_normalized):

        depth_vis = (depth_map_normalized * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return colored_depth