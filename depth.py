import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, device=None):
        """
        Initialize MiDaS depth estimator
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

    def estimate(self, frame):
        """
        Estimate depth from BGR frame
        
        Args:
            frame: OpenCV BGR image (numpy array)
            
        Returns:
            depth_map: Normalized depth map where:
                      - Higher values (closer to 1.0) = CLOSER objects
                      - Lower values (closer to 0.0) = FARTHER objects
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
        # FIXED SCALING (FLIPPED)
        # ==========================================================
        DISPARITY_SCALE = 800.0
        
        # Normalize raw disparity
        depth_map = depth_map_raw / DISPARITY_SCALE
        
        # Clip to [0, 1]
        depth_map = np.clip(depth_map, 0, 1)
        
        # PREVIOUSLY: depth_map = 1.0 - depth_map
        # NOW: We keep it as is.
        # RESULT: 
        #   High Raw Disparity (Close) -> High Output (1.0)
        #   Low Raw Disparity (Far)    -> Low Output (0.0)

        return depth_map

    def estimate_with_visualization(self, frame):
        """
        Estimate depth and return both depth map and visualization
        """
        depth_map = self.estimate(frame)
        
        # For Visualization:
        # We still want CLOSE objects to look BRIGHT/YELLOW.
        # Since Close is now ~1.0:
        # We just multiply by 255 (No inversion needed here either)
        
        vis_raw = depth_map * 255
        depth_vis = vis_raw.astype(np.uint8)
        
        # MAGMA: 0=Black (Far), 255=Yellow (Close)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        return depth_map, depth_colored
    
    def get_distance_at_point(self, depth_map, x, y, radius=5):
        h, w = depth_map.shape
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(w, int(x + radius))
        y2 = min(h, int(y + radius))
        
        region = depth_map[y1:y2, x1:x2]
        return region.mean() if region.size > 0 else 0.5