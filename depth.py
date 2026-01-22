import torch
import cv2
import numpy as np
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthEstimator:
    def __init__(self, device=None, max_retries=3):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"[DepthEstimator] Using device: {self.device}")

        self.midas = None
        self.transform = None
        
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"[DepthEstimator] Loading MiDaS model "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                self.midas = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                self.midas.to(self.device)
                self.midas.eval()
                
                logger.info("[DepthEstimator] Loading transforms...")
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.transform = transforms.small_transform
                
                logger.info("[DepthEstimator] MiDaS model loaded successfully")
                
                # FIXED: Verify model loaded correctly
                if self.midas is None or self.transform is None:
                    raise RuntimeError("MiDaS model or transforms failed to load")
                
                # FIXED: Test model with dummy input
                try:
                    dummy = torch.randn(1, 3, 256, 256).to(self.device)
                    with torch.no_grad():
                        _ = self.midas(dummy)
                    logger.info("[DepthEstimator] Model verification successful")
                except Exception as e:
                    raise RuntimeError(f"Model loaded but failed verification: {e}")
                
                break
                
            except Exception as e:
                logger.error(
                    f"[DepthEstimator] Attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"[DepthEstimator] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to load MiDaS model after {max_retries} attempts.\n"
                        f"Last error: {e}\n"
                        f"This is usually a network issue. "
                        f"Check your internet connection and try again."
                    )
        
        # Adaptive scaling components to prevent depth inconsistencies
        self.scale_history = deque(maxlen=30)  # Track last 30 frames
        self.min_history = deque(maxlen=30)
        self.initialized = False
        
        logger.info("[DepthEstimator] Initialization complete")

    def estimate(self, frame):
        """
        Estimate depth with ADAPTIVE scaling to prevent inconsistencies
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            depth_map: Normalized depth map where:
                      - Higher values (closer to 1.0) = CLOSER objects
                      - Lower values (closer to 0.0) = FARTHER objects
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided to depth estimator")

        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        input_batch = self.transform(img).to(self.device)

        # Run inference
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
            logger.warning("[DepthEstimator] Uniform depth detected, returning zeros")
            depth_map = np.zeros_like(depth_map_raw)
        else:
            depth_map = (depth_map_raw - smooth_min) / depth_range
            depth_map = np.clip(depth_map, 0, 1)
        
        # Invert so that: 1.0 = CLOSE (high disparity), 0.0 = FAR (low disparity)
        depth_map = 1.0 - depth_map

        return depth_map

    def estimate_with_visualization(self, frame):
        depth_map = self.estimate(frame)
        
        # Visualization: Close=Bright, Far=Dark
        vis_raw = depth_map * 255  # High values (close) = bright
        depth_vis = vis_raw.astype(np.uint8)
        
        # Apply MAGMA colormap for better visibility
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        return depth_map, depth_colored
    
    def get_distance_at_point(self, depth_map, x, y, radius=5):
        h, w = depth_map.shape
        x = int(x)
        y = int(y)
        
        # Calculate region bounds
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        
        # Extract region and calculate mean
        region = depth_map[y1:y2, x1:x2]
        return float(region.mean()) if region.size > 0 else 0.5
    
    def visualize_depth(self, depth_map_normalized):
        depth_vis = (depth_map_normalized * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return colored_depth


def test_depth_estimator():
    print("=" * 60)
    print("DEPTH ESTIMATOR TEST")
    print("=" * 60)
    
    # Initialize
    try:
        estimator = DepthEstimator()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    print("\n[Camera] Ready")
    print("Controls: 'q' to quit")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Estimate depth
            depth_map, depth_vis = estimator.estimate_with_visualization(frame)
            
            # Display side by side
            display = np.hstack([frame, depth_vis])
            cv2.imshow("Camera | Depth", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[System] Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    test_depth_estimator()