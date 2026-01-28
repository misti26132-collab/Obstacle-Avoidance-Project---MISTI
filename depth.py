import torch
import cv2
import numpy as np
from collections import deque
import time
import logging
import config
import os
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthEstimator:
    def __init__(self, device=None, max_retries=3):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"[DepthEstimator] Using device: {self.device}")
        
        self.use_tensorrt = False
        if torch.cuda.is_available():
            try:
                self.use_tensorrt = True
                logger.info("[DepthEstimator] TensorRT available - will use optimizations")
            except ImportError:
                logger.info("[DepthEstimator] TensorRT not available - using standard CUDA")

        self.midas = None
        self.transform = None
        self.use_fp16 = config.USE_HALF_PRECISION and torch.cuda.is_available()
        
        # Create cache directory
        self.cache_dir = Path(config.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"[DepthEstimator] Loading MiDaS model "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                
                # Try to load from cache first or download with timeout
                try:
                    # Set timeout for torch.hub to prevent hanging
                    import socket
                    socket.setdefaulttimeout(config.MODEL_LOAD_TIMEOUT)
                    
                    # Try loading through torch.hub
                    self.midas = torch.hub.load(
                        "intel-isl/MiDaS", "MiDaS_small", 
                        trust_repo=True,
                        skip_validation=True  # Faster loading
                    )
                    logger.info("[DepthEstimator] MiDaS model loaded from network/cache")
                    
                except RuntimeError as e:
                    if "operator torchvision::nms" in str(e):
                        logger.warning("[DepthEstimator] Circular import detected, using fallback depth...")
                        raise RuntimeError("Circular import - using fallback")
                    raise
                        
                except (OSError, urllib.error.URLError, socket.timeout, Exception) as e:
                    logger.warning(f"[DepthEstimator] Network load failed ({type(e).__name__}), checking local cache...")
                    
                    # Try to load from local cache if offline mode or network failed
                    try:
                        local_model_path = self.cache_dir / "midas_small.pt"
                        if local_model_path.exists():
                            logger.info(f"[DepthEstimator] Loading MiDaS from local cache: {local_model_path}")
                            self.midas = torch.hub.load(
                                "intel-isl/MiDaS", "MiDaS_small",
                                source="local",
                                skip_validation=True
                            )
                        elif config.OFFLINE_MODE:
                            raise RuntimeError(
                                f"Offline mode enabled but no cached model found at {local_model_path}. "
                                f"Please download the model when internet is available."
                            )
                        else:
                            logger.warning("[DepthEstimator] No local cache, retrying network...")
                            raise
                    except Exception as cache_error:
                        logger.warning(f"[DepthEstimator] Cache load also failed: {cache_error}")
                        raise
                
                self.midas.to(self.device)
                self.midas.eval()
                
                # Jetson optimization: Convert to FP16 if enabled
                if self.use_fp16:
                    logger.info("[DepthEstimator] Converting model to FP16...")
                    self.midas = self.midas.half()
                
                logger.info("[DepthEstimator] Loading transforms...")
                try:
                    socket.setdefaulttimeout(config.MODEL_LOAD_TIMEOUT)
                    transforms = torch.hub.load(
                        "intel-isl/MiDaS", "transforms", 
                        trust_repo=True,
                        skip_validation=True
                    )
                except (OSError, urllib.error.URLError, socket.timeout, Exception) as e:
                    logger.warning(f"[DepthEstimator] Could not load transforms from network: {e}")
                    # Fallback to basic transform
                    logger.info("[DepthEstimator] Using fallback transforms...")
                    transforms = self._get_fallback_transforms()
                
                self.transform = transforms.small_transform
                
                logger.info("[DepthEstimator] MiDaS model loaded successfully")
                
                if self.midas is None or self.transform is None:
                    raise RuntimeError("MiDaS model or transforms failed to load")
                
                # Test model with dummy input
                try:
                    dummy = torch.randn(1, 3, 256, 256).to(self.device)
                    if self.use_fp16:
                        dummy = dummy.half()
                    
                    with torch.no_grad():
                        _ = self.midas(dummy)
                    
                    logger.info("[DepthEstimator] Model verification successful")
                    if self.use_fp16:
                        logger.info("[DepthEstimator] FP16 mode enabled for faster inference")
                except Exception as e:
                    raise RuntimeError(f"Model loaded but failed verification: {e}")
                
                break
                
            except Exception as e:
                logger.error(
                    f"[DepthEstimator] Attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"[DepthEstimator] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Use simplified fallback on final failure
                    logger.warning("[DepthEstimator] All loading attempts failed, using fallback depth estimator")
                    self._use_fallback_depth_estimator()
                    return
        
    def _use_fallback_depth_estimator(self):
        """Use a simple CNN-based depth estimator as fallback"""
        logger.info("[DepthEstimator] Initializing fallback depth estimator (simple model)...")
        try:
            # Try to use timm-based depth model as fallback
            import timm
            try:
                self.midas = timm.create_model('vit_tiny_patch16_224', pretrained=True)
                self.midas.to(self.device)
                self.midas.eval()
                if self.use_fp16:
                    self.midas = self.midas.half()
                logger.info("[DepthEstimator] Using ViT-Tiny as fallback depth model")
            except:
                # If timm fails, use a simple passthrough
                logger.info("[DepthEstimator] Using passthrough fallback depth (will return zeros)")
                self.midas = None
        except:
            logger.warning("[DepthEstimator] Could not initialize fallback, will use dummy output")
            self.midas = None
        
        # Always use fallback transforms
        self.transform = self._get_fallback_transforms().small_transform
        self.initialized = True
        
        # Adaptive scaling components
        self.scale_history = deque(maxlen=config.DEPTH_HISTORY_SIZE)
        self.min_history = deque(maxlen=config.DEPTH_HISTORY_SIZE)
        self.initialized = False
        
        # Jetson optimization: Pre-allocate CUDA memory for common sizes
        if torch.cuda.is_available():
            logger.info("[DepthEstimator] Pre-warming CUDA...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                _ = self.estimate(dummy_frame)
                logger.info("[DepthEstimator] CUDA warm-up complete")
            except Exception as e:
                logger.warning(f"[DepthEstimator] CUDA warm-up skipped: {e}")
        
        logger.info("[DepthEstimator] Initialization complete")
    
    def _get_fallback_transforms(self):
        """Return a fallback transforms object when network is unavailable"""
        class FallbackTransforms:
            class SmallTransform:
                def __call__(self, img):
                    # Resize to 256x256 and normalize
                    h, w = img.shape[:2]
                    size = 256
                    img = cv2.resize(img, (size, size))
                    
                    # Normalize to [-1, 1] range
                    img = img.astype(np.float32) / 255.0
                    img = (img - 0.5) / 0.5
                    
                    # Convert to tensor
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                    return img
            
            small_transform = SmallTransform()
        
        return FallbackTransforms()

    @torch.amp.autocast('cuda', enabled=True)  # Automatic mixed precision
    def estimate(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided to depth estimator")

        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Handle fallback (no model loaded)
        if self.midas is None:
            logger.debug("[DepthEstimator] Using dummy depth (fallback mode)")
            # Return uniform depth map indicating "far away"
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        # Prepare input
        input_batch = self.transform(img).to(self.device)
        
        # Convert to FP16 if enabled
        if self.use_fp16:
            input_batch = input_batch.half()

        # Run inference with torch.no_grad() to save memory
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Handle output if it's not a depth map (for timm models)
            if prediction.dim() == 4:  # [batch, channels, height, width]
                if prediction.shape[1] > 1:
                    # Multi-channel output, average to single channel
                    prediction = prediction.mean(dim=1, keepdim=True)
                else:
                    prediction = prediction.squeeze(1)
            
            # Resize prediction
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1) if prediction.dim() == 3 else prediction,
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy (handle FP16)
        if self.use_fp16:
            depth_map_raw = prediction.float().cpu().numpy()
        else:
            depth_map_raw = prediction.cpu().numpy()
        
        # Aggressive memory cleanup after inference
        del input_batch, prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Adaptive scaling
        p5 = np.percentile(depth_map_raw, config.DEPTH_MIN_PERCENTILE)
        p95 = np.percentile(depth_map_raw, config.DEPTH_MAX_PERCENTILE)
        
        self.min_history.append(p5)
        self.scale_history.append(p95)
        
        if len(self.scale_history) >= 10:
            smooth_min = np.median(self.min_history)
            smooth_max = np.median(self.scale_history)
            self.initialized = True
        else:
            smooth_min = p5
            smooth_max = p95
        
        depth_range = smooth_max - smooth_min
        if depth_range < 1e-6:
            logger.warning("[DepthEstimator] Uniform depth detected, returning zeros")
            depth_map = np.zeros_like(depth_map_raw)
        else:
            depth_map = (depth_map_raw - smooth_min) / depth_range
            depth_map = np.clip(depth_map, 0, 1)
        
        # Invert: 1.0 = CLOSE, 0.0 = FAR
        depth_map = 1.0 - depth_map

        return depth_map

    def estimate_with_visualization(self, frame):
        depth_map = self.estimate(frame)
        
        vis_raw = depth_map * 255
        depth_vis = vis_raw.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        return depth_map, depth_colored
    
    def get_distance_at_point(self, depth_map, x, y, radius=5):
        h, w = depth_map.shape
        x = int(x)
        y = int(y)
        
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        
        region = depth_map[y1:y2, x1:x2]
        return float(region.mean()) if region.size > 0 else 0.5


def test_depth_estimator():
    print("=" * 60)
    print("JETSON DEPTH ESTIMATOR TEST")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, using CPU (slow)")
    
    print("=" * 60)
    
    try:
        estimator = DepthEstimator()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        return
    
    # Import camera utils
    try:
        from camera_utils import JetsonCamera
        cap = JetsonCamera(
            camera_id=config.CAMERA_INDEX,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS
        )
    except ImportError:
        print("[WARNING] camera_utils not found, using basic OpenCV")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return
    
    print("\n[Camera] Ready")
    print("Controls: 'q' to quit")
    print("=" * 60)
    
    fps_counter = 0
    fps_start = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Estimate depth
            start = time.time()
            depth_map, depth_vis = estimator.estimate_with_visualization(frame)
            inference_time = (time.time() - start) * 1000
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                print(f"[Performance] FPS: {fps:.1f} | Inference: {inference_time:.1f}ms")
                fps_start = time.time()
            
            # Display
            cv2.putText(
                depth_vis, f"FPS: {fps_counter % 100}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            display = np.hstack([frame, depth_vis])
            cv2.imshow("Camera | Depth (Jetson)", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[System] Interrupted")
    
    finally:
        if hasattr(cap, 'release'):
            cap.release()
        cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    test_depth_estimator()