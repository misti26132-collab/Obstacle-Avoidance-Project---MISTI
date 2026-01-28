import cv2
import logging
import time

logger = logging.getLogger(__name__)

class JetsonCamera:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_gstreamer = False
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Try different camera initialization methods in order"""
        
        # Import config here to avoid circular imports
        try:
            import config as config
        except ImportError:
            try:
                import config
            except ImportError:
                logger.error("No config file found!")
                config = None
        
        if config and hasattr(config, 'USE_GSTREAMER') and config.USE_GSTREAMER:
            logger.info("[Camera] Trying GStreamer (CSI camera)...")
            if self._init_gstreamer(config):
                self.is_gstreamer = True
                logger.info("[Camera] ✅ GStreamer initialized")
                return
            logger.warning("[Camera] GStreamer failed, trying V4L2...")
        
        logger.info("[Camera] Trying V4L2...")
        if self._init_v4l2():
            logger.info("[Camera] ✅ V4L2 initialized")
            return

        logger.warning("[Camera] V4L2 failed, trying basic OpenCV...")
        if self._init_basic():
            logger.info("[Camera] ✅ Basic OpenCV initialized")
            return
        
        logger.error("[Camera] ❌ All camera initialization methods failed!")

    def _init_gstreamer(self, config):
        """Initialize GStreamer pipeline for CSI camera"""
        try:
            # Use the exact pipeline from config
            if hasattr(config, 'GSTREAMER_PIPELINE'):
                pipeline = config.GSTREAMER_PIPELINE
            else:
                # Fallback pipeline if not in config
                pipeline = (
                    f"nvarguscamerasrc sensor-id={self.camera_id} ! "
                    f"video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate={self.fps}/1 ! "
                    f"nvvidconv flip-method=0 ! "
                    f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
                    f"videoconvert ! "
                    f"video/x-raw, format=BGR ! "
                    f"appsink drop=1"
                )
            
            logger.info(f"[Camera] GStreamer pipeline: {pipeline[:80]}...")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"[Camera] GStreamer test read successful: {frame.shape}")
                    return True
                else:
                    logger.warning("[Camera] GStreamer opened but can't read frames")
                    self.cap.release()
                    return False
            return False
            
        except Exception as e:
            logger.error(f"[Camera] GStreamer error: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def _init_v4l2(self):
        try:
            logger.info("[Camera] Opening V4L2 device...")
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                logger.error("[Camera] V4L2 device not found")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info("[Camera] V4L2 device opened, warming up...")
            time.sleep(1)
            
            # Warmup frames - CSI camera needs more time
            for i in range(20):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.debug(f"[Camera] Warmup {i+1}/20: OK ({frame.shape})")
                    if i >= 10:
                        logger.info("[Camera] ✅ V4L2 initialized and ready")
                        return True
                else:
                    logger.debug(f"[Camera] Warmup {i+1}/20: Failed")
                    time.sleep(0.2)
            
            logger.warning("[Camera] V4L2 could not get valid frames")
            self.cap.release()
            return False
            
        except Exception as e:
            logger.error(f"[Camera] V4L2 error: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def _init_basic(self):
        """Initialize basic OpenCV capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if self.cap.isOpened():
                # Try to set resolution (may not work)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Skip first few frames to let camera stabilize
                for _ in range(5):
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("[Camera] Basic opened but can't read frames")
                        self.cap.release()
                        return False
                
                logger.info("[Camera] Basic OpenCV initialized and ready")
                return True
            return False
            
        except Exception as e:
            logger.error(f"[Camera] Basic error: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def get_info(self):
        """Get camera information"""
        if self.cap is None or not self.cap.isOpened():
            return {"backend": "None", "width": 0, "height": 0, "fps": 0}
        
        return {
            "backend": "GStreamer" if self.is_gstreamer else "V4L2/OpenCV",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS))
        }

    def read(self):
        """Read a frame from the camera"""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def isOpened(self):
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()

if __name__ == "__main__":
    print("=" * 60)
    print("CAMERA TEST")
    print("=" * 60)
    
    logging.basicConfig(level=logging.INFO)
    
    for cam_id in [0, 1]:
        print(f"\nTrying camera {cam_id}...")
        try:
            cam = JetsonCamera(camera_id=cam_id, width=640, height=480, fps=30)
            
            if cam.isOpened():
                info = cam.get_info()
                print(f"✅ Camera {cam_id} opened!")
                print(f"   Backend: {info['backend']}")
                print(f"   Resolution: {info['width']}x{info['height']}")
                print(f"   FPS: {info['fps']}")
                
                # Test frame capture
                ret, frame = cam.read()
                if ret:
                    print(f"   Frame shape: {frame.shape}")
                    print(f"   SUCCESS!")
                else:
                    print(f"   ❌ Can't read frames")
                
                cam.release()
                break
            else:
                print(f"❌ Camera {cam_id} failed to open")
                
        except Exception as e:
            print(f"❌ Error with camera {cam_id}: {e}")
    
    print("\n" + "=" * 60)