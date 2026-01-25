import cv2
import logging
import config

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
        if config.USE_GSTREAMER:
            if self._init_gstreamer():
                self.is_gstreamer = True
                return
        
        if self._init_v4l2():
            return
            
        self._init_basic()

    def _init_gstreamer(self):
        pipeline = config.GSTREAMER_PIPELINE.format(
            sensor_id=self.camera_id,
            width=self.width,
            height=self.height,
            fps=self.fps,
            flip=config.CAMERA_FLIP_METHOD
        )

    
        logger.info(f"Pipeline: {pipeline}")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        # FIXED: Changed is_opened() to isOpened()
        return self.cap.isOpened()
    
    def get_info(self):
        if self.cap is None or not self.cap.isOpened():
            return {"backend": "None", "width": 0, "height": 0, "fps": 0}
        
        return {
            "backend": "GStreamer" if self.is_gstreamer else "V4L2",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS))
        }

    def _init_basic(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        return self.cap.isOpened()
    
    def _init_v4l2(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if self.cap.isOpened(): # FIXED
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        return False

    def _init_basic(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        return self.cap.isOpened() # FIXED

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()