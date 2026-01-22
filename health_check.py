import logging
import time

logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    
    def __init__(self):
        self.last_depth_update = time.time()
        self.last_detection_update = time.time()
        self.last_camera_frame = time.time()
        
        # Timeout thresholds (seconds)
        self.camera_timeout = 2.0
        self.depth_timeout = 5.0
        self.detection_timeout = 10.0
        
        logger.info("[HealthMonitor] Initialized")
    
    def update_depth(self):
        """Mark depth estimator as active"""
        self.last_depth_update = time.time()
    
    def update_detection(self):
        """Mark detection system as active"""
        self.last_detection_update = time.time()
    
    def update_camera(self):
        """Mark camera as active"""
        self.last_camera_frame = time.time()
    
    def check_health(self):
        """
        Check all subsystem health
        
        Returns:
            bool: True if all systems healthy, False otherwise
        """
        now = time.time()
        issues = []
        
        if now - self.last_camera_frame > self.camera_timeout:
            issues.append("Camera frozen")
        
        if now - self.last_depth_update > self.depth_timeout:
            issues.append("Depth estimator frozen")
        
        if now - self.last_detection_update > self.detection_timeout:
            issues.append("Detection system frozen")
        
        if issues:
            logger.warning(f"[Health] System issues detected: {', '.join(issues)}")
            return False
        
        return True
    
    def get_status(self):
        """
        Get detailed status of all components
        
        Returns:
            dict: Status of each component
        """
        now = time.time()
        return {
            'camera': {
                'healthy': (now - self.last_camera_frame) < self.camera_timeout,
                'last_update': self.last_camera_frame,
                'age': now - self.last_camera_frame
            },
            'depth': {
                'healthy': (now - self.last_depth_update) < self.depth_timeout,
                'last_update': self.last_depth_update,
                'age': now - self.last_depth_update
            },
            'detection': {
                'healthy': (now - self.last_detection_update) < self.detection_timeout,
                'last_update': self.last_detection_update,
                'age': now - self.last_detection_update
            }
        }