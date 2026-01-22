import os
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import tempfile
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoboflowDetector:
    def __init__(self, api_key=None, workspace_name=None, workflow_id=None):
        """
        Initialize Roboflow detector with workflow API
        
        Args:
            api_key: Roboflow API key (optional, reads from env)
            workspace_name: Roboflow workspace (optional, reads from env)
            workflow_id: Roboflow workflow ID (optional, reads from env)
        """
        # Get configuration from environment variables with fallbacks
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = workspace_name or os.getenv("ROBOFLOW_WORKSPACE", "hamzeh-alqaqa")
        self.workflow_id = workflow_id or os.getenv("ROBOFLOW_WORKFLOW", "find-pillars")
        
        if not self.api_key:
            raise ValueError(
                "Roboflow API key not found. Either:\n"
                "1. Set ROBOFLOW_API_KEY environment variable in .env file, or\n"
                "2. Pass api_key parameter to RoboflowDetector()\n"
                "\nGet your API key from: https://app.roboflow.com/settings/api"
            )
        
        # Initialize client
        try:
            self.client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=self.api_key
            )
            logger.info(f"[Roboflow] Initialized with workflow: {self.workflow_id}")
            logger.info(f"[Roboflow] Workspace: {self.workspace_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Roboflow client: {e}")
    
    def detect_from_frame(self, frame, use_cache=True):
        """
        Run detection on a cv2 frame (numpy array) from webcam
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            use_cache: Whether to cache workflow definition (recommended for real-time)
            
        Returns:
            Detection results from Roboflow workflow or None if error
        """
        if frame is None or frame.size == 0:
            logger.error("[Roboflow] Invalid frame provided")
            return None
        
        temp_path = None
        try:
            # Save frame to temporary file (Roboflow API requires file path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                success = cv2.imwrite(temp_path, frame)
                
                # FIXED: Verify write was successful
                if not success:
                    logger.error("[Roboflow] Failed to write temporary image")
                    return None
            
            # FIXED: Verify file exists and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error("[Roboflow] Temporary file is invalid")
                return None
            
            # Run detection using workflow
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": temp_path},
                use_cache=use_cache
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[Roboflow] Detection error: {type(e).__name__}: {e}")
            return None
        
        finally:
            # FIXED: Clean up temp file safely
            if temp_path:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"[Roboflow] Failed to delete temp file: {e}")
    
    def parse_detections(self, result):
        """
        Parse Roboflow workflow results into standard format
        
        Args:
            result: Raw Roboflow API response
            
        Returns:
            List of detections with format:
            [{'class': 'person', 'confidence': 0.95, 'box': [x1, y1, x2, y2]}, ...]
        """
        if not result:
            return []
        
        detections = []
        
        try:
            # Roboflow workflow outputs are nested - need to extract predictions
            predictions = []
            
            # Handle list result
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            # Extract predictions from various possible structures
            if isinstance(result, dict):
                # Option 1: Direct predictions key
                if 'predictions' in result:
                    predictions = result['predictions']
                
                # Option 2: Nested in output
                elif 'output' in result:
                    output = result['output']
                    if isinstance(output, dict):
                        if 'predictions' in output:
                            predictions = output['predictions']
                        else:
                            # Look for any key that contains predictions
                            for key, value in output.items():
                                if isinstance(value, dict) and 'predictions' in value:
                                    predictions = value['predictions']
                                    break
                                elif isinstance(value, list):
                                    predictions = value
                                    break
                
                # Option 3: Search all keys for predictions-like structure
                if not predictions:
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            # Check if it looks like predictions
                            if isinstance(value[0], dict) and any(
                                k in value[0] for k in ['class', 'x', 'y', 'width', 'height']
                            ):
                                predictions = value
                                break
            
            # Parse predictions into standard format
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                
                # Get class name
                class_name = pred.get('class', pred.get('class_name', 'unknown'))
                
                # Get confidence
                confidence = pred.get('confidence', pred.get('conf', 0.0))
                
                # Get bounding box coordinates (Roboflow format: center x, y, width, height)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                
                # Convert to x1, y1, x2, y2 format
                x1 = x - width / 2
                y1 = y - height / 2
                x2 = x + width / 2
                y2 = y + height / 2
                
                detection = {
                    'class': class_name,
                    'confidence': float(confidence),
                    'box': [x1, y1, x2, y2]
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"[Roboflow] Error parsing detections: {e}")
            logger.debug(f"[Roboflow] Raw result structure: {type(result)}")
            if isinstance(result, dict):
                logger.debug(f"[Roboflow] Result keys: {list(result.keys())}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: OpenCV frame
            detections: List of detections from parse_detections()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            class_name = det['class']
            confidence = det['confidence']
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return annotated


def main():
    """
    Test the Roboflow detector with webcam
    """
    print("=" * 60)
    print("ROBOFLOW OBSTACLE DETECTION - WEBCAM TEST")
    print("=" * 60)
    
    # Initialize detector
    try:
        detector = RoboflowDetector()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo set environment variable:")
        print("  Create a .env file with:")
        print("  ROBOFLOW_API_KEY='your_key_here'")
        return
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        return
    
    # Initialize webcam
    print("\n[Camera] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[Camera] Webcam ready")
    print("\nControls:")
    print("  'q' - Quit")
    print("  SPACE - Run detection on current frame")
    print("  'c' - Toggle continuous detection")
    print("=" * 60)
    
    continuous_mode = False
    last_result = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame")
                continue
            
            display_frame = frame.copy()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') or continuous_mode:
                # Run detection
                print("\n[Detecting...]")
                result = detector.detect_from_frame(frame)
                
                if result:
                    detections = detector.parse_detections(result)
                    print(f"[Found] {len(detections)} obstacles")
                    
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f}")
                    
                    display_frame = detector.draw_detections(frame, detections)
                    last_result = detections
                else:
                    print("[Result] No detections or API error")
            
            elif key == ord('c'):
                continuous_mode = not continuous_mode
                mode = "ON" if continuous_mode else "OFF"
                print(f"\n[Mode] Continuous detection: {mode}")
            
            # Draw last detections if available
            if last_result and not continuous_mode:
                display_frame = detector.draw_detections(frame, last_result)
            
            # Add status text
            status = "CONTINUOUS" if continuous_mode else "MANUAL (Press SPACE)"
            cv2.putText(
                display_frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("Roboflow Detection", display_frame)
    
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user")
    
    finally:
        print("\n[Cleanup] Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("[Done] Test complete")
        print("=" * 60)


if __name__ == "__main__":
    main()