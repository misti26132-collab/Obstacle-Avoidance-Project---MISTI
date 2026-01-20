import cv2
import time
from TestCam import ObstacleDetector
from depth import DepthEstimator
from Speech import SpeechEngine

def main():
    # Camera setup with optimizations
    cap = cv2.VideoCapture(0)
    
    # Reduce resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # IMPORTANT: Clear buffer before each read
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Initialize components (share depth estimator)
    depth_estimator = DepthEstimator()
    detector = ObstacleDetector("yolov8n.pt", depth_estimator)
    speaker = SpeechEngine(cooldown=3.0)  # 3 seconds default cooldown
    
    print("Obstacle Avoidance System Started")
    print("Press 'q' to quit")
    
    # Warm-up: Skip first few frames (they're often corrupted)
    for _ in range(5):
        cap.read()
    
    fps_counter = 0
    fps_start = time.time()

    while True:
        # CRITICAL: Flush buffer to get latest frame
        cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Failed to read frame")
            break

        # Process frame
        frame, direction, distance = detector.process(frame)

        # Audio feedback (non-blocking)
        if direction is not None and distance is not None:
            speaker.speak(direction, distance)

        # Display FPS every 30 frames
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = time.time() - fps_start
            fps = 30 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f} | Latest: {direction} - {distance}")
            fps_start = time.time()

        cv2.imshow("Obstacle Avoidance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    print("System stopped")

if __name__ == "__main__":
    main()