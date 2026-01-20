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
    speaker = SpeechEngine(cooldown=2.5)
    
    print("Obstacle Avoidance System Started")
    print("Press 'q' to quit")
    
    # Warm-up: Skip first few frames (they're often corrupted)
    for _ in range(5):
        cap.read()
    
    last_print_time = time.time()
    print_interval = 0.5  # Print debug info every 0.5 seconds

    while True:
        # CRITICAL: Flush buffer to get latest frame
        cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Failed to read frame")
            break

        # Process frame
        frame, direction, distance = detector.process(frame)

        # Audio feedback with rate-limited debug output
        current_time = time.time()
        if direction is not None and distance is not None:
            if current_time - last_print_time >= print_interval:
                print(f"Detection: {direction} - {distance}")
                last_print_time = current_time
            speaker.speak(direction, distance)

        cv2.imshow("Obstacle Avoidance Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    print("System stopped")
    print("System stopped")

if __name__ == "__main__":
    main()