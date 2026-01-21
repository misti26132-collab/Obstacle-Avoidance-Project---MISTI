import cv2
import time
import numpy as np
from TestCam import ObstacleDetector
from depth import DepthEstimator
from Speech import SpeechEngine

def main():
    print("=" * 60)
    print("OBSTACLE AVOIDANCE SYSTEM FOR BLIND ASSISTANCE")
    print("=" * 60)
    
    # Camera setup with optimizations
    print("\n[Camera] Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Please check connection.")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    print("[Camera] Camera initialized successfully")
    
    # Initialize depth estimator
    print("\n[Depth] Initializing depth estimator...")
    try:
        depth_estimator = DepthEstimator()
    except Exception as e:
        print(f"[ERROR] Failed to initialize depth estimator: {e}")
        cap.release()
        return
    
    # Initialize obstacle detector
    print("\n[Detector] Initializing obstacle detector...")
    try:
        # Try to load custom model first
        detector = ObstacleDetector("best.pt", depth_estimator, use_coco=True)
        print("[Detector] Running in DUAL MODEL mode (Custom + COCO)")
    except:
        # Fallback to COCO-only if custom model doesn't exist yet
        print("[Detector] Custom model not found, falling back to COCO-only mode")
        detector = ObstacleDetector("yolov8n.pt", depth_estimator, use_coco=False)

    print("\n[Speech] Initializing speech engine...")
    speaker = SpeechEngine(cooldown=3.0)
    
    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Toggle split view (camera + depth)")
    print("  'd' - Toggle debug info")
    print("=" * 60)
    print()
    
    # Warm-up: Skip first few frames (often corrupted/dark)
    print("[System] Warming up camera...")
    for _ in range(5):
        cap.read()
    
    # Display settings
    show_split_view = False
    show_debug = True
    
    # Performance monitoring
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    # Main loop
    try:
        while True:
            # Get latest frame
            cap.grab()
            ret, frame = cap.retrieve()
            
            if not ret:
                print("[WARNING] Failed to read frame")
                time.sleep(0.1)
                continue

            # Process frame
            try:
                annotated_frame, direction, distance, obstacle_class = detector.process(frame)
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
                annotated_frame = frame
                direction, distance, obstacle_class = None, None, None

            # Audio feedback (non-blocking)
            if direction is not None and distance is not None:
                speaker.speak(direction, distance, obstacle_class)

            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start
                fps_display = 30 / elapsed if elapsed > 0 else 0
                
                if show_debug:
                    status = f"{direction} - {distance}" if direction else "No obstacles"
                    print(f"[Status] FPS: {fps_display:.1f} | {status}")
                
                fps_start = time.time()

            # Create display frame
            if show_split_view:
                # Get depth visualization
                depth_map, depth_vis = depth_estimator.estimate_with_visualization(frame)
                
                # Resize for display
                display_height = 480
                display_width = 640
                frame_resized = cv2.resize(annotated_frame, (display_width, display_height))
                depth_resized = cv2.resize(depth_vis, (display_width, display_height))
                
                # Add labels
                cv2.putText(frame_resized, "CAMERA", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_resized, "DEPTH", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine horizontally
                display_frame = np.hstack([frame_resized, depth_resized])
            else:
                display_frame = annotated_frame

            # Add FPS overlay
            if show_debug:
                cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display frame
            cv2.imshow("Obstacle Avoidance System", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[System] Shutting down...")
                break
            elif key == ord('s'):
                show_split_view = not show_split_view
                mode = "Split View" if show_split_view else "Camera Only"
                print(f"[Display] Switched to {mode}")
            elif key == ord('d'):
                show_debug = not show_debug
                status = "enabled" if show_debug else "disabled"
                print(f"[Display] Debug info {status}")

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n[System] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        speaker.stop()
        print("[System] Shutdown complete")
        print("=" * 60)

if __name__ == "__main__":
    main()