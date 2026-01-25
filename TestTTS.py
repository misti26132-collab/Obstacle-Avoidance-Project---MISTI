import cv2
import time
from Speech import SpeechEngine

cap = cv2.VideoCapture('/dev/video0')

if not cap.isOpened():
    print("ERROR: Camera not available.")
    exit()

# FIXED: Use config for speech settings
speaker = SpeechEngine(cooldown=2.0)

print("=" * 60)
print("TTS Test - Testing all distance/direction combinations")
print("=" * 60)
print("Press 'q' to quit")
print()

test_sequence = [
    ("center", "very_close", "person"),
    ("left", "very_close", "chair"),
    ("right", "very_close", "car"),
    ("center", "close", "person"),
    ("left", "close", "bench"),
    ("right", "close", "bicycle"),
    ("center", "far", "person"),
]

test_idx = 0
last_test_time = time.time()

print("Starting test sequence...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Cycle through test messages every 3 seconds
        current_time = time.time()
        if current_time - last_test_time >= 3.0:
            direction, distance, obstacle = test_sequence[test_idx]
            print(f"\n[Test {test_idx + 1}/{len(test_sequence)}] Testing: {direction} - {distance} - {obstacle}")
            speaker.speak(direction, distance, obstacle)
            
            test_idx = (test_idx + 1) % len(test_sequence)
            last_test_time = current_time

        # Display the camera feed
        cv2.putText(frame, f"Test {test_idx + 1}/{len(test_sequence)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("TTS Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[System] Interrupted by user")

finally:
    print("\n" + "=" * 60)
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    print("TTS test completed")
    print("=" * 60)