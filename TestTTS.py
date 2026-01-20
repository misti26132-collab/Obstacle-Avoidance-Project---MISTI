import cv2
import time
from Speech import SpeechEngine

cap = cv2.VideoCapture(0)
speaker = SpeechEngine(cooldown=2.0)

print("TTS Test - Testing all distance/direction combinations")
print("Press 'q' to quit")

test_sequence = [
    ("center", "very_close"),
    ("left", "very_close"),
    ("right", "very_close"),
    ("center", "close"),
    ("left", "close"),
    ("right", "close"),
    ("center", "far"),
]

test_idx = 0
last_test_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Cycle through test messages every 3 seconds
    current_time = time.time()
    if current_time - last_test_time >= 3.0:
        direction, distance = test_sequence[test_idx]
        print(f"\nTesting: {direction} - {distance}")
        speaker.speak(direction, distance)
        
        test_idx = (test_idx + 1) % len(test_sequence)
        last_test_time = current_time

    cv2.imshow("TTS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
speaker.stop()
print("TTS test completed")