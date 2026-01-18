import cv2
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

# 1. Setup the Speech Worker (The "Voice" Thread)
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()
        time.sleep(0.5) # Short pause between sentences

# Start the speech thread immediately
worker_thread = threading.Thread(target=speech_worker, daemon=True)
worker_thread.start()

def request_speech(text):
    if speech_queue.empty():
        speech_queue.put(text)

# 2. Setup YOLO and Camera
model = YOLO('yolov8n.pt')
url = "camera ip here"  # Replace with your IP camera URL
cap = cv2.VideoCapture(url)

# Variables to slow down the logic triggers
detection_frames = 0
COOLDOWN_TIME = 2  # Seconds to wait between commands
last_command_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes
    
    current_time = time.time()

    if len(boxes) > 0:
        detection_frames += 1
        
        # Only process logic if we've seen the object for 5+ frames
        if detection_frames > 5 and (current_time - last_command_time > COOLDOWN_TIME):
            # Get position of the first detected object
            x_center = boxes[0].xywh[0][0].item()
            width = frame.shape[1]

            # Logic for direction
            if x_center < (width / 3):
                msg = "Obstacle left, move right"
            elif x_center > (2 * width / 3):
                msg = "Obstacle right, move left"
            else:
                msg = "Obstacle center, stop"

            request_speech(msg)
            last_command_time = current_time
            detection_frames = 0 # Reset after triggering
    else:
        detection_frames = 0

    # Show the results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Navigation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
speech_queue.put(None) # Tell the speech thread to stop
cap.release()
cv2.destroyAllWindows()