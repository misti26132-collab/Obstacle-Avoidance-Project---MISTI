import cv2
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

# 1. Setup the Speech Worker
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 175) # Speech rate
    while True:
        try:
            text = speech_queue.get()
            if text is None: 
                break
            
            print(f"Speaking: {text}") # Debug print to see it working
            engine.say(text)
            engine.runAndWait()
            
            speech_queue.task_done()
        except Exception as e:
            print(f"Speech Error: {e}")
            engine = pyttsx3.init()

# Start the speech thread
worker_thread = threading.Thread(target=speech_worker, daemon=True)
worker_thread.start()

def request_speech(text):
    speech_queue.put(text)

# 2. Setup YOLO and Video Capture
model = YOLO('yolov8n.pt')
url = "camera ip here" # url = 0  # Use 0 for local webcam or your IP URL to use with an external webcam
cap = cv2.VideoCapture(0) # use 0 or url

detection_frames = 0
COOLDOWN_TIME = 3 
last_command_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes
    current_time = time.time()

    if len(boxes) > 0:
        detection_frames += 1
        
        # Logic trigger: 5 frames of detection AND cooldown has passed
        if detection_frames >= 5 and (current_time - last_command_time > COOLDOWN_TIME):
            x_center = boxes[0].xywh[0][0].item()
            width = frame.shape[1]

            if x_center < (width / 3):
                msg = "Obstacle left, move right"
            elif x_center > (2 * width / 3):
                msg = "Obstacle right, move left"
            else:
                msg = "Obstacle center, stop"

            request_speech(msg)
            last_command_time = current_time
            detection_frames = 0
    else: 
        detection_frames = max(0, detection_frames - 1)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow("TTS", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

speech_queue.put(None) 
cap.release()
cv2.destroyAllWindows()