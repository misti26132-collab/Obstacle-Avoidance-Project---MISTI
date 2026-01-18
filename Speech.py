import cv2
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

# 1. Setup the Speech Worker
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None: 
            break
        try:
            print(f"Executing Speech: {text}")
            engine = pyttsx3.init()
            
            engine.setProperty('rate', 180)
            
            engine.say(text)
            engine.runAndWait()
            
            engine.stop()
            del engine 
            
        except Exception as e:
            print(f"Speech Thread Error: {e}")
        
        finally:
            speech_queue.task_done()

worker_thread = threading.Thread(target=speech_worker, daemon=True)
worker_thread.start()

def request_speech(text):
    if speech_queue.qsize() < 2:
        speech_queue.put(text)

# Initialize YOLO and Video
model = YOLO('yolov8n.pt')
url = "camera ip here" # Replace with your IP camera
cap = cv2.VideoCapture(url) # URL or 0 for webcam

detection_frames = 0
COOLDOWN_TIME = 3.0  # Seconds between speech triggers
last_command_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes
    current_time = time.time()

    if len(boxes) > 0:
        detection_frames += 1
        
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
        detection_frames = 0

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Navigation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()