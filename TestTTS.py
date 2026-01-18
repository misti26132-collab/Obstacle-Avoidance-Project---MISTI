import cv2
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

# ==============================
# TEXT TO SPEECH (WINDOWS-SAFE)
# ==============================

speech_queue = queue.Queue()

def speech_worker():
    """
    WINDOWS-SAFE TTS WORKER
    Creates a NEW pyttsx3 engine for each message.
    This avoids the silent freeze bug.
    """
    while True:
        text = speech_queue.get()

        if text is None:
            break

        try:
            print(f"[TTS] {text}")

            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

            del engine  # force cleanup

        except Exception as e:
            print("[TTS ERROR]", e)

        speech_queue.task_done()


def request_speech(text):
    # Prevent speech backlog
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break

    speech_queue.put(text)


tts_thread = threading.Thread(target=speech_worker, daemon=True)
tts_thread.start()

# ==============================
# YOLO SETUP
# ==============================

model = YOLO("yolov8n.pt")
model.to("cpu")
model.model.fp16 = False

# ==============================
# VIDEO CAPTURE
# ==============================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not available")
    exit()

# ==============================
# LOGIC PARAMETERS
# ==============================

COOLDOWN_TIME = 3.0
last_spoken_time = 0
last_direction = None

print("System running. Press Q to quit.")

# ==============================
# MAIN LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes
    current_time = time.time()

    if boxes is not None and len(boxes) > 0:
        x_center = boxes.xywh[0][0].item()
        frame_width = frame.shape[1]

        if x_center < frame_width / 3:
            direction = "left"
            message = "Obstacle on the left. Move right."
        elif x_center > 2 * frame_width / 3:
            direction = "right"
            message = "Obstacle on the right. Move left."
        else:
            direction = "center"
            message = "Obstacle ahead. Stop."

        # Speak again if cooldown passed OR direction changed
        if (
            current_time - last_spoken_time > COOLDOWN_TIME
            or direction != last_direction
        ):
            request_speech(message)
            last_spoken_time = current_time
            last_direction = direction

    else:
        last_direction = None

    annotated_frame = results[0].plot()
    cv2.imshow("Obstacle Detection + TTS", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()
