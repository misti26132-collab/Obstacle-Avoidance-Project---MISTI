import cv2
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

# ==========================================================
# TEXT-TO-SPEECH (WINDOWS SAFE – ZIRA VOICE)
# ==========================================================

speech_queue = queue.Queue()

def speech_worker():
    """
    Windows-safe TTS worker.
    Creates a NEW pyttsx3 engine for every message.
    Forces Microsoft David voice by exact SAPI ID.
    """
    DAVID_VOICE_ID = (
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\"
        "TTS_MS_EN-US_DAVID_11.0"
    )

    while True:
        text = speech_queue.get()

        # Shutdown signal
        if text is None:
            break

        try:
            print(f"[TTS] {text}")

            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("voice", DAVID_VOICE_ID)

            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

        except Exception as e:
            print("[TTS ERROR]", e)

        speech_queue.task_done()


def request_speech(text):
    """
    Clears old queued messages so speech does not stack.
    """
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break

    speech_queue.put(text)


# Start TTS thread
tts_thread = threading.Thread(target=speech_worker, daemon=True)
tts_thread.start()

# ==========================================================
# YOLOv8 SETUP
# ==========================================================

model = YOLO("yolov8n.pt")

# Force CPU + FP32 (avoids half-precision crashes on Windows)
model.to("cpu")
model.model.fp16 = False

# ==========================================================
# VIDEO CAPTURE
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not available.")
    exit()

# ==========================================================
# LOGIC PARAMETERS
# ==========================================================

COOLDOWN_TIME = 3.0      # Seconds between spoken messages
last_spoken_time = 0
last_direction = None

print("Obstacle avoidance running (Zira voice).")
print("Press 'Q' to quit.")

# ==========================================================
# MAIN LOOP
# ==========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, classes=[0, 15, 16, 24, 56], conf=0.5, verbose=False)
    boxes = results[0].boxes
    current_time = time.time()

    if boxes is not None and len(boxes) > 0:
        # Use the most confident detection
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

        # Speak if cooldown passed or direction changed
        if (
            current_time - last_spoken_time > COOLDOWN_TIME
            or direction != last_direction
        ):
            request_speech(message)
            last_spoken_time = current_time
            last_direction = direction

    else:
        # Reset when no obstacle is detected
        last_direction = None

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow("Obstacle Detection + TTS (DAVID)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================================
# CLEANUP
# ==========================================================

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()
