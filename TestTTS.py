import cv2
from Speech import SpeechEngine

cap = cv2.VideoCapture(0)
speaker = SpeechEngine(cooldown=2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    speaker.speak("center", "close")
    cv2.imshow("TTS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
