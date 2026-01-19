import cv2
from TestCam import ObstacleDetector
from Speech import SpeechEngine

def main():
    cap = cv2.VideoCapture(0)

    detector = ObstacleDetector("yolov8n.pt")
    speaker = SpeechEngine(cooldown=2.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, direction, distance = detector.process(frame)

        if direction is not None:
            speaker.speak(direction, distance)

        cv2.imshow("Obstacle Avoidance Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
