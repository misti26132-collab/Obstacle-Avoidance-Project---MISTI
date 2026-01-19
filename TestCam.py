import cv2
import torch
from ultralytics import YOLO
import numpy as np

class ObstacleDetector:
    def __init__(self, model_path):
        self.device = "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def process(self, frame):
        h, w, _ = frame.shape
        results = self.model(frame, device=self.device, conf=0.4, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return frame, None, None

        boxes = results.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = np.argmax(areas)

        x1, y1, x2, y2 = boxes[idx]
        cx = (x1 + x2) / 2
        area = areas[idx]

        # Direction
        if cx < w / 3:
            direction = "left"
        elif cx > 2 * w / 3:
            direction = "right"
        else:
            direction = "center"

        # Distance proxy (demo)
        if area < 0.05 * (w * h):
            distance = "far"
        elif area < 0.15 * (w * h):
            distance = "close"
        else:
            distance = "very close"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{direction.upper()} | {distance.upper()}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        return frame, direction, distance
