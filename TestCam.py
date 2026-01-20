import cv2
import torch
import numpy as np
from ultralytics import YOLO
from depth import DepthEstimator

class ObstacleDetector:
    def __init__(self, model_path):
        self.device = "cpu"

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.depth_estimator = DepthEstimator()

    def process(self, frame):
        h, w, _ = frame.shape

        depth_map = self.depth_estimator.estimate(frame)

        results = self.model(frame, device=self.device, conf=0.4, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return frame, None, None

        boxes = results.boxes.xyxy.cpu().numpy()

        closest_depth = float("inf")
        closest_box = None

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_depth = depth_map[y1:y2, x1:x2]

            if box_depth.size == 0:
                continue

            mean_depth = box_depth.mean()

            if mean_depth < closest_depth:
                closest_depth = mean_depth
                closest_box = (x1, y1, x2, y2)

        if closest_box is None:
            return frame, None, None

        x1, y1, x2, y2 = closest_box
        cx = (x1 + x2) / 2

        if cx < w / 3:
            direction = "left"
        elif cx > 2 * w / 3:
            direction = "right"
        else:
            direction = "center"

        if closest_depth > 0.65:
            distance = "far"
        elif closest_depth > 0.4:
            distance = "close"
        else:
            distance = "very close"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{direction.upper()} | {distance.upper()}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        return frame, direction, distance
