import cv2
import torch
from ultralytics import YOLO

# ==============================
# Load YOLOv8 model safely
# ==============================

# Load the YOLOv8 nano model (lightweight & fast)
model = YOLO("yolov8n.pt")

# Force model to run on CPU (prevents FP16 / CUDA errors)
model.to("cpu")

# Disable FP16 (half precision) because CPU does not support it
model.model.fp16 = False


# ==============================
# Depth Estimator (Dummy Example)
# ==============================

class DepthEstimator:
    """
    This class demonstrates how depth *could* be estimated.
    NOTE: This is NOT real depth estimation.
    It only creates a fake depth map based on detection confidence.
    """

    def __init__(self):
        pass

    def estimate_depth(self, results):
        """
        Generate a dummy depth map using YOLOv8 detection results
        """
        # Get the original image from YOLO results
        img = results[0].orig_img
        height, width, _ = img.shape

        # Initialize depth map (all zeros)
        depth_map = torch.zeros((height, width))

        # Get bounding boxes from YOLOv8 output
        boxes = results[0].boxes

        # If no objects detected, return empty depth map
        if boxes is None:
            return depth_map.numpy()

        # Loop through each detected object
        for box, conf in zip(boxes.xyxy, boxes.conf):
            x1, y1, x2, y2 = map(int, box)

            # Dummy depth value (closer = higher confidence)
            depth_value = 1.0 / (conf.item() + 1e-6)

            # Fill depth map inside bounding box
            depth_map[y1:y2, x1:x2] = depth_value

        return depth_map.numpy()


# ==============================
# Open Webcam
# ==============================

# Open default webcam (0)
cap = cv2.VideoCapture(0)

# Create depth estimator object
depth_estimator = DepthEstimator()

# Check if webcam opened correctly
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Press 'Q' to quit")

# ==============================
# Main Loop (Real-Time Detection)
# ==============================

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame was not read correctly, exit
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 object detection on the frame
    results = model(frame, conf=0.5)

    # Draw bounding boxes and labels on frame
    annotated_frame = results[0].plot()

    # Generate dummy depth map (for demonstration)
    depth_map = depth_estimator.estimate_depth(results)

    # Normalize depth map for visualization
    depth_vis = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX
    ).astype("uint8")

    # Apply color map to depth image
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Show YOLO detection window
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Show depth estimation window
    cv2.imshow("Dummy Depth Map", depth_vis)

    # Exit when 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==============================
# Cleanup
# ==============================

cap.release()
cv2.destroyAllWindows()
