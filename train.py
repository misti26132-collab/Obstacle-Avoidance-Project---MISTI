
import numpy as np
from ultralytics import YOLO
import torch
import os

DATA_YAML = 'v2_dataset/data.yaml'
EPOCHS = 50
BATCH_SIZE = 8          
IMAGE_SIZE = 640
PROJECT_NAME = "blind_navigation"
RUN_NAME = "obstacles_v1"

def main():
    print("=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (training will be slower)")

    print("=" * 60)

    # Verify dataset
    if not os.path.exists(DATA_YAML):
        print(f" ERROR: Cannot find {DATA_YAML}")
        print("Check your dataset folder name and path.")
        return

    print("Dataset found")
    print(f"Using dataset: {DATA_YAML}")
    print("=" * 60)

    print("Loading YOLOv8 pretrained weights (COCO)...")
    model = YOLO("yolov8n.pt")

    # Train
    print("\n STARTING TRAINING")
    print("=" * 60)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        patience=10,
        project=PROJECT_NAME,
        name=RUN_NAME,
        save=True,

        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        optimizer="AdamW",
        lr0=0.001,
        verbose=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    save_dir = f"{PROJECT_NAME}/{RUN_NAME}"
    print(f"Best model: {save_dir}/weights/best.pt")
    print(f"Last model: {save_dir}/weights/last.pt")

    # Validate
    print("\nRunning validation...")
    metrics = model.val()

    print("\n📊 METRICS")
    print(f"mAP@50:     {metrics.box.map50:.3f}")
    print(f"mAP@50-95:  {metrics.box.map:.3f}")
    print(f"Precision (mean): {np.mean(metrics.box.p):.3f}")
    print(f"Recall (mean):    {np.mean(metrics.box.r):.3f}")

    print("\n DONE")

if __name__ == "__main__":
    main()
