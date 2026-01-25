import cv2

gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

print("Opening camera with GStreamer...")
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Camera open failed")
    exit(1)

print("✓ Camera opened successfully")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")
        break
    
    cv2.imshow("CSI Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done")