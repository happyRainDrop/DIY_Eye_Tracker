# RUNS ON LAPTOP
# Run this to get the region of interest to zoom in on
# This outputs coordinates for you to input to eye_camera_stream.py for the pi

import cv2


PI_STREAM = "http://192.168.50.149:8000/video_feed"


cap = cv2.VideoCapture(PI_STREAM)


ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not connect to camera")


print("""
Instructions:
1. Drag a box around the pupil region
2. Press ENTER to accept
3. Press ESC to cancel
""")


roi = cv2.selectROI(
    "Select pupil ROI",
    frame,
    fromCenter=False,
    showCrosshair=True
)


cv2.destroyAllWindows()


x, y, w, h = roi


if w == 0 or h == 0:
    print("No selection made")
else:

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    print("\nCopy these values into the Pi:")
    print("--------------------------------")
    print(f"Upper-left X: {x1}")
    print(f"Upper-left Y: {y1}")
    print(f"Bottom-right X: {x2}")
    print(f"Bottom-right Y: {y2}")

cap.release()