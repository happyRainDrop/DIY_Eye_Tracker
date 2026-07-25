# THIS RUNS ON PI

from picamera2 import Picamera2
import cv2
from flask import Flask, Response

app = Flask(__name__)

# -------------------------------
# ROI CONFIGURATION
# -------------------------------

print("Enter crop coordinates.")
print("Leave blank to stream full frame.")

x1 = input("Upper-left X: ")
y1 = input("Upper-left Y: ")
x2 = input("Bottom-right X: ")
y2 = input("Bottom-right Y: ")

if x1 and y1 and x2 and y2:
    ROI = (int(x1), int(y1), int(x2), int(y2))
    print(f"Cropping enabled: {ROI}")
else:
    ROI = None
    print("Streaming full frame")


# -------------------------------
# CAMERA SETUP
# -------------------------------

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={"size": (640,480)}
)

picam2.configure(camera_config)
picam2.start()


# -------------------------------
# STREAM GENERATOR
# -------------------------------

def generate_frames():

    while True:

        frame = picam2.capture_array()

        # Crop if ROI exists
        if ROI:
            x1, y1, x2, y2 = ROI

            frame = frame[y1:y2, x1:x2]


        # Encode cropped image
        ret, buffer = cv2.imencode(
            '.jpg',
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )


# -------------------------------
# WEB ROUTES
# -------------------------------

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def index():

    return """
    <html>
    <body>
    <h1>NoIR Camera Feed</h1>
    <img src='/video_feed'>
    </body>
    </html>
    """


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000
    )