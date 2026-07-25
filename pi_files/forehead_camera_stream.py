# RUNS ON PI
# On laptop: Go to http://192.168.50.149:8001/video_feed to watch!

import cv2
from flask import Flask, Response

app = Flask(__name__)

cap = cv2.VideoCapture("/dev/video0")


def frames():

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():

    return Response(
        frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


app.run(
    host="0.0.0.0",
    port=8001
)