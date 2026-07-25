# RUNS ON PI

from picamera2 import Picamera2
import cv2
from flask import Flask, Response


app = Flask(__name__)


# =====================================
# EYE CAMERA ROI SETUP
# =====================================

print("Eye camera crop settings")
print("Leave blank for full frame")


x1 = input("Upper-left X: ")
y1 = input("Upper-left Y: ")
x2 = input("Bottom-right X: ")
y2 = input("Bottom-right Y: ")


if x1 and y1 and x2 and y2:

    EYE_ROI = (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )

    print("Eye crop:", EYE_ROI)

else:

    EYE_ROI = None
    print("Eye camera full frame")



# =====================================
# CSI EYE CAMERA
# =====================================

eye_cam = Picamera2()

eye_config = eye_cam.create_preview_configuration(
    main={
        "size": (640,480)
    }
)

eye_cam.configure(
    eye_config
)

eye_cam.start()



# =====================================
# USB FOREHEAD CAMERA
# =====================================

world_cam = cv2.VideoCapture(
    "/dev/video0"
)


world_cam.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

world_cam.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)



# =====================================
# FRAME ENCODER
# =====================================

def encode(frame):

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    return buffer.tobytes()



# =====================================
# STREAM GENERATOR
# =====================================

def generate(camera):

    while True:


        # -----------------------------
        # Eye camera
        # -----------------------------

        if camera == "eye":

            frame = eye_cam.capture_array()


            if EYE_ROI:

                x1,y1,x2,y2 = EYE_ROI

                frame = frame[
                    y1:y2,
                    x1:x2
                ]



        # -----------------------------
        # Forehead camera
        # -----------------------------

        else:

            ret, frame = world_cam.read()

            if not ret:
                continue


            # Flip upside down
            frame = cv2.flip(
                frame,
                -1
            )


        jpeg = encode(frame)


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg +
            b"\r\n"
        )



# =====================================
# ROUTES
# =====================================

@app.route("/")
def home():

    return """
    <h1>Dual Camera Stream</h1>

    Eye Camera:
    <br>
    <img src="/eye_feed">

    <br><br>

    Forehead Camera:
    <br>
    <img src="/world_feed">
    """



@app.route("/eye_feed")
def eye_feed():

    return Response(
        generate("eye"),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )



@app.route("/world_feed")
def world_feed():

    return Response(
        generate("world"),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000
    )