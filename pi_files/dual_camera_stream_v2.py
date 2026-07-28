import threading
import time
import cv2
import zmq
from picamera2 import Picamera2

# =====================================
# EYE CAMERA ROI SETUP
# =====================================
print("Eye camera crop settings (Leave blank for full frame)")
x1 = input("Upper-left X: ")
y1 = input("Upper-left Y: ")
x2 = input("Bottom-right X: ")
y2 = input("Bottom-right Y: ")

if x1 and y1 and x2 and y2:
    EYE_ROI = (int(x1), int(y1), int(x2), int(y2))
    print("Eye crop:", EYE_ROI)
else:
    EYE_ROI = None
    print("Eye camera full frame")

# =====================================
# THREADED USB CAMERA READER
# =====================================

class USBVideoCapture:

    def __init__(self, src="/dev/video0", width=640, height=480):
        # Open via explicit V4L2 backend
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
    	    raise RuntimeError(f"Cannot open {src}")

        # Request MJPEG encoding to keep USB bandwidth low and FPS high
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

        # Start thread to continuously pull USB frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        self.cap.release()
        
# =====================================
# MAIN PI STREAMING CODE
# =====================================
# 1. CSI Eye Camera Setup
cameras = Picamera2.global_camera_info()
print("PI CAMERAS: ")
print(cameras)
eye_cam = Picamera2()
eye_config = eye_cam.create_preview_configuration(main={"size": (640, 480)})
eye_cam.configure(eye_config)
eye_cam.start()

# 2. Threaded USB Forehead Camera Setup
# Ensure /dev/video0 is the correct index (pass integer 0 instead of path)

world_cam = USBVideoCapture(src=0, width=640, height=480)

# ZMQ Setup
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.setsockopt(zmq.SNDHWM, 1)
socket.bind("tcp://*:5555")

print("Streaming frames over TCP port 5555...")

try:
    while True:
        # --- Process Eye Frame ---
        eye_frame = eye_cam.capture_array()
        if EYE_ROI:
            ex1, ey1, ex2, ey2 = EYE_ROI
            eye_frame = eye_frame[ey1:ey2, ex1:ex2]
        _, eye_jpeg = cv2.imencode(".jpg", eye_frame)

        # --- Process World Frame ---
        ret, world_frame = world_cam.read()
        if ret:
            world_frame = cv2.flip(world_frame, -1)
            _, world_jpeg = cv2.imencode(".jpg", world_frame)
        else:
            world_jpeg = None

        # --- Send Frames ---
        #socket.send(b"\x00" + eye_jpeg.tobytes())

        #if world_jpeg is not None:
            #socket.send(b"\x01" + world_jpeg.tobytes())

        socket.send_multipart([eye_jpeg.tobytes(), world_jpeg.tobytes()])

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nStopping stream...")
finally:
    eye_cam.stop()
    world_cam.release()
    socket.close()
    context.term()
