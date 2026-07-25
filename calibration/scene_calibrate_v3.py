import os
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import pygame
import zmq

# --- IMPORT ORLOSKY DETECTOR LOGIC ---
from Orlosky3DEyeTrackerLite import getPupilAndPicture

# Configuration
PI_IP = "192.168.50.149"
PORT = "5555"

SETUP_TIME = 5
SETTLE_TIME = 1.0  # seconds
SAMPLE_TIME = 3.0  # seconds
LIME = (50, 255, 50)
DARK_THRESH = 100
DEBUG_DIR = "calibration_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
pink = (255, 0, 255)

# Remove old calibration files
dir_path = Path(DEBUG_DIR)
for item in dir_path.iterdir():
    if item.is_file():
        item.unlink()


# --------------------------------
# ZMQ VIDEO RECEIVER THREAD CLASS
# --------------------------------
class VideoReceiverThread(threading.Thread):
    def __init__(self, pi_ip, port):
        super().__init__()
        self.daemon = True
        self.address = f"tcp://{pi_ip}:{port}"

        self.latest_eye_frame = None
        self.latest_world_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Lower receive high water mark to keep latency minimal
        socket.setsockopt(zmq.RCVHWM, 2)

        socket.connect(self.address)
        print(f"Connected to stream at {self.address}")

        while self.running:
            try:
                # Receive multi-part message (Eye JPEG + World JPEG)
                parts = socket.recv_multipart()

                if len(parts) >= 2:
                    eye_bytes, world_bytes = parts[0], parts[1]

                    # Decode Eye Frame
                    if eye_bytes:
                        eye_np = np.frombuffer(eye_bytes, dtype=np.uint8)
                        eye_img = cv2.imdecode(eye_np, cv2.IMREAD_COLOR)
                    else:
                        eye_img = None

                    # Decode World Frame
                    if world_bytes:
                        world_np = np.frombuffer(world_bytes, dtype=np.uint8)
                        world_img = cv2.imdecode(world_np, cv2.IMREAD_COLOR)
                    else:
                        world_img = None

                    with self.lock:
                        self.latest_eye_frame = eye_img
                        self.latest_world_frame = world_img

            except Exception as e:
                print(f"Receiver error: {e}")
                time.sleep(0.01)

        socket.close()
        context.term()

    def get_frames(self):
        with self.lock:
            eye = (
                self.latest_eye_frame.copy()
                if self.latest_eye_frame is not None
                else None
            )
            world = (
                self.latest_world_frame.copy()
                if self.latest_world_frame is not None
                else None
            )
            return eye, world

    def stop(self):
        self.running = False


# --------------------------------
# FIND GREEN CALIBRATION DOT IN FOREHEAD CAMERA
# --------------------------------
def find_target(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 120, 120])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# --------------------------------
# MAIN PROGRAM
# --------------------------------
if __name__ == "__main__":
    # Start video receiver thread
    receiver = VideoReceiverThread(PI_IP, PORT)
    receiver.start()

    # --------------------------------
    # CAMERA POSITION SETUP
    # --------------------------------
    print("""
    SETUP MODE
    Adjust yourself until:
    - Eye camera sees pupil
    - Forehead camera sees laptop screen

    Press SPACE when ready.
    Press Q to quit.
    """)

    EYE_IMG_W, EYE_IMG_H = 0, 0
    WORLD_IMG_W, WORLD_IMG_H = 0, 0

    while True:
        eye, world = receiver.get_frames()

        if eye is not None:
            EYE_IMG_H, EYE_IMG_W = eye.shape[:2]
            cv2.imshow("Eye camera", eye)

        if world is not None:
            WORLD_IMG_H, WORLD_IMG_W = world.shape[:2]
            cv2.imshow("Forehead camera", world)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space bar
            break
        if key == ord("q"):
            receiver.stop()
            cv2.destroyAllWindows()
            quit()

    print(f"Eye image: w x h = {EYE_IMG_W} x {EYE_IMG_H}")
    print(f"World image: w x h = {WORLD_IMG_W} x {WORLD_IMG_H}")
    cv2.destroyAllWindows()

    # --------------------------------
    # PYGAME CALIBRATION SCREEN
    # --------------------------------
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_w, screen_h = screen.get_size()

    points = [
        (0.1, 0.1), (0.3, 0.1), (0.5, 0.1), (0.7, 0.1), (0.9, 0.1),
        (0.9, 0.3), (0.7, 0.3), (0.5, 0.3), (0.3, 0.3), (0.1, 0.3),
        (0.1, 0.5), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5),
        (0.9, 0.7), (0.7, 0.7), (0.5, 0.7), (0.3, 0.7), (0.1, 0.7),
        (0.1, 0.9), (0.3, 0.9), (0.5, 0.9), (0.7, 0.9), (0.9, 0.9)
    ]

    calibration = []

    # --------------------------------
    # CALIBRATION LOOP
    # --------------------------------
    for index, (px, py) in enumerate(points):
        dot = (int(px * screen_w), int(py * screen_h))
        print("Looking at:", dot)

        gaze_2d_samples = []
        target_samples = []
        world_last = None
        eye_last = None
        gaze_pic_last = None

        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, pink, dot, 18)
        pygame.display.flip()

        time.sleep(SETTLE_TIME)
        start = time.time()

        while time.time() - start < SAMPLE_TIME:
            # Process Pygame event queue to maintain UI responsiveness
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    receiver.stop()
                    pygame.quit()
                    cv2.destroyAllWindows()
                    quit()

            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, LIME, dot, 18)
            pygame.display.flip()

            eye, world = receiver.get_frames()
            gaze_coords = None

            if eye is not None and world is not None:
                # CALL THE 2D TRACKER HERE
                gaze_coords, gaze_pic = getPupilAndPicture(eye)
                target = find_target(world)

                if gaze_coords is not None and target is not None:
                    gaze_2d_samples.append(gaze_coords)
                    if time.time() - start > SAMPLE_TIME / 5:
                        target_samples.append(target)

                    world_last = world.copy()
                    eye_last = eye.copy()
                    gaze_pic_last = (
                        gaze_pic.copy() if gaze_pic is not None else None
                    )

        if len(gaze_2d_samples) > 5 and len(target_samples) > 1:
            # --- Clean & Average 2D Gaze Vector Samples ---
            samples_2d = np.array(gaze_2d_samples)
            median_2d = np.median(samples_2d, axis=0)
            dist_2d = np.linalg.norm(samples_2d - median_2d, axis=1)
            thresh_2d = np.mean(dist_2d) + 2 * np.std(dist_2d)
            samples_2d = samples_2d[dist_2d < thresh_2d]

            avg_gaze_2d = np.mean(samples_2d, axis=0)

            # --- Clean & Average Target Positions ---
            target_samples = np.array(target_samples)
            median_t = np.median(target_samples, axis=0)
            dist_t = np.linalg.norm(target_samples - median_t, axis=1)
            thresh_t = np.mean(dist_t) + 2 * np.std(dist_t)
            target_samples = target_samples[dist_t < thresh_t]
            avg_target = np.mean(target_samples, axis=0)
            avg_target = (int(avg_target[0]), int(avg_target[1]))

            # --- Save 2D Gaze Matrix Mapping Data ---
            calibration.append([
                avg_gaze_2d[0], avg_gaze_2d[1],
                avg_target[0], avg_target[1],
            ])

            print(f"Recorded Gaze: {avg_gaze_2d} -> Target Pixel: {avg_target}")

            # Debug logs
            if world_last is not None:
                cv2.circle(world_last, avg_target, 15, pink, 3)
                cv2.imwrite(f"{DEBUG_DIR}/dot_{index}_target.png", world_last)
            np.save(f"{DEBUG_DIR}/dot_{index}_2d_vector.npy", avg_gaze_2d)
            
            if gaze_pic_last is not None:
                cv2.imwrite(f"{DEBUG_DIR}/dot_{index}_eye.png", gaze_pic_last)

        else:
            print(f"\tOops, only {len(gaze_2d_samples)} gaze samples found and only {len(target_samples)} target samples found.")

    # Save final matrix
    calibration = np.array(calibration)
    np.save("calibration_2d_vectors.npy", calibration)
    print("\n2D Calibration array saved successfully.")

    # Cleanup
    receiver.stop()
    pygame.quit()
    cv2.destroyAllWindows()