import cv2
import pygame
import numpy as np
import time
import os
from pathlib import Path

# --- IMPORT ORLOSKY DETECTOR LOGIC ---
# Assuming you have packaged the 3D tracking loop into a function/class
from old.Orlosky3DEyeTracker import get_3d_gaze_vector

SETUP_TIME = 5
SETTLE_TIME = 1.0      # seconds
SAMPLE_TIME = 3.0       # seconds
LIME = (50, 255, 50)

PI = "http://192.168.50.149:8000"
eye_stream = cv2.VideoCapture(PI + "/eye_feed")
world_stream = cv2.VideoCapture(PI + "/world_feed")

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
# FIND GREEN CALIBRATION DOT IN FOREHEAD CAMERA
# --------------------------------
def find_target(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 120, 120])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

while True:
    ret1, eye = eye_stream.read()
    ret2, world = world_stream.read()

    EYE_IMG_W, EYE_IMG_H = 0, 0 
    WORLD_IMG_W, WORLD_IMG_H = 0, 0

    if ret1:
        # Instead of generic pupil detection, use the Orlosky tracking visualizer if available
        EYE_IMG_H, EYE_IMG_W = eye.shape[:2]
        cv2.imshow("Eye camera", eye)

    if ret2: 
        WORLD_IMG_H, WORLD_IMG_W = world.shape[:2]
        cv2.imshow("Forehead camera", world)

    key = cv2.waitKey(1) & 0xff
    if key == 32:
        break
    if key == ord("q"):
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
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.9, 0.5), (0.5, 0.5), (0.1, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
]

calibration = []

# --------------------------------
# CALIBRATION LOOP
# --------------------------------
for index, (px, py) in enumerate(points):
    dot = (int(px * screen_w), int(py * screen_h))
    print("Looking at:", dot)

    gaze_3d_samples = []
    target_samples = []
    world_last = None
    eye_last = None

    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, pink, dot, 18)
    pygame.display.flip()

    time.sleep(SETTLE_TIME)
    start = time.time()

    while time.time() - start < SAMPLE_TIME:
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, LIME, dot, 18)
        pygame.display.flip()

        ret1, eye = eye_stream.read()
        ret2, world = world_stream.read()

        if ret1 and ret2:
            # CALL THE 3D TRACKER HERE
            gaze_vector = get_3d_gaze_vector(eye) 
            target = find_target(world)

            if gaze_vector is not None and target is not None:
                gaze_3d_samples.append(gaze_vector)
                if (time.time() - start > SAMPLE_TIME / 5):
                    target_samples.append(target)

                world_last = world.copy()
                eye_last = eye.copy()

    if len(gaze_3d_samples) > 5:
        # --- Clean & Average 3D Gaze Vector Samples ---
        samples_3d = np.array(gaze_3d_samples)
        median_3d = np.median(samples_3d, axis=0)
        dist_3d = np.linalg.norm(samples_3d - median_3d, axis=1)
        thresh_3d = np.mean(dist_3d) + 2 * np.std(dist_3d)
        samples_3d = samples_3d[dist_3d < thresh_3d]
        
        # Compute mean vector and re-normalize it to be a pure unit vector
        avg_gaze_3d = np.mean(samples_3d, axis=0)
        avg_gaze_3d /= np.linalg.norm(avg_gaze_3d)

        # --- Clean & Average Target Positions ---
        target_samples = np.array(target_samples)
        median_t = np.median(target_samples, axis=0)
        dist_t = np.linalg.norm(target_samples - median_t, axis=1)
        thresh_t = np.mean(dist_t) + 2 * np.std(dist_t)
        target_samples = target_samples[dist_t < thresh_t]
        avg_target = np.mean(target_samples, axis=0)
        avg_target = (int(avg_target[0]), int(avg_target[1]))

        # --- Save 3D Gaze Matrix Mapping Data ---
        # Storing: [Gaze_X, Gaze_Y, Gaze_Z, Target_X, Target_Y]
        calibration.append([
            avg_gaze_3d[0], avg_gaze_3d[1], avg_gaze_3d[2], 
            avg_target[0], avg_target[1]
        ])

        print(f"Recorded 3D Gaze Vector: {avg_gaze_3d} -> Target Pixel: {avg_target}")

        # Debug logs
        if world_last is not None:
            cv2.circle(world_last, avg_target, 15, pink, 3)
            cv2.imwrite(f"{DEBUG_DIR}/dot_{index}_target.png", world_last)
        np.save(f"{DEBUG_DIR}/dot_{index}_3d_vector.npy", avg_gaze_3d)

    else:
        print(f"\tOops, only {len(gaze_3d_samples)} 3D gaze samples found.")

# Save final matrix
calibration = np.array(calibration)
np.save("calibration_3d_vectors.npy", calibration)
print("\n3D Calibration array saved successfully.")

pygame.quit()
eye_stream.release()
world_stream.release()
cv2.destroyAllWindows()