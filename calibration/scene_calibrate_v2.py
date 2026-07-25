import cv2
import pygame
import numpy as np
import time
import os
from pathlib import Path

# --- IMPORT ORLOSKY DETECTOR LOGIC ---
# Assuming you have packaged the 2d tracking loop into a function/class
from Orlosky3DEyeTrackerLite import getPupilAndPicture

SETUP_TIME = 5
SETTLE_TIME = 1.0      # seconds
SAMPLE_TIME = 5.0       # seconds
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

    gaze_2d_samples = []
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
        gaze_coords = None

        if ret1 and ret2:
            # CALL THE 2d TRACKER HERE
            gaze_coords, gaze_pic = getPupilAndPicture(eye) 
            target = find_target(world)

            if gaze_coords is not None and target is not None:
                gaze_2d_samples.append(gaze_coords)
                if (time.time() - start > SAMPLE_TIME / 5):
                    target_samples.append(target)

                world_last = world.copy()
                eye_last = eye.copy()

    if len(gaze_2d_samples) > 5:
        # --- Clean & Average 2d Gaze Vector Samples ---
        samples_2d = np.array(gaze_2d_samples)
        median_2d = np.median(samples_2d, axis=0)
        dist_2d = np.linalg.norm(samples_2d - median_2d, axis=1)
        thresh_2d = np.mean(dist_2d) + 2 * np.std(dist_2d)
        samples_2d = samples_2d[dist_2d < thresh_2d]
        
        # Compute mean vector and re-normalize it to be a pure unit vector
        avg_gaze_2d = np.mean(samples_2d, axis=0)
        # avg_gaze_2d /= np.linalg.norm(avg_gaze_2d)

        # --- Clean & Average Target Positions ---
        target_samples = np.array(target_samples)
        median_t = np.median(target_samples, axis=0)
        dist_t = np.linalg.norm(target_samples - median_t, axis=1)
        thresh_t = np.mean(dist_t) + 2 * np.std(dist_t)
        target_samples = target_samples[dist_t < thresh_t]
        avg_target = np.mean(target_samples, axis=0)
        avg_target = (int(avg_target[0]), int(avg_target[1]))

        # --- Save 2d Gaze Matrix Mapping Data ---
        # Storing: [Gaze_X, Gaze_Y, Gaze_Z, Target_X, Target_Y]
        calibration.append([
            avg_gaze_2d[0], avg_gaze_2d[1], # avg_gaze_2d[2], 
            avg_target[0], avg_target[1]
        ])

        print(f"Recorded Gaze: {avg_gaze_2d} -> Target Pixel: {avg_target}")

        # Debug logs
        if world_last is not None:
            cv2.circle(world_last, avg_target, 15, pink, 3)
            cv2.imwrite(f"{DEBUG_DIR}/dot_{index}_target.png", world_last)
        np.save(f"{DEBUG_DIR}/dot_{index}_2d_vector.npy", avg_gaze_2d)
        print(f"\t\t the gaze here is {gaze_coords}")
        cv2.imwrite(f"{DEBUG_DIR}/dot_{index}_eye.png", gaze_pic)

    else:
        print(f"\tOops, only {len(gaze_2d_samples)} gaze samples found.")

# Save final matrix
calibration = np.array(calibration)
np.save("calibration_2d_vectors.npy", calibration)
print("\n2d Calibration array saved successfully.")

pygame.quit()
eye_stream.release()
world_stream.release()
cv2.destroyAllWindows()