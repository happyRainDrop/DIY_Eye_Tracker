import cv2
import pygame
import numpy as np
import time
import os
from pathlib import Path

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
pink = (255,0,255)

# Remove old calibration files
dir_path = Path(DEBUG_DIR)
for item in dir_path.iterdir(): # Loop and delete files only
    if item.is_file():
        item.unlink()


# --------------------------------
# PUPIL DETECTOR
# --------------------------------
def find_pupil(frame):

    h, w = frame.shape[:2]


    # --------------------------
    # Grayscale + blur
    # --------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.GaussianBlur(
        gray,
        (7,7),
        0
    )


    # --------------------------
    # Threshold dark objects
    # --------------------------

    _, thresh = cv2.threshold(
        gray,
        DARK_THRESH,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Erode dialate
    OPEN_SIZE = 3
    OPEN_ITER = 1

    CLOSE_SIZE = 9
    CLOSE_ITER = 1

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OPEN_SIZE,OPEN_SIZE)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLOSE_SIZE,CLOSE_SIZE)
    )


    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=OPEN_ITER
    )


    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=CLOSE_ITER
    )


    # --------------------------
    # Find contours
    # --------------------------

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for c in contours:

        area = cv2.contourArea(c)


        # reject tiny blobs
        if area < 20:
            continue


        x,y,cw,ch = cv2.boundingRect(c)


        # --------------------------------
        # Ignore contours touching edges
        # (glasses frames)
        # --------------------------------

        edge_margin = 10

        if (
            x < edge_margin or
            y < edge_margin or
            x+cw > w-edge_margin or
            y+ch > h-edge_margin
        ):
            continue


        # --------------------------------
        # Reject huge objects
        # glasses are usually large
        # --------------------------------

        if area > (w*h*0.05):
            continue


        # --------------------------------
        # Check circularity
        # pupil is more circular
        # --------------------------------

        perimeter = cv2.arcLength(c,True)

        if perimeter == 0:
            continue


        circularity = (
            4*np.pi*area /
            (perimeter*perimeter)
        )


        if circularity < 0.25:
            continue


        candidates.append(c)



    if not candidates:
        return None



    # choose darkest/largest remaining pupil blob

    c=max(
        candidates,
        key=cv2.contourArea
    )


    M=cv2.moments(c)


    if M["m00"] == 0:
        return None
    
    return (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))

# --------------------------------
# CONVERT PUPIL POSITION TO VECTOR
# --------------------------------
def pupil_to_vector(pupil, neutral):

    return np.array([
        pupil[0]-neutral[0],
        pupil[1]-neutral[1]
    ])

# --------------------------------
# FIND GREEN CALIBRATION DOT
# IN FOREHEAD CAMERA
# --------------------------------
def find_target(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35,120,120])
    upper = np.array([90,255,255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask,(5,5),0)

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    c = max(contours,key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None

    M = cv2.moments(c)

    if M["m00"] == 0:
        return None

    return (
        int(M["m10"]/M["m00"]),
        int(M["m01"]/M["m00"])
    )

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

    ret1,eye=eye_stream.read()
    ret2,world=world_stream.read()

    EYE_IMG_W, EYE_IMG_H = 0,0 
    WORLD_IMG_W, WORLD_IMG_H = 0,0

    if ret1:
        pupil=find_pupil(eye)
        if pupil: cv2.circle(eye,pupil,5,(0,255,0),-1)
        EYE_IMG_H, EYE_IMG_W = eye.shape[:2]
        cv2.imshow("Eye camera",eye)

    if ret2: 
        WORLD_IMG_H, WORLD_IMG_W = world.shape[:2]
        cv2.imshow("Forehead camera",world)

    # Space to end, q to quit
    key=cv2.waitKey(1)&0xff
    if key==32:
        break
    if key==ord("q"):
        quit()

print(f"Eye image: w x h = {EYE_IMG_W} x {EYE_IMG_H}")
print(f"World image: w x h = {WORLD_IMG_W} x {WORLD_IMG_H}")

cv2.destroyAllWindows()
os.makedirs("calibration_debug", exist_ok=True)

# --------------------------------
# FIND STRAIGHT AHEAD PUPIL CENTER
# --------------------------------

print("""
Look straight ahead at the center of the screen.
Collecting neutral pupil position...
""")
neutral_samples=[]
start=time.time()
while time.time()-start < 3:

    ret,eye=eye_stream.read()

    if ret:
        pupil=find_pupil(eye)
        if pupil:
            neutral_samples.append(pupil)

neutral=np.mean(neutral_samples,axis=0)
neutral=(int(neutral[0]),int(neutral[1]))
print("Neutral pupil:",neutral)

# --------------------------------
# PYGAME CALIBRATION SCREEN
# --------------------------------

pygame.init()
screen=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
screen_w,screen_h=screen.get_size()

points=[
    (0.1,0.1),
    (0.5,0.1),
    (0.9,0.1),

    (0.9,0.5),
    (0.5,0.5),
    (0.1,0.5),

    (0.1,0.9),
    (0.5,0.9),
    (0.9,0.9)
]

calibration=[]

# --------------------------------
# CALIBRATION
# --------------------------------

for index,(px,py) in enumerate(points):

    dot=(int(px*screen_w), int(py*screen_h))

    print("Looking at:",dot)

    pupil_samples=[]
    target_samples = []
    world_last=None
    eye_last=None

    # Show the pre-settled dot
    screen.fill((0,0,0))
    pygame.draw.circle(screen, pink, dot, 18)
    pygame.display.flip()

    # Let the eye settle
    time.sleep(SETTLE_TIME)
    start = time.time()

    while time.time() - start < SAMPLE_TIME:

        screen.fill((0,0,0))
        pygame.draw.circle(screen, LIME, dot, 18)
        pygame.display.flip()

        ret1,eye=eye_stream.read()
        ret2,world=world_stream.read()

        if ret1 and ret2:
            pupil=find_pupil(eye)
            target=find_target(world)

            if pupil and target:
                pupil_samples.append(pupil)
                if (time.time() - start > SAMPLE_TIME/5):
                    target_samples.append(target)

                world_last=world.copy()
                eye_last=eye.copy()



    if len(pupil_samples)>5:

        # -----------------------------
        # Average pupil position
        # -----------------------------
        samples = np.array(pupil_samples)
        median = np.median(samples, axis=0) # Median pupil location
        dist = np.linalg.norm(samples - median, axis=1) # Distance from median
        threshold = np.mean(dist) + 2*np.std(dist) # Keep samples within 2 standard deviations
        samples = samples[dist < threshold]
        avg_pupil = np.mean(samples, axis=0)
        avg_pupil = (int(avg_pupil[0]), int(avg_pupil[1]))

        # -----------------------------
        # Average target position
        # -----------------------------
        target_samples = np.array(target_samples)
        median = np.median(target_samples, axis=0)
        dist = np.linalg.norm(target_samples - median,axis=1)
        threshold = np.mean(dist) + 2*np.std(dist)
        target_samples = target_samples[dist < threshold]
        avg_target = np.mean(target_samples, axis=0)
        avg_target = (int(avg_target[0]), int(avg_target[1]))

        # -----------------------------
        # Convert to pupil vector
        # -----------------------------
        vector=pupil_to_vector(avg_pupil, neutral)

        # -----------------------------
        # Save calibration data
        # -----------------------------
        calibration.append([vector[0], vector[1], avg_target[0], avg_target[1]])

        print(
            "Recorded:",
            vector,
            "->",
            dot
        )

        # -----------------------------
        # Save pupil image
        # -----------------------------
        cv2.circle(eye_last, avg_pupil, 8, (0,255,0), -1)

        cv2.imwrite(
            f"{DEBUG_DIR}/dot_{index}_pupil.png",
            eye_last
        )

        # -----------------------------
        # Save world target image
        # -----------------------------
        target=find_target(world_last)
        if target:
            cv2.circle(world_last, avg_target, 15, pink, 3)

        cv2.imwrite(
            f"{DEBUG_DIR}/dot_{index}_target.png",
            world_last
        )

        # -----------------------------
        # Save vector
        # -----------------------------
        np.save(
            f"{DEBUG_DIR}/dot_{index}_vector.npy",
            vector
        )

    else:
        print(f"\tOops, only {len(pupil_samples)} pupil samples and {len(target_samples)} target samples found")

# --------------------------------
# SAVE
# --------------------------------

calibration=np.array(calibration)
np.save("calibration_vectors.npy",calibration)
print()
print("Calibration saved:")
print(calibration)


pygame.quit()

eye_stream.release()
world_stream.release()
cv2.destroyAllWindows()