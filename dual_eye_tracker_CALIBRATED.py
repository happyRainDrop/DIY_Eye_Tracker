import cv2
import numpy as np
from gaze_mapper import GazeMapper

PI = "http://192.168.50.149:8000"
CALIBRATION_FILE = "calibration_vectors.npy"
DARK_THRESH = 100
mapper = GazeMapper(CALIBRATION_FILE,neutral_pupil=(73,54))

# --------------------------------
# CAMERA STREAMS
# --------------------------------
eye_stream = cv2.VideoCapture(PI + "/eye_feed")
world_stream = cv2.VideoCapture(PI + "/world_feed")

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


    # --------------------------
    # Remove small noise
    # --------------------------

    kernel = np.ones((5,5),np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
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


    return (
        int(M["m10"]/M["m00"]),
        int(M["m01"]/M["m00"])
    )

# --------------------------------
# MAIN LOOP
# --------------------------------

while True:
    ret1, eye = eye_stream.read()
    ret2, world = world_stream.read()
    if not ret1 or not ret2:
        continue

    pupil = find_pupil(eye)

    if pupil: 
        
        # Show pupil location
        cv2.circle(eye,pupil,5,(0,255,0),-1)

        # Map onto world
        gaze = mapper.map_pupil_to_world(pupil)

        if gaze:

            gx,gy=gaze
            h,w=world.shape[:2]

            if (0 <= gx < w and 0 <= gy < h):
                cv2.circle(world, (gx,gy),12, (0,255,0),-1)


    cv2.imshow("Eye Camera - Pupil",eye)
    cv2.imshow("World Camera - Gaze",world)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

eye_stream.release()
world_stream.release()

cv2.destroyAllWindows()