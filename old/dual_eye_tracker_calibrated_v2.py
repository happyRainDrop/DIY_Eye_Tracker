import cv2
import numpy as np

from Orlosky3DEyeTrackerLite import getPupilAndPicture, process_frame

PI = "http://192.168.50.149:8000"

eye_stream = cv2.VideoCapture(PI + "/eye_feed")
world_stream = cv2.VideoCapture(PI + "/world_feed")

# -------------------------------------------------------------
# 1. LOAD CALIBRATION
# -------------------------------------------------------------
try:
    # rows:
    # [pupil_x, pupil_y, world_x, world_y]
    calib_data = np.load("calibration_2d_vectors.npy")
    print(f"Loaded {len(calib_data)} calibration points.")
    print(f" Shape: {calib_data.shape}")
    print(f" Calib Data: {calib_data}")
except FileNotFoundError:
    print("Calibration file not found.")
    exit()

P = calib_data[:, :2]
T_x = calib_data[:, 2]
T_y = calib_data[:, 3]


# -------------------------------------------------------------
# 2. POLYNOMIAL FEATURE MATRIX
# -------------------------------------------------------------
def create_feature_matrix(points):
    """
    Second-order polynomial features.

    Input:
        [px, py]

    Output:
        [1,
         px,
         py,
         px²,
         py²,
         px*py]
    """
    px = points[:, 0]
    py = points[:, 1]

    return np.column_stack((
        np.ones_like(px),
        px,
        py,
        px**2,
        py**2,
        px * py
    ))


A = create_feature_matrix(P)

model_coeffs_x, _, _, _ = np.linalg.lstsq(A, T_x, rcond=None)
model_coeffs_y, _, _, _ = np.linalg.lstsq(A, T_y, rcond=None)
print(f" Model coeffs x: {model_coeffs_x}")
print(f" Model coeffs y: {model_coeffs_y}")

# -------------------------------------------------------------
# 3. KEEP POINT ON SCREEN
# -------------------------------------------------------------
def project_to_bounds(x, y, width, height):

    cx = width / 2
    cy = height / 2

    dx = x - cx
    dy = y - cy

    if 0 <= x < width and 0 <= y < height:
        return int(x), int(y), False

    scales = []

    if dx != 0:
        scales.append((0 - cx) / dx)
        scales.append((width - 1 - cx) / dx)

    if dy != 0:
        scales.append((0 - cy) / dy)
        scales.append((height - 1 - cy) / dy)

    candidates = []

    for t in scales:
        if t <= 0:
            continue

        px = cx + t * dx
        py = cy + t * dy

        if 0 <= px < width and 0 <= py < height:
            candidates.append((t, px, py))

    if candidates:
        _, px, py = min(candidates, key=lambda c: c[0])
        return int(px), int(py), True

    return (
        int(np.clip(x, 0, width - 1)),
        int(np.clip(y, 0, height - 1)),
        True,
    )


# -------------------------------------------------------------
# 4. LIVE LOOP
# -------------------------------------------------------------
print("Starting Live Tracking...")

while True:

    ret_eye, eye_frame = eye_stream.read()
    ret_world, world_frame = world_stream.read()

    if not ret_eye or not ret_world:
        continue

    pupil, annotated_eye = getPupilAndPicture(eye_frame)

    cv2.imshow("Eye Tracker", eye_frame)

    if pupil is not None:

        px, py = pupil

        features = create_feature_matrix(np.array([[px, py]]))[0]

        world_x = float(np.dot(features, model_coeffs_x))
        world_y = float(np.dot(features, model_coeffs_y))

        #print(f"Features: {features}")
        #print(f"   Mapping pupil to world: {px}, {py} --> {world_x}, {world_y}")

        h, w = world_frame.shape[:2]

        world_x, world_y, clamped = project_to_bounds(
            world_x,
            world_y,
            w,
            h
        )

        color = (0, 165, 255) if clamped else (0, 255, 0)

        cv2.circle(world_frame, (world_x, world_y), 8, (0, 0, 255), -1)
        cv2.circle(world_frame, (world_x, world_y), 18, color, 2)

        cv2.line(world_frame,
                 (world_x - 22, world_y),
                 (world_x + 22, world_y),
                 color,
                 1)

        cv2.line(world_frame,
                 (world_x, world_y - 22),
                 (world_x, world_y + 22),
                 color,
                 1)

        if clamped:
            cv2.putText(world_frame, "OUT OF VIEW", (world_x + 15, world_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,)

    else:
        cv2.putText(world_frame,"TRACKING LOST",(20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255),2,)

    cv2.imshow("World Gaze Mapping", world_frame)
    #process_frame(eye_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

eye_stream.release()
world_stream.release()
cv2.destroyAllWindows()