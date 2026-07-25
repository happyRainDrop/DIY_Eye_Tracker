import cv2
import numpy as np

# --- IMPORT ORLOSKY DETECTOR LOGIC ---
from old.Orlosky3DEyeTracker import get_3d_gaze_vector, process_frame

PI = "http://192.168.50.149:8000"
eye_stream = cv2.VideoCapture(PI + "/eye_feed")
world_stream = cv2.VideoCapture(PI + "/world_feed")

# -------------------------------------------------------------
# 1. LOAD CALIBRATION AND FIT POLYNOMIAL MAPPING
# -------------------------------------------------------------
try:
    # Your calibration shape: [9, 5] -> rows of [G_x, G_y, G_z, T_x, T_y]
    calib_data = np.load("calibration_3d_vectors.npy")
    print(f"Successfully loaded calibration data for {len(calib_data)} points.")
except FileNotFoundError:
    print("Error: 'calibration_3d_vectors.npy' not found! Run calibration script first.")
    exit()

# Extract components
G = calib_data[:, :3]   # 3D Gaze Vectors (X, Y, Z)
T_x = calib_data[:, 3]  # Target World X
T_y = calib_data[:, 4]  # Target World Y

def create_feature_matrix(gaze_vectors):
    """
    Transforms 3D gaze vectors into 2nd-order polynomial features.
    Matches the quadratic structure to handle fish-eye lenses.
    """
    x = gaze_vectors[:, 0]
    y = gaze_vectors[:, 1]
    z = gaze_vectors[:, 2]
    
    # Quadratic features: [1, x, y, z, x^2, y^2, z^2, x*y, y*z, z*x]
    return np.column_stack((
        np.ones_like(x), x, y, z,
        x**2, y**2, z**2,
        x*y, y*z, z*x
    ))

# Fit the regression mapping coefficients
A = create_feature_matrix(G)
model_coeffs_x, _, _, _ = np.linalg.lstsq(A, T_x, rcond=None)
model_coeffs_y, _, _, _ = np.linalg.lstsq(A, T_y, rcond=None)

# -------------------------------------------------------------
# 2. BOUNDS BOUNDING MATHEMATICS (RAY-CAST CLIPPING TO EDGE)
# -------------------------------------------------------------
def project_to_bounds(x, y, width, height):
    """
    If gaze falls outside screen boundaries, projects the point onto 
    the frame edge along a vector extending from the screen center.
    """
    cx = width / 2.0
    cy = height / 2.0

    dx = x - cx
    dy = y - cy

    # If it is inside the viewport, return directly
    if 0 <= x < width and 0 <= y < height:
        return int(x), int(y), False  # False = not clamped

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
        return int(px), int(py), True  # True = clamped to edge

    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)), True

# -------------------------------------------------------------
# 3. LIVE LOOP
# -------------------------------------------------------------
print("Starting Live Tracking... Press 'q' to quit.")

while True:
    ret1, eye_frame = eye_stream.read()
    ret2, world_frame = world_frame_raw = world_stream.read()
    
    if not ret1 or not ret2:
        print("Waiting for streams...")
        continue

    # Requirement 1: Run the diagnostic window from eye tracker script
    _ = process_frame(eye_frame)

    # Requirement 2: Fetch 3D vector, map, and enforce screen constraints
    gaze_vector = get_3d_gaze_vector(eye_frame)
    
    if gaze_vector is not None:
        # Convert single vector into a row matrix and get features
        feat = create_feature_matrix(np.array([gaze_vector]))[0]
        
        # Polynomial prediction
        wx = float(np.dot(feat, model_coeffs_x))
        wy = float(np.dot(feat, model_coeffs_y))
        
        # Enforce view restraints using world camera dimensions
        h, w = world_frame.shape[:2]
        world_x, world_y, is_clamped = project_to_bounds(wx, wy, w, h)
        
        # Visuals: Change reticle color if user is looking out-of-bounds
        if is_clamped:
            # Orange warning reticle pinned to the edge
            color_ring = (0, 165, 255) 
            cv2.putText(world_frame, "OUT OF VIEW", (world_x + 15, world_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_ring, 1)
        else:
            # Green normal reticle
            color_ring = (0, 255, 0)
            
        # Draw target reticle dot 
        cv2.circle(world_frame, (world_x, world_y), 8, (0, 0, 255), -1)          # Red center dot
        cv2.circle(world_frame, (world_x, world_y), 18, color_ring, 2)           # Status ring
        cv2.line(world_frame, (world_x - 22, world_y), (world_x + 22, world_y), color_ring, 1)
        cv2.line(world_frame, (world_x, world_y - 22), (world_x, world_y + 22), color_ring, 1)
    else:
        cv2.putText(world_frame, "TRACKING LOST", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Live World Gaze Mapping", world_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

eye_stream.release()
world_stream.release()
cv2.destroyAllWindows()