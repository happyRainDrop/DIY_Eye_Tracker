import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import zmq

# Import pupil extraction function from your existing pipeline
from Orlosky3DEyeTrackerLite import getPupilAndPicture

PI_IP = "192.168.50.149"
PORT = "5555"


# -------------------------------------------------------------
# ZMQ VIDEO RECEIVER THREAD
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# POLYNOMIAL MODEL & BOUNDS HELPERS
# -------------------------------------------------------------
def create_feature_matrix_2nd_order(points):
    """Generates 2nd-order polynomial features: [1, px, py, px^2, py^2, px*py]."""
    px = points[:, 0]
    py = points[:, 1]
    return np.column_stack((np.ones_like(px), px, py, px**2, py**2, px * py))

def create_feature_matrix(points):
    """
    Generates 3rd-order polynomial features.

    Features:
    [1,
     px, py,
     px², px·py, py²,
     px³, px²·py, px·py², py³]
    """
    px = points[:, 0]
    py = points[:, 1]

    return np.column_stack((
        np.ones_like(px),

        # Linear
        px,
        py,

        # Quadratic
        px**2,
        px * py,
        py**2,

        # Cubic
        px**3,
        (px**2) * py,
        px * (py**2),
        py**3,
    ))

def project_to_bounds(x, y, width, height):
    """Ray-casts gaze coordinate from screen center to frame boundaries if out-of-bounds."""
    cx = width / 2.0
    cy = height / 2.0
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
        px_c = cx + t * dx
        py_c = cy + t * dy
        if 0 <= px_c < width and 0 <= py_c < height:
            candidates.append((t, px_c, py_c))

    if candidates:
        _, px_c, py_c = min(candidates, key=lambda c: c[0])
        return int(px_c), int(py_c), True

    return (
        int(np.clip(x, 0, width - 1)),
        int(np.clip(y, 0, height - 1)),
        True,
    )


# -------------------------------------------------------------
# MAIN APPLICATION GUI
# -------------------------------------------------------------
class EyeTrackerGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracker & Side-by-Side Recorder")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Receiver Thread Handle
        self.receiver_thread = None

        # State Variables
        self.is_running = False
        self.is_recording = False

        # Timing and FPS Tracking
        self.measured_fps = 10.0  # Default fallback FPS
        self.recorded_frames = []

        # Model Coefficients
        self.model_coeffs_x = None
        self.model_coeffs_y = None

        self._load_calibration()
        self._build_ui()
        self._start_tracking_thread()

    def _load_calibration(self):
        """Loads calibration data and trains polynomial mapping model."""
        try:
            calib_data = np.load("calibration_2d_vectors.npy")
            P = calib_data[:, :2]
            T_x = calib_data[:, 2]
            T_y = calib_data[:, 3]

            A = create_feature_matrix(P)
            self.model_coeffs_x, _, _, _ = np.linalg.lstsq(A, T_x, rcond=None)
            self.model_coeffs_y, _, _, _ = np.linalg.lstsq(A, T_y, rcond=None)
            print("Calibration loaded and polynomial model trained successfully.")
        except FileNotFoundError:
            messagebox.showerror(
                "Error", "calibration_2d_vectors.npy not found!"
            )
            self.root.destroy()

    def _build_ui(self):
        """Constructs layout, buttons, and display canvases."""
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_record = ttk.Button(
            control_frame, text="Start Recording", command=self.toggle_recording
        )
        self.btn_record.pack(side=tk.LEFT, padx=5)

        self.lbl_status = ttk.Label(
            control_frame, text="Status: Streaming", font=("Arial", 11, "bold")
        )
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        video_frame = ttk.Frame(self.root, padding=10)
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lbl_eye_feed = ttk.Label(video_frame)
        self.lbl_eye_feed.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_world_feed = ttk.Label(video_frame)
        self.lbl_world_feed.pack(side=tk.LEFT, padx=5, pady=5)

    def toggle_recording(self):
        """Toggles video recording on and off."""
        if not self.is_recording:
            # START RECORDING
            self.recorded_frames = []
            self.is_recording = True
            self.btn_record.config(text="Stop & Save Recording")
            self.lbl_status.config(
                text="Status: RECORDING...", foreground="red"
            )
        else:
            # STOP RECORDING
            self.is_recording = False
            self.btn_record.config(text="Start Recording")
            self.lbl_status.config(
                text="Status: Streaming", foreground="black"
            )

            # Prompt user for save path and write out video
            self._save_recorded_video()

    def _save_recorded_video(self):
        """Opens file dialog and encodes recorded frames at exact measured playback speed."""
        if not self.recorded_frames:
            messagebox.showwarning("Warning", "No frames were recorded.")
            return

        # Prompt for save path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")],
            title="Save Recorded Video As...",
        )

        if not file_path:
            # User canceled save dialog
            self.recorded_frames.clear()
            return

        # Write video to file at the measured FPS
        comp_h, comp_w = self.recorded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            file_path, fourcc, self.measured_fps, (comp_w, comp_h)
        )

        for frame in self.recorded_frames:
            writer.write(frame)

        writer.release()
        self.recorded_frames.clear()

        messagebox.showinfo(
            "Saved Successfully",
            f"Video saved at actual original speed ({self.measured_fps:.1f} FPS):\n{file_path}",
        )

    def _start_tracking_thread(self):
        """Spawns ZeroMQ receiver and processing thread."""
        self.is_running = True

        # Initialize and start ZMQ Video Receiver Thread
        self.receiver_thread = VideoReceiverThread(PI_IP, PORT)
        self.receiver_thread.start()

        # Start Processing Loop Thread
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def _tracking_loop(self):
        """Main processing loop."""
        display_w, display_h = 480, 360  # Size for UI preview labels

        # Dynamic FPS calculation variables
        frame_count = 0
        fps_start_time = time.time()

        while self.is_running:
            # Fetch latest frame pair from ZMQ thread
            eye_frame, world_frame = self.receiver_thread.get_frames()

            if eye_frame is None or world_frame is None:
                time.sleep(0.001)
                continue

            # FPS calculation updated every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                elapsed = now - fps_start_time
                if elapsed > 0:
                    self.measured_fps = round(10.0 / elapsed, 2)
                fps_start_time = now

            # Process Eye Frame
            pupil, annotated_eye = getPupilAndPicture(eye_frame)

            # Process Gaze Mapping
            if pupil is not None:
                px, py = pupil
                features = create_feature_matrix(np.array([[px, py]]))[0]

                world_x = float(np.dot(features, self.model_coeffs_x))
                world_y = float(np.dot(features, self.model_coeffs_y))

                h, w = world_frame.shape[:2]
                world_x, world_y, clamped = project_to_bounds(
                    world_x, world_y, w, h
                )

                color = (0, 165, 255) if clamped else (0, 255, 0)

                # Draw gaze target overlay
                cv2.circle(world_frame, (world_x, world_y), 8, (0, 0, 255), -1)
                cv2.circle(world_frame, (world_x, world_y), 18, color, 2)
                cv2.line(
                    world_frame,
                    (world_x - 22, world_y),
                    (world_x + 22, world_y),
                    color,
                    1,
                )
                cv2.line(
                    world_frame,
                    (world_x, world_y - 22),
                    (world_x, world_y + 22),
                    color,
                    1,
                )

                if clamped:
                    cv2.putText(
                        world_frame,
                        "OUT OF VIEW",
                        (world_x + 15, world_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
            else:
                cv2.putText(
                    world_frame,
                    "TRACKING LOST",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Rescale annotated eye stream to match world height for side-by-side view
            h_w, w_w = world_frame.shape[:2]
            h_e, w_e = annotated_eye.shape[:2]
            scale_ratio = h_w / float(h_e)
            eye_resized = cv2.resize(
                annotated_eye, (int(w_e * scale_ratio), h_w)
            )

            # Construct Composite Frame (Eye | World)
            composite_frame = np.hstack((eye_resized, world_frame))

            # Collect raw frames into buffer while recording
            if self.is_recording:
                self.recorded_frames.append(composite_frame.copy())

            # Prepare images for GUI rendering (BGR -> RGB)
            img_eye_rgb = cv2.cvtColor(
                cv2.resize(annotated_eye, (display_w, display_h)),
                cv2.COLOR_BGR2RGB,
            )
            img_world_rgb = cv2.cvtColor(
                cv2.resize(world_frame, (display_w, display_h)),
                cv2.COLOR_BGR2RGB,
            )

            img_eye_tk = ImageTk.PhotoImage(image=Image.fromarray(img_eye_rgb))
            img_world_tk = ImageTk.PhotoImage(
                image=Image.fromarray(img_world_rgb)
            )

            # Push frames to Tkinter main thread
            self.root.after(0, self._update_gui, img_eye_tk, img_world_tk)

    def _update_gui(self, img_eye, img_world):
        """Updates GUI label components."""
        self.lbl_eye_feed.configure(image=img_eye)
        self.lbl_eye_feed.image = img_eye

        self.lbl_world_feed.configure(image=img_world)
        self.lbl_world_feed.image = img_world

    def on_closing(self):
        """Cleans up resources when closing window."""
        self.is_running = False
        if self.receiver_thread is not None:
            self.receiver_thread.stop()
        self.root.destroy()


# -------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackerGUI(root)
    root.mainloop()