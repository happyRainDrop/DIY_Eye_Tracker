import threading
import time
import cv2
import numpy as np
import zmq

PI_IP = "192.168.50.149"
PORT = "5555"

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
        
        # Lower receive high water mark to keep latency minimal without dropping topic frames
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
            eye = self.latest_eye_frame.copy() if self.latest_eye_frame is not None else None
            world = self.latest_world_frame.copy() if self.latest_world_frame is not None else None
            return eye, world

    def stop(self):
        self.running = False


if __name__ == "__main__":
    receiver = VideoReceiverThread(PI_IP, PORT)
    receiver.start()

    print("Listening for video stream... Press 'q' to quit.")

    while True:
        eye_frame, world_frame = receiver.get_frames()

        if eye_frame is not None:
            cv2.imshow("Eye Stream (PC)", eye_frame)
        if world_frame is not None:
            cv2.imshow("World Stream (PC)", world_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    receiver.stop()
    cv2.destroyAllWindows()