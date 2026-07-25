# Plot eye tracking from pi eye cam
import cv2
from old.OrloskyPupilDetector import process_frame

cap = cv2.VideoCapture("http://192.168.50.149:8000/video_feed")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    pupil_ellipse = process_frame(frame)  # returns rotated rectangle / ellipse
    if pupil_ellipse:
        center = tuple(map(int, pupil_ellipse[0]))
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

    cv2.imshow("Eye + Pupil", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
