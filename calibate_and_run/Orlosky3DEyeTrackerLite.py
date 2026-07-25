import cv2
import numpy as np
import tkinter as tk
import os
from tkinter import filedialog
import time

PI_STREAM = "http://192.168.50.149:8000/video_feed"


# Crop the image to maintain a specific aspect ratio (width:height) before resizing.
def crop_to_aspect_ratio(image, width=640, height=480):
    
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

# Apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# Finds a square area of dark pixels in the image
def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

# Mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

# Returns the largest contour that is not extremely long or tall
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    return [largest_contour] if largest_contour is not None else []

# Process frames for pupil detection
def process_frames(thresholded_image_medium, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = ((0, 0), (0, 0), 0)
    if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
        ellipse = cv2.fitEllipse(reduced_contours[0])
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 0), -1)
        final_rotated_rect = ellipse

    cv2.imshow("Frame with Ellipse", frame)

    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)

    return final_rotated_rect

# Process a single frame for pupil detection
def process_frame(frame):

    frame = cv2.flip(frame, -1)
    frame = cv2.flip(frame, 0)

    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    return process_frames(thresholded_image_medium, frame, gray_frame, darkest_point, False, False)

def getPupilAndPicture(frame):
    """
    Detects the pupil in a frame.

    Args:
        frame: BGR image from OpenCV.

    Returns:
        ((center_x, center_y), annotated_frame)

        If no pupil is found:
            (None, annotated_frame)
    """
    start_time = time.perf_counter()
    
    frame = cv2.flip(frame, -1)
    frame = cv2.flip(frame, 0)

    annotated_frame = crop_to_aspect_ratio(frame.copy())

    darkest_point = get_darkest_area(annotated_frame)
    gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    thresholded = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    thresholded = mask_outside_square(thresholded, darkest_point, 250)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    if len(contours) > 0 and len(contours[0]) > 5:
        ellipse = cv2.fitEllipse(contours[0])

        # Draw ellipse
        cv2.ellipse(annotated_frame, ellipse, (0, 255, 0), 2)

        # Draw center
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(annotated_frame, (center_x, center_y),4,(0, 255, 255),-1)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        #print(f"Function took {execution_time:.6f} seconds to complete.")
        return (center_x, center_y), annotated_frame

    return None, annotated_frame

# Process video frames for pupil detection
def process_video(video_path, input_method):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))

    cap = cv2.VideoCapture(video_path) if input_method == 1 else cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    debug_mode_on = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Prompt user to select a video file
def select_video():
# Process video from the selected camera
    #global selected_camera
    global capture_stuck_ellipses
    global capture_frame_counter
    global stuck_ellipses

    #cam_index = int(selected_camera.get())

    #cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    cap = cv2.VideoCapture(PI_STREAM)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("ORiginal Frame", frame)

        process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key in (ord('f'), ord('F')):
            print("error, invalid")
            #toggle_eye_sphere_adjustment()
        elif key == ord('e'):
            capture_stuck_ellipses = not capture_stuck_ellipses
            capture_frame_counter = 0
            print(f"Ellipse capture mode: {'ON' if capture_stuck_ellipses else 'OFF'}")
        elif key == ord('c'):
            stuck_ellipses.clear()
            print("Cleared stuck ellipses.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    select_video()