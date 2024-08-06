import numpy as np
from scipy.signal import savgol_filter
import cv2
def draw_two_parallel_lines(frame):
    for i in range(frame.shape[0]):
        for j in range(-1, 1):  # Making the lines two pixels wide
            if 0 <= i+j < frame.shape[0]:
                # First small line from top-left to bottom-right
                frame[i, i+j] = 192  # Light grey color
                # Second small line parallel to the first one
                if i + 10 < frame.shape[0]:
                    frame[i + 10, i+j] = 192  # Light grey color
    return frame
# Function to draw a wider light grey line from one corner to another
def draw_wide_diagonal_line(frame):
    
    for i in range(frame.shape[0]):
        for j in range(-2, 3):  # Making the line wider
            if 0 <= i+j < frame.shape[0]:
                frame[i, i+j] = 192  # Light grey color
    return frame
def detect_center(or_pre):
    s = []
    for i in range(or_pre.shape[-1]):
        # Load the image
        l = (or_pre[...,i]/or_pre[...,i].max())*255
        image = l.astype(np.uint8)

        # Threshold the image to binary
        _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area to exclude the catheter and its shadow
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

        # Combine all points from the filtered contours
        all_points = np.vstack(filtered_contours)

        # Find the convex hull of the combined points
        hull = cv2.convexHull(all_points)

        # Draw the convex hull on a black background
        output = np.zeros_like(image)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)  # Ensure output is in correct format

        # Draw the convex hull in white color on the output image
        cv2.drawContours(output, [hull], -1, (255, 255, 255), 1)

        # Convert output back to grayscale for display
        output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # Calculate the moments of the convex hull
        moments = cv2.moments(hull)

        # Calculate the centroid (center) of the convex hull
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
        else:
            cX, cY = 0, 0
        s.append((cX, cY))
    s= np.array(s)
    s = np.copy(s)


    for dim in range(2):  # For both x and y dimensions
        s[:, dim] = savgol_filter(s[:, dim], 31, 2)
    return s