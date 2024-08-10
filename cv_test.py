import cv2
import numpy as np
from picamera2 import Picamera2

# Define the lower and upper bounds of the red color in HSV space
red_lower = (150, 140, 1)
red_upper = (190, 255, 255)

# Create an instance of Picamera2
picam2 = Picamera2()


# Function to find the color mask of the ball
def find_color_mask(frame):
    # Resize the frame and convert it to HSV color space
    resized = cv2.resize(frame, (320, 240))
    blurred = cv2.GaussianBlur(resized, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask of the red color
    mask = cv2.inRange(hsv, red_lower, red_upper)

    # Perform morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


# Function to find the largest contour (ball) in the frame
def find_largest_contour(frame):
    # Find contours in the frame
    contours, _ = cv2.findContours(
        frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        M = cv2.moments(largest_contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        (x, y) = (0, 0)
        radius = 5
        center = (0, 0)

    return x, y, radius, center


# Main loop
while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    if frame is None:
        print("Error: Frame not captured")
        break

    # Find the color mask of the ball
    mask = find_color_mask(frame)

    # Find the largest contour (ball) in the frame
    x, y, radius, center = find_largest_contour(mask)

    if radius > 10:
        # Draw the circle and centroid on the frame
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Break the loop and stop moving if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
