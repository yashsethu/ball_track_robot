from picamera2 import Picamera2
import numpy as np
import cv2
import time

# This took a really long time to find, lots of image manipulation involved to find HSV of ball
redLower = (150, 140, 1)
redUpper = (190, 255, 255)


# Returns a mask of all colors within the red HSV space
def find_color_mask(frame):
    # Blur the image and convert to HSV color space
    resize = cv.resize(frame, (320, 240))
    blurred = cv2.GaussianBlur(resize, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Erode and Dilate to remove noise
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


# Finds the largest "blob" on the screen
def find_largest_contour(frame):
    # Finds contours in the provided image
    cnts, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        (x, y) = (0, 0)
        radius = 5
        center = (0, 0)

    return x, y, radius, center


while True:
    frame = picam2.capture_array()

    if frame is None:
        print("Error: Frame not captured")
        break

    mask = find_color_mask(frame)

    x, y, radius, center = find_largest_contour(mask)

    if radius > 10:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press q to break the loop and stop moving
        break

cv2.destroyAllWindows()
picam2.stop()
