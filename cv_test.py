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
