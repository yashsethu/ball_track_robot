from picamera2 import Picamera2
import numpy as np
import cv2
import time

# This took a really long time to find, lots of image manipulation involved to find HSV of ball
redLower = (150, 140, 1)
redUpper = (190, 255, 255)
