import cv2

redLower = (150, 140, 1)
redUpper = (190, 255, 255)


def find_color_mask(frame):
    resized_frame = cv2.resize(frame, (320, 240))
    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, redLower, redUpper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def find_largest_contour(frame):
    contours, _ = cv2.findContours(
        frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        area = cv2.contourArea(contour)
    else:
        x, y = 0, 0
        radius = 0
        center = (0, 0)
        area = 0
    return x, y, radius, center, area
