import cv2

redLower = (150, 140, 1)
redUpper = (190, 255, 255)


# Returns a mask of all colors within the red HSV space
def find_color_mask(frame):
    # Resize the frame
    resize = cv2.resize(frame, (320, 240))

    # Convert to HSV color space
    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)

    # Create a mask using the specified lower and upper red values
    mask = cv2.inRange(hsv, redLower, redUpper)

    # Erode and dilate the mask to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


# Finds the largest "blob" on the screen
def find_largest_contour(frame):
    # Find contours in the provided image
    contours, _ = cv2.findContours(
        frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        # Find the contour with the largest area
        contour = max(contours, key=cv2.contourArea)

        # Get the minimum enclosing circle and moments of the contour
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)

        # Calculate the center and area of the contour
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        area = cv2.contourArea(contour)
    else:
        # If no contours are found, set default values
        (x, y) = (0, 0)
        radius = 0
        center = (0, 0)
        area = 0

    return x, y, radius, center, area
