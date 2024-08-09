import cv2
import time

# Define the lower and upper bounds for the red color
redLower = (150, 140, 1)
redUpper = (190, 255, 255)

# Create a video capture object
vid = cv2.VideoCapture(0)

# Configure the PiCamera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()
time.sleep(2)


# Returns a mask of all colors within the red HSV space
def find_color_mask(frame):
    # Resize and blur the image, then convert to HSV color space
    resize = cv2.resize(frame, (320, 240))
    blurred = cv2.GaussianBlur(resize, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Erode and dilate to remove noise
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


# Finds the largest contour on the screen
def find_largest_contour(frame):
    # Find contours in the provided image
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
        # Draw the circle and centroid on the frame
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press q to break the loop and stop moving
        break

cv2.destroyAllWindows()
picam2.stop()

while True:
    # Capture the video frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # Break the loop if the 'q' button is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
