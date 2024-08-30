import cv2
import time
import matplotlib.pyplot as plt

# Define the lower and upper bounds for the red color
redLower = (150, 140, 1)
redUpper = (190, 255, 255)

# Create a video capture object
vid = cv2.VideoCapture(0)

# Lists to store accuracy values
accuracy_values = []
probability_values = []
camera_data = []
inference_times = []

while True:
    ret, frame = vid.read()

    if not ret:
        print("Error: Frame not captured")
        break

    # Resize and blur the image, then convert to HSV color space
    resize = cv2.resize(frame, (320, 240))
    blurred = cv2.GaussianBlur(resize, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Erode and dilate to remove noise
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the provided image
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        (x, y) = (0, 0)
        radius = 5
        center = (0, 0)

    if radius > 10:
        # Draw the circle and centroid on the frame
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    cv2.imshow("Tracking", frame)

    # Calculate accuracy and append to the list
    accuracy = radius / 10.0
    accuracy_values.append(accuracy)

    # Calculate probability and append to the list
    probability = 1 - (accuracy / 10.0)
    probability_values.append(probability)

    # Append camera data to the list
    camera_data.append(frame)

    # Measure inference time
    start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press q to break the loop and stop moving
        break

    inference_time = time.time() - start_time
    inference_times.append(inference_time)

# Plot the accuracy graph
plt.plot(accuracy_values)
plt.xlabel("Frame")
plt.ylabel("Accuracy")
plt.title("Accuracy over Time")
plt.show()

# Plot the probability graph
plt.plot(probability_values)
plt.xlabel("Frame")
plt.ylabel("Probability")
plt.title("Probability over Time")
plt.show()

# Plot the inference time graph
plt.plot(inference_times)
plt.xlabel("Frame")
plt.ylabel("Inference Time (s)")
plt.title("Inference Time over Time")
plt.show()

# Display camera data
for frame in camera_data:
    cv2.imshow("Camera Data", frame)
    if cv2.waitKey(1) & 0xFF == ord(
        "q"
    ):  # Press q to break the loop and stop displaying camera data
        break

# Release the video capture object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
cv2.destroyAllWindows()
