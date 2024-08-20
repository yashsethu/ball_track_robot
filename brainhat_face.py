import cv2
from picamera2 import Picamera2
import time

# Initialize the camera
camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
camera.start()
time.sleep(2)

# Allow the camera to warm up
time.sleep(0.1)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Capture frames from the camera
while True:
    # Retrieve the image array from the frame
    image = camera.capture_array()

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw rectangles around the detected faces
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create a named window with fullscreen flag
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Face Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    # Display the output
    cv2.imshow("Face Detection", image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera resources
camera.close()
cv2.destroyAllWindows()
