import cv2
import numpy as np
import tensorflow as tf

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


# Load and process an image
image = cv2.imread("path/to/image.jpg")
output_image = detect_faces(image)

# Display the output image
cv2.imshow("Face Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
