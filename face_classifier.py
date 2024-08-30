import cv2
from picamera2 import Picamera2
import time
import board
import adafruit_dotstar as dotstar
import colorsys
import time
import os
from picamera2 import Picamera2
import board
import adafruit_dotstar as dotstar
import colorsys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import RPi.GPIO as GPIO

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


# Function to generate data from webcam
def generate_data_from_webcam(num_samples):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Capture the webcam feed
    cap = cv2.VideoCapture(0)

    # Loop to capture frames and save as images
    for i in range(num_samples):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the frame
        frame = cv2.resize(frame, (200, 200))
        frame = frame / 255.0

        # Save the frame as an image
        filename = f"datasets/sample_{i}.jpg"
        cv2.imwrite(filename, frame)

        # Display the frame
        cv2.imshow("frame", frame)

        # Wait for user confirmation to proceed to the next frame
        cv2.waitKey(0)

        # Delay between capturing frames
        time.sleep(0.1)

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


# Generate data from webcam and save to /datasets directory
generate_data_from_webcam(10)

# Rest of the code...

# Load and preprocess the dataset
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Split the first dataset into training and testing sets
train_data_1, test_data_1 = train_test_split(train_generator_1, test_size=0.2)

# Split the second dataset into training and testing sets
train_data_2, test_data_2 = train_test_split(train_generator_2, test_size=0.2)

# Split the filtered second dataset into training and testing sets
train_data_2_filtered, test_data_2_filtered = train_test_split(
    train_generator_2_filtered, test_size=0.2
)


# Lower the resolution, resize the image, and change the colors
def preprocess_image(image):
    # Lower the resolution
    image = cv2.resize(image, (100, 100))

    # Change the colors
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


# Set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Select
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Left
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Up
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Right
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Down

# Initialize the DotStar LED
dots = dotstar.DotStar(board.D6, board.D5, 3, brightness=0.03)

# Define LED colors
COLOR_SELECT = (255, 0, 0)  # Red
COLOR_LEFT = (0, 255, 0)  # Green
COLOR_UP = (0, 0, 255)  # Blue
COLOR_RIGHT = (255, 255, 0)  # Yellow
COLOR_DOWN = (255, 0, 255)  # Magenta

# Define additional LED colors
COLOR_1 = (255, 255, 255)  # White
COLOR_2 = (255, 165, 0)  # Orange


# Function to smoothly change the LED color
def smooth_rainbow(offset):
    hue = offset % 360 / 360.0  # Convert the hue to a value between 0 and 1
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert the HSV color to RGB
    # Convert the RGB values to a scale of 0-255 and set the LED color
    dots.fill(tuple(int(c * 255) for c in rgb))


# Function to set LED color to COLOR_1
def set_color_1():
    dots.fill(COLOR_1)


# Function to set LED color to COLOR_2
def set_color_2():
    dots.fill(COLOR_2)


# Main loop
offset = 0

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

    if GPIO.input(16) == 0:
        print("Select:", GPIO.input(16))
        dots.fill(COLOR_SELECT)
    elif GPIO.input(22) == 0:
        print("Left:", GPIO.input(22))
        dots.fill(COLOR_LEFT)
    elif GPIO.input(23) == 0:
        print("Up:", GPIO.input(17))
        dots.fill(COLOR_UP)
    elif GPIO.input(24) == 0:
        print("Right:", GPIO.input(23))
        dots.fill(COLOR_RIGHT)
    elif GPIO.input(27) == 0:
        print("Down:", GPIO.input(27))
        dots.fill(COLOR_DOWN)

    if GPIO.input(17) == 0:
        print("End", GPIO.input(27))
        GPIO.cleanup()
        break

    smooth_rainbow(offset)
    offset += 1
    time.sleep(0.01)  # Adjust this value to change the speed of the color transition

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera resources
camera.close()
cv2.destroyAllWindows()
