import cv2
from picamera2 import Picamera2
import time
import board
import adafruit_dotstar as dotstar
import colorsys
import RPi.GPIO as GPIO

# Initialize the camera
camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
camera.start()
time.sleep(2)

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


# Function to set LED color
def set_led_color(color):
    dots.fill(color)


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

    # Check GPIO inputs and set LED color accordingly
    if GPIO.input(16) == 0:
        print("Select:", GPIO.input(16))
        set_led_color(COLOR_SELECT)
    elif GPIO.input(22) == 0:
        print("Left:", GPIO.input(22))
        set_led_color(COLOR_LEFT)
    elif GPIO.input(23) == 0:
        print("Up:", GPIO.input(17))
        set_led_color(COLOR_UP)
    elif GPIO.input(24) == 0:
        print("Right:", GPIO.input(23))
        set_led_color(COLOR_RIGHT)
    elif GPIO.input(27) == 0:
        print("Down:", GPIO.input(27))
        set_led_color(COLOR_DOWN)

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
