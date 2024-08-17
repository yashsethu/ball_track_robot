import time
import RPi.GPIO as GPIO
import adafruit_dotstar as dotstar
import colorsys
import board

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


# Function to smoothly change the LED color
def smooth_rainbow(offset):
    hue = offset % 360 / 360.0  # Convert the hue to a value between 0 and 1
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert the HSV color to RGB
    # Convert the RGB values to a scale of 0-255 and set the LED color
    dots.fill(tuple(int(c * 255) for c in rgb))


# Main loop
offset = 0
while True:
    if GPIO.input(16) == 0:
        print("Select:", GPIO.input(16))
    elif GPIO.input(22) == 0:
        print("Left:", GPIO.input(22))
    elif GPIO.input(23) == 0:
        print("Up:", GPIO.input(17))
    elif GPIO.input(24) == 0:
        print("Right:", GPIO.input(23))
    elif GPIO.input(27) == 0:
        print("Down:", GPIO.input(27))

    if GPIO.input(17) == 0:
        print("End", GPIO.input(27))
        GPIO.cleanup()
        break

    smooth_rainbow(offset)
    offset += 1
    time.sleep(0.01)  # Adjust this value to change the speed of the color transition
