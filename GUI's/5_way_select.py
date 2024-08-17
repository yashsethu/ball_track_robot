import board
import adafruit_dotstar as dotstar
import time

# Import the necessary libraries
import RPi.GPIO as GPIO

# Set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(6, GPIO.OUT)

# Define the directions
directions = {17: "Up", 27: "Down", 22: "Left", 23: "Right", 24: "Center"}

# Initialize the DotStar LED
dot = dotstar.DotStar(board.D5, board.D6, 1)


# Function to detect the direction
def detect_direction(channel):
    if channel in directions:
        direction = directions[channel]
        print("Direction detected:", direction)
        if direction == "Up":
            dot[0] = (255, 0, 0)  # Set LED color to red
        elif direction == "Down":
            dot[0] = (0, 255, 0)  # Set LED color to green
        elif direction == "Left":
            dot[0] = (0, 0, 255)  # Set LED color to blue
        elif direction == "Right":
            dot[0] = (255, 255, 0)  # Set LED color to yellow
        elif direction == "Center":
            dot[0] = (255, 255, 255)  # Set LED color to white


# Function to blink the LED
def blink_led():
    dot[0] = (255, 0, 0)  # Set LED color to red
    time.sleep(0.5)
    dot[0] = (0, 0, 0)  # Turn off LED
    time.sleep(0.5)


# Add event detection for each GPIO pin
for pin in directions.keys():
    GPIO.add_event_detect(pin, GPIO.FALLING, callback=detect_direction, bouncetime=200)

# Create the GUI
root = Tk()
direction_label = Label(root, text="")
color_label = Label(root, text="")
direction_label.pack()
color_label.pack()


# Function to detect the direction
def detect_direction(channel):
    if channel in directions:
        direction = directions[channel]
        print("Direction detected:", direction)
        direction_label.config(text="Direction detected: " + direction)
        if direction == "Up":
            dot[0] = (255, 0, 0)  # Set LED color to red
            color_label.config(text="LED Color: Red", fg="red")
        elif direction == "Down":
            dot[0] = (0, 255, 0)  # Set LED color to green
            color_label.config(text="LED Color: Green", fg="green")
        elif direction == "Left":
            dot[0] = (0, 0, 255)  # Set LED color to blue
            color_label.config(text="LED Color: Blue", fg="blue")
        elif direction == "Right":
            dot[0] = (255, 255, 0)  # Set LED color to yellow
            color_label.config(text="LED Color: Yellow", fg="yellow")


# Add event detection for each GPIO pin
for pin in directions.keys():
    GPIO.add_event_detect(pin, GPIO.FALLING, callback=detect_direction, bouncetime=200)

# Main loop
try:
    root.mainloop()
except KeyboardInterrupt:
    GPIO.cleanup()
