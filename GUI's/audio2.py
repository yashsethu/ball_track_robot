import pygame
import sys
import cv2
import time
from picamera import PiCamera
import RPi.GPIO as GPIO

# Set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Game loop
while True:
    # Event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Check if the up button is pressed
    if GPIO.input(16) == GPIO.LOW:
        print("Up button pressed")

    # Draw everything
    screen.fill((255, 255, 255))
    pygame.display.flip()

# Camera preview
picam = PiCamera()
picam.start_preview()
time.sleep(2)
picam.stop_preview()

# OpenCV frame capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured")
        break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

# Flask app
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
