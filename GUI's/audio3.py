import speech_recognition as sr
import psutil
from tkinter import Label, Button
from gpiozero import Button as GPIOButton
from flask import Flask, render_template
import pygame
import sys
import cv2
import time
from picamera import PiCamera
import RPi.GPIO as GPIO

app = Flask(__name__)

# Set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
DINO_HEIGHT = 50
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 50
DINO_Y = HEIGHT - DINO_HEIGHT
OBSTACLE_X = WIDTH
FRAME_RATE = 30

# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the Dino and obstacle
dino = pygame.image.load("dino.png").get_rect(midleft=(50, DINO_Y))
obstacle = pygame.image.load("obstacle.png").get_rect(midright=(OBSTACLE_X, DINO_Y))

# Set up some game variables
jump = False
jump_height = 10
gravity = 1
obstacle_speed = 5

# Set up dictionaries for game settings, images, and colors
settings = {"jump_height": 10, "gravity": 1, "obstacle_speed": 5}

images = {
    "dino": pygame.image.load("dino.png"),
    "obstacle": pygame.image.load("obstacle.png"),
}

colors = {"white": (255, 255, 255)}


def game_over():
    print("Game Over!")
    pygame.quit()
    sys.exit()


picam2 = PiCamera()
picam2.start_preview()
time.sleep(2)

while True:
    frame = picam2.capture()

    if frame is None:
        print("Error: Frame not captured")
        break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop_preview()

# Game loop
while True:
    # Event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Check if the up button is pressed
    if GPIO.input(16) == GPIO.LOW and dino.y == DINO_Y:
        jump = True

    # Make the Dino jump
    if jump:
        dino.y -= settings["jump_height"]
        settings["jump_height"] -= settings["gravity"]
        if dino.y >= DINO_Y:
            dino.y = DINO_Y
            jump = False
            settings["jump_height"] = 10

    # Move the obstacle
    obstacle.x -= settings["obstacle_speed"]
    if obstacle.x < 0:
        obstacle.x = OBSTACLE_X

    # Check for collision
    if dino.colliderect(obstacle):
        game_over()

    # Draw everything
    screen.fill(colors["white"])
    screen.blit(images["dino"], dino)
    screen.blit(images["obstacle"], obstacle)
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(FRAME_RATE)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title and default size
        self.setWindowTitle("Modern GUI")
        self.resize(400, 200)

        # Set a custom font
        font = QFont("Arial", 12)
        self.setFont(font)

        # Create the labels for CPU, memory, and disk usage
        self.cpu_label = QLabel(self)
        self.memory_label = QLabel(self)
        self.disk_label = QLabel(self)

        # Create a vertical layout for performance data
        performance_layout = QVBoxLayout()
        performance_layout.addWidget(self.cpu_label)
        performance_layout.addWidget(self.memory_label)
        performance_layout.addWidget(self.disk_label)

        # Create a horizontal layout for buttons and sliders
        control_layout = QHBoxLayout()

        # Create a slider for volume control
        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.valueChanged.connect(self.handle_volume_change)
        control_layout.addWidget(volume_slider)

        # Create a button to toggle a feature
        self.feature_button = QPushButton("Toggle Feature")
        self.feature_button.clicked.connect(self.toggle_feature)
        control_layout.addWidget(self.feature_button)

        # Add the control layout to the performance layout
        performance_layout.addLayout(control_layout)

        # Create a central widget to hold the layouts
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set the layout of the central widget
        central_widget.setLayout(performance_layout)

        # Set the window's style sheet
        self.setStyleSheet(
            """
            QMainWindow { background-color: #333; color: #fff; }
            QLabel { font-size: 18px; }
        """
        )

        # Create a timer to update the performance data every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_performance_data)
        self.timer.start(1000)

        # Initialize feature state
        self.feature_enabled = False

    def update_performance_data(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        self.cpu_label.setText(f"CPU Usage: {cpu_usage}%")
        self.memory_label.setText(f"Memory Usage: {memory_usage}%")
        self.disk_label.setText(f"Disk Usage: {disk_usage}%")

    def handle_volume_change(self, value):
        print(f"Volume level: {value}")

    def toggle_feature(self):
        self.feature_enabled = not self.feature_enabled
        if self.feature_enabled:
            self.feature_button.setText("Feature Enabled")
        else:
            self.feature_button.setText("Feature Disabled")


@app.route("/")
def index():
    return render_template("index.html")


# ... existing code ...

if __name__ == "__main__":
    app.run()
