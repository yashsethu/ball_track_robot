import pygame
import sys
import RPi.GPIO as GPIO

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib


class PerformanceWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Performance Window")

        # Set the window size
        self.set_default_size(800, 600)

        # Create a vertical box layout
        layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(layout)

        # Create the labels for CPU, memory, and disk usage
        self.cpu_label = Gtk.Label(label="CPU Usage: ")
        self.memory_label = Gtk.Label(label="Memory Usage: ")
        self.disk_label = Gtk.Label(label="Disk Usage: ")

        # Create the progress bars for CPU, memory, and disk usage
        self.cpu_progress = Gtk.ProgressBar()
        self.memory_progress = Gtk.ProgressBar()
        self.disk_progress = Gtk.ProgressBar()

        # Create the refresh button
        self.refresh_button = Gtk.Button(label="Refresh")

        # Create the chart to display historical CPU usage data
        self.cpu_chart = Gtk.DrawingArea()

        # Add the widgets to the layout
        layout.pack_start(self.cpu_label, False, False, 0)
        layout.pack_start(self.cpu_progress, False, False, 0)
        layout.pack_start(self.memory_label, False, False, 0)
        layout.pack_start(self.memory_progress, False, False, 0)
        layout.pack_start(self.disk_label, False, False, 0)
        layout.pack_start(self.disk_progress, False, False, 0)
        layout.pack_start(self.refresh_button, False, False, 0)
        layout.pack_start(self.cpu_chart, True, True, 0)

        # Schedule the update_performance_data function to be called every second
        GLib.timeout_add_seconds(1, self.update_performance_data)

        # Update the performance data initially
        self.update_performance_data()

    def update_performance_data(self):
        # Fetch data from APIs
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        # Update labels and progress bars
        self.cpu_label.set_text(f"CPU Usage: {cpu_usage:.2f}%")
        self.memory_label.set_text(f"Memory Usage: {memory_usage:.2f}%")
        self.disk_label.set_text(f"Disk Usage: {disk_usage:.2f}%")

        self.cpu_progress.set_fraction(cpu_usage / 100)
        self.memory_progress.set_fraction(memory_usage / 100)
        self.disk_progress.set_fraction(disk_usage / 100)

        # Update the CPU usage chart
        self.update_cpu_chart(cpu_usage)

        return True

    def update_cpu_chart(self, cpu_usage):
        # Get the current figure and axis
        fig = (
            self.cpu_chart.get_property("window")
            .get_property("cairo-context")
            .get_target()
            .figure
        )
        ax = fig.gca()

        # Get the existing CPU usage data
        x_data, y_data = ax.lines[0].get_data()

        # Append the new CPU usage data
        x_data = list(x_data) + [len(x_data)]
        y_data = list(y_data) + [cpu_usage]

        # Clear the existing chart
        ax.clear()

        # Plot the CPU usage data
        ax.plot(x_data, y_data, color="blue")

        # Set the chart title and labels
        ax.set_title("CPU Usage Over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("CPU Usage (%)")

        # Redraw the chart
        fig.canvas.draw()


if __name__ == "__main__":
    win = PerformanceWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


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


# Set up the game window
with pygame.display.set_mode((WIDTH, HEIGHT)) as screen:
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
