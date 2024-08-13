from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QSlider,
    QHBoxLayout,
)
from PyQt5.QtCore import QTimer, Qt
import psutil
from PyQt5.QtGui import QFont
import sys


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
