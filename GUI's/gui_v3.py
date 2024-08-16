import sys
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
from PyQt5.QtCore import QTimer
import psutil


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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

        # Create a list of buttons
        self.buttons = [QPushButton(f"Button {i}", self) for i in range(10)]
        for button in self.buttons:
            control_layout.addWidget(button)

        # Create a slider for volume control
        self.volume_slider = QSlider()
        self.volume_slider.setOrientation(Qt.Horizontal)
        control_layout.addWidget(self.volume_slider)

        # Create a vertical layout for the main window
        layout = QVBoxLayout()
        layout.addLayout(performance_layout)
        layout.addLayout(control_layout)

        # Set the layout to the main window
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Keep track of the currently selected button
        self.current_button = 0
        self.buttons[self.current_button].setStyleSheet(
            "QPushButton { background-color: yellow; }"
        )

        # Start updating performance data
        self.update_performance_data()

    def update_performance_data(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        self.cpu_label.setText(f"CPU Usage: {cpu_usage}%")
        self.memory_label.setText(f"Memory Usage: {memory_usage}%")
        self.disk_label.setText(f"Disk Usage: {disk_usage}%")

        QTimer.singleShot(1000, self.update_performance_data)

    def handle_select(self, event):
        selected_item = event.direction()
        if selected_item == "up":
            self.buttons[self.current_button].setStyleSheet(
                "QPushButton { background-color: none; }"
            )
            self.current_button = (self.current_button - 1) % len(self.buttons)
            self.buttons[self.current_button].setStyleSheet(
                "QPushButton { background-color: yellow; }"
            )
        elif selected_item == "down":
            self.buttons[self.current_button].setStyleSheet(
                "QPushButton { background-color: none; }"
            )
            self.current_button = (self.current_button + 1) % len(self.buttons)
            self.buttons[self.current_button].setStyleSheet(
                "QPushButton { background-color: yellow; }"
            )
        elif selected_item == "center":
            self.buttons[self.current_button].click()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
