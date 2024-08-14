import sys
import psutil
import requests
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolTip,
)
from PyQt5.QtCore import Qt, QTimer


class PerformanceWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window size
        self.setGeometry(100, 100, 400, 600)

        # Create the central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create the labels for CPU, memory, and disk usage
        self.cpu_label = QLabel("CPU Usage: ")
        self.memory_label = QLabel("Memory Usage: ")
        self.disk_label = QLabel("Disk Usage: ")

        # Create the progress bars for CPU, memory, and disk usage
        self.cpu_progress = QProgressBar()
        self.memory_progress = QProgressBar()
        self.disk_progress = QProgressBar()

        # Create the refresh button
        self.refresh_button = QPushButton("Refresh")

        # Set tooltips for the widgets
        self.cpu_label.setToolTip("Current CPU usage")
        self.memory_label.setToolTip("Current memory usage")
        self.disk_label.setToolTip("Current disk usage")
        self.refresh_button.setToolTip("Refresh performance data")

        # Add the widgets to the layout
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_progress)
        layout.addWidget(self.memory_label)
        layout.addWidget(self.memory_progress)
        layout.addWidget(self.disk_label)
        layout.addWidget(self.disk_progress)
        layout.addWidget(self.refresh_button)

        # Set the central widget
        self.setCentralWidget(central_widget)

        # Schedule the update_performance_data function to be called every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_performance_data)
        self.timer.start(1000)

        # Update the performance data initially
        self.update_performance_data()

    def update_performance_data(self):
        # Fetch data from APIs
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        # Update labels and progress bars
        self.cpu_label.setText(f"CPU Usage: {cpu_usage:.2f}%")
        self.memory_label.setText(f"Memory Usage: {memory_usage:.2f}%")
        self.disk_label.setText(f"Disk Usage: {disk_usage:.2f}%")

        self.cpu_progress.setValue(cpu_usage)
        self.memory_progress.setValue(memory_usage)
        self.disk_progress.setValue(disk_usage)


class MyApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.main_window = PerformanceWindow()
        self.main_window.show()


if __name__ == "__main__":
    app = MyApp(sys.argv)
    sys.exit(app.exec_())
