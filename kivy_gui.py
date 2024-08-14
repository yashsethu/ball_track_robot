import psutil
import gi
import matplotlib.pyplot as plt

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
        x_data = ax.lines[0].get_xdata()
        y_data = ax.lines[0].get_ydata()

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
