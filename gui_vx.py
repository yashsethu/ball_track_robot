import psutil
import tkinter as tk
import matplotlib.pyplot as plt


class PerformanceWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the window size
        self.geometry("800x600")

        # Create a vertical box layout
        layout = tk.Frame(self)
        layout.pack(pady=10)

        # Create the labels for CPU, memory, and disk usage
        self.cpu_label = tk.Label(layout, text="CPU Usage: ")
        self.memory_label = tk.Label(layout, text="Memory Usage: ")
        self.disk_label = tk.Label(layout, text="Disk Usage: ")

        # Create the progress bars for CPU, memory, and disk usage
        self.cpu_progress = tk.Progressbar(layout, length=200)
        self.memory_progress = tk.Progressbar(layout, length=200)
        self.disk_progress = tk.Progressbar(layout, length=200)

        # Create the refresh button
        self.refresh_button = tk.Button(
            layout, text="Refresh", command=self.update_performance_data
        )

        # Create the chart to display historical CPU usage data
        self.cpu_chart = tk.Canvas(self, width=600, height=400)

        # Add the widgets to the layout
        self.cpu_label.pack()
        self.cpu_progress.pack()
        self.memory_label.pack()
        self.memory_progress.pack()
        self.disk_label.pack()
        self.disk_progress.pack()
        self.refresh_button.pack()
        self.cpu_chart.pack()

        # Update the performance data initially
        self.update_performance_data()

    def update_performance_data(self):
        # Fetch data from APIs
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        # Update labels and progress bars
        self.cpu_label.config(text=f"CPU Usage: {cpu_usage:.2f}%")
        self.memory_label.config(text=f"Memory Usage: {memory_usage:.2f}%")
        self.disk_label.config(text=f"Disk Usage: {disk_usage:.2f}%")

        self.cpu_progress.config(value=cpu_usage)
        self.memory_progress.config(value=memory_usage)
        self.disk_progress.config(value=disk_usage)

        # Update the CPU usage chart
        self.update_cpu_chart(cpu_usage)

    def update_cpu_chart(self, cpu_usage):
        # Clear the existing chart
        self.cpu_chart.delete("all")

        # Get the chart dimensions
        chart_width = self.cpu_chart.winfo_width()
        chart_height = self.cpu_chart.winfo_height()

        # Calculate the coordinates for the CPU usage data point
        x = chart_width / 2
        y = chart_height - (cpu_usage / 100) * chart_height

        # Draw a line representing the CPU usage data point
        self.cpu_chart.create_line(x, chart_height, x, y, fill="blue")

        # Draw the chart title and labels
        self.cpu_chart.create_text(
            chart_width / 2,
            20,
            text="CPU Usage Over Time",
            font=("Arial", 16),
            fill="black",
        )
        self.cpu_chart.create_text(
            chart_width / 2,
            chart_height - 20,
            text="Time (seconds)",
            font=("Arial", 12),
            fill="black",
        )
        self.cpu_chart.create_text(
            20, chart_height / 2, text="CPU Usage (%)", font=("Arial", 12), fill="black"
        )


if __name__ == "__main__":
    win = PerformanceWindow()
    win.mainloop()
