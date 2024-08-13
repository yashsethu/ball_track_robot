import psutil
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder

# Set the Kivy default window size and theme
Window.size = (400, 600)
Builder.load_string(
    """
<PerformanceTab>:
    background_color: 0.2, 0.2, 0.2, 1
    Label:
        text: "CPU Usage: "
        color: 1, 1, 1, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
        on_touch_down: self.show_tooltip("CPU usage represents the percentage of CPU resources being utilized.")
    Label:
        text: "Memory Usage: "
        color: 1, 1, 1, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
        on_touch_down: self.show_tooltip("Memory usage represents the percentage of RAM being utilized.")
    Label:
        text: "Disk Usage: "
        color: 1, 1, 1, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
        on_touch_down: self.show_tooltip("Disk usage represents the percentage of disk space being utilized.")
    Label:
        id: cpu_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Label:
        id: memory_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Label:
        id: disk_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Button:
        text: "Refresh"
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
        on_release: root.update_performance_data()
    ProgressBar:
        id: cpu_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}
    ProgressBar:
        id: memory_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}
    ProgressBar:
        id: disk_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}

<MainWindow>:
    tab_pos: "top_left"
    canvas.before:
        Color:
            rgba: 0.1, 0.1, 0.1, 1
        Rectangle:
            pos: self.pos
            size: self.size
"""
)


class PerformanceTab(TabbedPanelItem):
    def __init__(self, **kwargs):
        super(PerformanceTab, self).__init__(**kwargs)
        self.orientation = "vertical"

        # Create the labels for CPU, memory, and disk usage
        self.cpu_label = Label(text="CPU Usage: ", color=(0, 1, 0, 1))
        self.memory_label = Label(text="Memory Usage: ", color=(0, 1, 0, 1))
        self.disk_label = Label(text="Disk Usage: ", color=(0, 1, 0, 1))

        # Create the progress bars for CPU, memory, and disk usage
        self.cpu_progress = ProgressBar(value=0, max=100)
        self.memory_progress = ProgressBar(value=0, max=100)
        self.disk_progress = ProgressBar(value=0, max=100)

        # Create the refresh button
        self.refresh_button = Button(text="Refresh")

        # Add the widgets to the layout
        self.add_widget(self.cpu_label)
        self.add_widget(self.cpu_progress)
        self.add_widget(self.memory_label)
        self.add_widget(self.memory_progress)
        self.add_widget(self.disk_label)
        self.add_widget(self.disk_progress)
        self.add_widget(self.refresh_button)

        # Schedule the update_performance_data function to be called every second
        Clock.schedule_interval(self.update_performance_data, 1)

    def update_performance_data(self, dt=None):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        self.cpu_label.text = f"CPU Usage: {cpu_usage:.2f}%"
        self.memory_label.text = f"Memory Usage: {memory_usage:.2f}%"
        self.disk_label.text = f"Disk Usage: {disk_usage:.2f}%"

        self.cpu_progress.value = cpu_usage
        self.memory_progress.value = memory_usage
        self.disk_progress.value = disk_usage


class MainWindow(TabbedPanel):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        self.tab_pos = "top_left"

        # Create the performance tab
        performance_tab = PerformanceTab(text="Performance")
        self.add_widget(performance_tab)


class MyApp(App):
    def build(self):
        return MainWindow()


if __name__ == "__main__":
    MyApp().run()
