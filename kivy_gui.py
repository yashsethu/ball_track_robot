import psutil
import requests
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.tooltip import ToolTip

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
    Label:
        id: temperature_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Label:
        id: humidity_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Label:
        id: weather_forecast_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Label:
        id: air_pollution_index_label
        color: 0, 1, 0, 1
        size_hint: None, None
        size: self.texture_size
        pos_hint: {"center_x": 0.5}
    Label:
        id: us_debt_label
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
    ProgressBar:
        id: temperature_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}
    ProgressBar:
        id: humidity_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}
    ProgressBar:
        id: weather_forecast_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}
    ProgressBar:
        id: air_pollution_index_progress
        value: 0
        max: 100
        size_hint: None, None
        size: 200, 20
        pos_hint: {"center_x": 0.5}
    ProgressBar:
        id: us_debt_progress
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
        self.temperature_label = Label(text="Temperature: ", color=(0, 1, 0, 1))
        self.humidity_label = Label(text="Humidity: ", color=(0, 1, 0, 1))
        self.weather_forecast_label = Label(
            text="Weather Forecast: ", color=(0, 1, 0, 1)
        )
        self.air_pollution_index_label = Label(
            text="Air Pollution Index: ", color=(0, 1, 0, 1)
        )
        self.us_debt_label = Label(text="US Debt: ", color=(0, 1, 0, 1))

        # Create the progress bars for CPU, memory, and disk usage
        self.cpu_progress = ProgressBar(value=0, max=100)
        self.memory_progress = ProgressBar(value=0, max=100)
        self.disk_progress = ProgressBar(value=0, max=100)
        self.temperature_progress = ProgressBar(value=0, max=100)
        self.humidity_progress = ProgressBar(value=0, max=100)
        self.weather_forecast_progress = ProgressBar(value=0, max=100)
        self.air_pollution_index_progress = ProgressBar(value=0, max=100)
        self.us_debt_progress = ProgressBar(value=0, max=100)

        # Create the refresh button
        self.refresh_button = Button(text="Refresh")

        # Add the widgets to the layout
        self.add_widget(self.cpu_label)
        self.add_widget(self.cpu_progress)
        self.add_widget(self.memory_label)
        self.add_widget(self.memory_progress)
        self.add_widget(self.disk_label)
        self.add_widget(self.disk_progress)
        self.add_widget(self.temperature_label)
        self.add_widget(self.temperature_progress)
        self.add_widget(self.humidity_label)
        self.add_widget(self.humidity_progress)
        self.add_widget(self.weather_forecast_label)
        self.add_widget(self.weather_forecast_progress)
        self.add_widget(self.air_pollution_index_label)
        self.add_widget(self.air_pollution_index_progress)
        self.add_widget(self.us_debt_label)
        self.add_widget(self.us_debt_progress)
        self.add_widget(self.refresh_button)

        # Schedule the update_performance_data function to be called every second
        Clock.schedule_interval(self.update_performance_data, 1)

    def update_performance_data(self, dt=None):
        # Fetch data from APIs
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent

        # Fetch temperature from weather API
        weather_api_url = "https://api.weather.com/..."
        response = requests.get(weather_api_url)
        weather_data = response.json()
        temperature = weather_data.get("temperature")

        # Fetch humidity from weather API
        humidity = weather_data.get("humidity")

        # Fetch weather forecast from weather API
        weather_forecast = weather_data.get("forecast")

        # Fetch air pollution index from air quality API
        air_quality_api_url = "https://api.airquality.com/..."
        response = requests.get(air_quality_api_url)
        air_quality_data = response.json()
        air_pollution_index = air_quality_data.get("air_pollution_index")

        # Fetch US debt value from economic API
        economic_api_url = "https://api.economic.com/..."
        response = requests.get(economic_api_url)
        economic_data = response.json()
        us_debt = economic_data.get("us_debt")

        # Update labels and progress bars
        self.cpu_label.text = f"CPU Usage: {cpu_usage:.2f}%"
        self.memory_label.text = f"Memory Usage: {memory_usage:.2f}%"
        self.disk_label.text = f"Disk Usage: {disk_usage:.2f}%"
        self.temperature_label.text = f"Temperature: {temperature}°C"
        self.humidity_label.text = f"Humidity: {humidity}%"
        self.weather_forecast_label.text = f"Weather Forecast: {weather_forecast}"
        self.air_pollution_index_label.text = (
            f"Air Pollution Index: {air_pollution_index}"
        )
        self.us_debt_label.text = f"US Debt: ${us_debt}"

        self.cpu_progress.value = cpu_usage
        self.memory_progress.value = memory_usage
        self.disk_progress.value = disk_usage
        self.temperature_progress.value = temperature
        self.humidity_progress.value = humidity
        self.weather_forecast_progress.value = (
            0  # Replace with actual weather forecast progress
        )
        self.air_pollution_index_progress.value = air_pollution_index
        self.us_debt_progress.value = us_debt

    def show_tooltip(self, text):
        tooltip = ToolTip(text=text)
        tooltip.show_for_widget(self)


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
