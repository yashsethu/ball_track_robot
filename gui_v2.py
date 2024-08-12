import tkinter as tk
import psutil
import requests
from adafruit_circuitpython_onboard import OnboardFiveWay


def update_performance_data():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent

    cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
    memory_label.config(text=f"Memory Usage: {memory_usage}%")
    disk_label.config(text=f"Disk Usage: {disk_usage}%")

    root.after(1000, update_performance_data)


def handle_select(event):
    selected_item = event.direction
    if selected_item == "up":
        # Handle up direction logic
        print("Up direction selected")
    elif selected_item == "down":
        # Handle down direction logic
        print("Down direction selected")
    elif selected_item == "left":
        # Handle left direction logic
        print("Left direction selected")
    elif selected_item == "right":
        # Handle right direction logic
        print("Right direction selected")
    elif selected_item == "center":
        # Handle center direction logic
        print("Center direction selected")
    else:
        # Handle unknown direction logic
        print("Unknown direction selected")


def get_weather_data():
    # Make a request to the weather API to get the weather data
    response = requests.get(
        "https://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q=YOUR_LOCATION"
    )
    data = response.json()

    # Extract the required information from the response
    time_of_day = data["current"]["last_updated"]
    weather = data["current"]["condition"]["text"]
    air_pollution_index = data["current"]["air_quality"]["us-epa-index"]

    # Update the labels with the fetched data
    time_label.config(text=f"Time of Day: {time_of_day}")
    weather_label.config(text=f"Weather: {weather}")
    air_pollution_label.config(text=f"Air Pollution Index: {air_pollution_index}")


def toggle_light():
    # Toggle the state of the light
    if light_button.config("text")[-1] == "Turn On":
        light_button.config(text="Turn Off", bg="green")
        # Code to turn on the light
        print("Light turned on")
    else:
        light_button.config(text="Turn On", bg="red")
        # Code to turn off the light
        print("Light turned off")


root = tk.Tk()
root.title("Performance Data")

cpu_label = tk.Label(root, text="CPU Usage: ")
cpu_label.pack()

memory_label = tk.Label(root, text="Memory Usage: ")
memory_label.pack()

disk_label = tk.Label(root, text="Disk Usage: ")
disk_label.pack()

time_label = tk.Label(root, text="Time of Day: ")
time_label.pack()

weather_label = tk.Label(root, text="Weather: ")
weather_label.pack()

air_pollution_label = tk.Label(root, text="Air Pollution Index: ")
air_pollution_label.pack()

update_performance_data()

onboard = OnboardFiveWay()
onboard.direction_any = handle_select

# Fetch weather data initially and then update it every 5 minutes
get_weather_data()
root.after(300000, get_weather_data)

light_button = tk.Button(root, text="Turn On", command=toggle_light)
light_button.pack()

root.mainloop()
