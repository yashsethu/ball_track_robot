import time
import RPi.GPIO as GPIO
import adafruit_dotstar as dotstar
import colorsys
import board

# Set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Select
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Left
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Up
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Right
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Down

# Initialize the DotStar LED
dots = dotstar.DotStar(board.D6, board.D5, 3, brightness=0.03)

# Define LED colors
COLOR_SELECT = (255, 0, 0)  # Red
COLOR_LEFT = (0, 255, 0)  # Green
COLOR_UP = (0, 0, 255)  # Blue
COLOR_RIGHT = (255, 255, 0)  # Yellow
COLOR_DOWN = (255, 0, 255)  # Magenta


# Function to smoothly change the LED color
def smooth_rainbow(offset):
    hue = offset % 360 / 360.0  # Convert the hue to a value between 0 and 1
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert the HSV color to RGB
    # Convert the RGB values to a scale of 0-255 and set the LED color
    dots.fill(tuple(int(c * 255) for c in rgb))


# Main loop
offset = 0
while True:
    if GPIO.input(16) == 0:
        print("Select:", GPIO.input(16))
    elif GPIO.input(22) == 0:
        print("Left:", GPIO.input(22))
    elif GPIO.input(23) == 0:
        print("Up:", GPIO.input(17))
    elif GPIO.input(24) == 0:
        print("Right:", GPIO.input(23))
    elif GPIO.input(27) == 0:
        print("Down:", GPIO.input(27))

    if GPIO.input(17) == 0:
        print("End", GPIO.input(27))
        GPIO.cleanup()
        break

    smooth_rainbow(offset)
    offset += 1
    time.sleep(0.01)  # Adjust this value to change the speed of the color transition


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
