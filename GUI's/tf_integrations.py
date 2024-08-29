import tkinter as tk
import psutil
from tkinter import Label, Button
from gpiozero import Button as GPIOButton
import tflite_runtime.interpreter as tflite


class BrainCraftHATGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("BrainCraft HAT GUI")
        self.window.option_add("*Font", "Helvetica 10")  # Set the font size

        # Get screen width and height
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Set window size to screen size
        self.window.geometry(f"{screen_width}x{screen_height}+0+0")

        self.window.bind(
            "<Escape>", lambda event: self.window.destroy()
        )  # Close window with escape key

        self.cpu_temp_label = Label(self.window, text="CPU Temperature: ")
        self.cpu_temp_label.pack()

        self.cpu_usage_label = Label(self.window, text="CPU Usage: ")
        self.cpu_usage_label.pack()

        self.refresh_button = Button(
            self.window, text="Refresh", command=self.update_labels
        )
        self.refresh_button.pack()

        self.window.bind("<KeyPress-Return>", self.handle_select)

        self.update_labels()

    def update_labels(self):
        cpu_temp = psutil.sensors_temperatures()["cpu_thermal"][0].current
        cpu_usage = psutil.cpu_percent()
        self.cpu_temp_label.config(text=f"CPU Temperature: {cpu_temp}°C")
        self.cpu_usage_label.config(text=f"CPU Usage: {cpu_usage}%")

    def handle_select(self, event):
        direction = event.keysym
        if direction == "Up":
            self.handle_up()
        elif direction == "Down":
            self.handle_down()
        elif direction == "Left":
            self.handle_left()
        elif direction == "Right":
            self.handle_right()
        elif direction == "Return":
            self.handle_select_enter()

    def handle_up(self):
        # Handle up direction
        pass

    def handle_down(self):
        # Handle down direction
        pass

    def handle_left(self):
        # Handle left direction
        pass

    def handle_right(self):
        # Handle right direction
        pass

    def handle_select_enter(self):
        # Handle select/enter direction
        pass

    def start(self):
        self.window.mainloop()


gui = BrainCraftHATGUI()
gui.start()

# Define the GPIO pins for the 5-way switch
select_button = GPIOButton(16, pull_up=True)
left_button = GPIOButton(22, pull_up=True)
up_button = GPIOButton(23, pull_up=True)
right_button = GPIOButton(24, pull_up=True)
down_button = GPIOButton(27, pull_up=True)

# Create a list of buttons
buttons = [tk.Button(gui.window, text=f"Button {i}") for i in range(10)]
for button in buttons:
    button.pack()

# Keep track of the currently selected button
current_button = 0
buttons[current_button].config(relief=tk.SUNKEN)


# Define the functions to handle button presses
def handle_up():
    global current_button
    buttons[current_button].config(relief=tk.RAISED)
    current_button = (current_button - 1) % len(buttons)
    buttons[current_button].config(relief=tk.SUNKEN)


def handle_down():
    global current_button
    buttons[current_button].config(relief=tk.RAISED)
    current_button = (current_button + 1) % len(buttons)
    buttons[current_button].config(relief=tk.SUNKEN)


def handle_left():
    # Handle left direction
    pass


def handle_right():
    # Handle right direction
    pass


def handle_select_button():
    # "Click" the currently selected button
    buttons[current_button].invoke()


# Bind the button press events to the handle functions
up_button.when_pressed = handle_up
down_button.when_pressed = handle_down
left_button.when_pressed = handle_left
right_button.when_pressed = handle_right
select_button.when_pressed = handle_select_button

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Perform inference using the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
