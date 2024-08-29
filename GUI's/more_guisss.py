import tkinter as tk
import psutil
from tkinter import Label, Button
from gpiozero import Button as GPIOButton


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
