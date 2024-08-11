import time
import pigpio
from gpiozero import DistanceSensor
from picamera import PiCamera

# Constants
PAN_PIN = 18
TILT_PIN = 23
FREQUENCY = 50
SENSOR_PROXIMITY = 10
REROUTING_PROXIMITY = 17.5
SENSOR_C_CONFIG = {"echo": 18, "trigger": 22}
SENSOR_L_CONFIG = {"echo": 29, "trigger": 31}
SENSOR_R_CONFIG = {"echo": 32, "trigger": 33}
CAMERA_CONFIG = {"format": "RGB888", "size": (320, 240)}


# Initialize GPIO
def initialize_gpio():
    pwm = pigpio.pi()
    pwm.set_mode(PAN_PIN, pigpio.OUTPUT)
    pwm.set_mode(TILT_PIN, pigpio.OUTPUT)
    pwm.set_PWM_frequency(PAN_PIN, FREQUENCY)
    pwm.set_PWM_frequency(TILT_PIN, FREQUENCY)


# Initialize sensors
def initialize_sensors():
    sensor_C = DistanceSensor(**SENSOR_C_CONFIG)
    sensor_L = DistanceSensor(**SENSOR_L_CONFIG)
    sensor_R = DistanceSensor(**SENSOR_R_CONFIG)
    return sensor_C, sensor_L, sensor_R


# Initialize camera
def initialize_camera():
    picam2 = PiCamera()
    picam2.start_preview()
    time.sleep(1)
    return picam2


# Helper functions
def linear_scale(value, in_min, in_max, out_min, out_max):
    """Scales a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# Main program
def main():
    try:
        initialize_gpio()
        sensor_C, sensor_L, sensor_R = initialize_sensors()
        picam2 = initialize_camera()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # Initialize variables outside the loop
    h_direction = None

    # Main loop
    while True:
        try:
            frame = picam2.capture()
            if frame is None:
                raise Exception("Error: Frame not captured")
            # Rest of your code here
        except Exception as e:
            print(f"Error in main loop: {e}")
            break


if __name__ == "__main__":
    main()
