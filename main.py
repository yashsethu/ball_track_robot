import RPi.GPIO as GPIO
import pigpio
from gpiozero import DistanceSensor
from picamera2 import Picamera2
import cv2
import time

from motor import forward, stop, sharp_left, sharp_right
from ball import find_color_mask, find_largest_contour

# Constants
PAN_PIN = 16
TILT_PIN = 26
PAN_CENTER = 1500
TILT_CENTER = 1500
MIN_RADIUS = 40

# Initialize GPIO and PWM
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)
pwm = pigpio.pi()
pwm.set_mode(PAN_PIN, pigpio.OUTPUT)
pwm.set_mode(TILT_PIN, pigpio.OUTPUT)
pwm.set_PWM_frequency(PAN_PIN, 50)
pwm.set_PWM_frequency(TILT_PIN, 50)

# Initialize sensors
sensor_proximity = 10
rerouting_proximity = 17.5
sensor_C = DistanceSensor(echo=18, trigger=22)
sensor_L = DistanceSensor(echo=29, trigger=31)
sensor_R = DistanceSensor(echo=32, trigger=33)

# Initialize camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()
time.sleep(1)


# Helper functions
def linear_scale(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def no_obstacle(distance_C, distance_L, distance_R):
    # Implement your logic here
    pass


# Main loop
while True:
    try:
        frame = picam2.capture_array()

        if frame is None:
            print("Error: Frame not captured")
            break

        height, width = frame.shape[:2]
        print(str(height) + " X " + str(width))

        mask = find_color_mask(frame)
        x, y, radius, center, area = find_largest_contour(mask)
        print("Area: " + str(area))
        print("Coordinates: " + str(x) + ", " + str(y))

        distance_C = sensor_C.distance * 100
        distance_L = sensor_L.distance * 100
        distance_R = sensor_R.distance * 100
        print("D: " + str(distance_C) + ", " + str(distance_L) + ", " + str(distance_R))

        if radius > MIN_RADIUS:
            found = True
            in_frame = True
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
        else:
            found = False
            in_frame = False

        if not no_obstacle(distance_C, distance_L, distance_R):
            print("Obstacle detected")
            stop()
            sleep(0.05)
        elif not found:
            sharp_right()
            sleep(0.075)
            stop()
        elif found and in_frame:
            if x > 260:
                h_direction = "right"
                sharp_right()
                sleep(0.075)
            elif x < 60:
                h_direction = "left"
                sharp_left()
                sleep(0.075)
            elif 110 <= x <= 210:
                forward()
                sleep(0.2)
            stop()
        elif found and not in_frame:
            if h_direction == "right":
                sharp_right()
            elif h_direction == "left":
                sharp_left()
            sleep(0.075)
            stop()

        print(h_direction)

        cv2.imshow("Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop()
            break
    except Exception as e:
        print("Error:", str(e))
        stop()
        break

# Clean up
pwm.set_servo_pulsewidth(PAN_PIN, PAN_CENTER)
pwm.set_servo_pulsewidth(TILT_PIN, TILT_CENTER)
sleep(1)
pwm.set_PWM_dutycycle(PAN_PIN, 0)
pwm.set_PWM_dutycycle(TILT_PIN, 0)
pwm.set_PWM_frequency(PAN_PIN, 0)
pwm.set_PWM_frequency(TILT_PIN, 0)
cv2.destroyAllWindows()
picam2.stop()
GPIO.cleanup()
