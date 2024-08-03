# Import motor and camera setup from our files
from motor import (
    forward,
    stop,
    sharp_left,
    sharp_right,
)
from ball import find_color_mask, find_largest_contour
from time import sleep
import RPi.GPIO as GPIO
import pigpio

pan = 16
tilt = 26

pan_a = 1500
tilt_a = 1500

v_direction = "none"
h_direction = "right"

pwm = pigpio.pi()

pwm.set_mode(pan, pigpio.OUTPUT)
pwm.set_mode(tilt, pigpio.OUTPUT)

pwm.set_PWM_frequency(pan, 50)
pwm.set_PWM_frequency(tilt, 50)

print("90 deg")
pwm.set_servo_pulsewidth(pan, 1500)
pwm.set_servo_pulsewidth(tilt, 1500)

from gpiozero import DistanceSensor
from time import sleep

# Setup ultrasonic sensor
sensor_proximity = 10
rerouting_proximity = 17.5

sensor_C = DistanceSensor(echo=18, trigger=22)
sensor_L = DistanceSensor(echo=29, trigger=31)
sensor_R = DistanceSensor(echo=32, trigger=33)


def sonar(sensor):
    distance = sensor.distance * 100
    return distance


def no_obstacle(distance_c, distance_l, distance_r):
    if (
        distance_c > sensor_proximity
        and distance_l > sensor_proximity
        and distance_r > sensor_proximity
    ):
        return True
    else:
        return False


# Setup motors

MOTOR1B = 16  # LEFT motor
MOTOR1E = 15

MOTOR2B = 11  # RIGHT motor
MOTOR2E = 13

GPIO.setup(MOTOR1B, GPIO.OUT)
GPIO.setup(MOTOR1E, GPIO.OUT)
GPIO.setup(MOTOR2B, GPIO.OUT)
GPIO.setup(MOTOR2E, GPIO.OUT)

# Setup camera
from picamera2 import Picamera2
import cv2
import time

redLower = (150, 140, 1)
redUpper = (190, 255, 255)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()
time.sleep(1)

# Ultrasonic Sensor proximity parameter (centimeter)
sensor_proximity = 10
rerouting_proximity = 17.5

# Computer vision lower and upper turning range parameters for tracking
lower_range = 30
upper_range = 290

found = False
in_frame = False
direction = "none"

while True:
    # Process current frame with our functions
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

    # Distance coming from front ultrasonic sensor
    distance_C = sonar(Trigger_C, Echo_C)
    distance_L = sonar(Trigger_L, Echo_L)
    distance_R = sonar(Trigger_R, Echo_R)
    print("D: " + str(distance_C) + ", " + str(distance_L) + ", " + str(distance_R))

    if radius > 40:
        if not found:
            found = True
            print("Found: " + str(found))
        in_frame = True
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
    else:
        in_frame = False

    if not no_obstacle(distance_C, distance_L, distance_R):
        print("Obstacle detetcted")
        stop()
        sleep(0.05)
    elif not found:
        sharp_right()
        sleep(0.075)
        stop()
    elif found and in_frame:
        if x > 260:
            direction = "right"
            sharp_right()
            sleep(0.075)
        elif x < 60:
            direction = "left"
            sharp_left()
            sleep(0.075)
        elif 110 <= x <= 210:
            forward()
            sleep(0.2)
        stop()
    elif found and not in_frame:
        if direction == "right":
            sharp_right()
        elif direction == "left":
            sharp_left()
        sleep(0.075)
        stop()

    print(direction)

    cv2.imshow("Feed", frame)  # Shows frame with bounding box

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press q to break the loop and stop moving
        stop()
        break

while True:
    frame = picam2.capture_array()

    mask = find_color_mask(frame)
    x, y, radius, center, area = find_largest_contour(mask)

    if radius > 20:
        found = True
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
    else:
        found = False

    def linear_scale(value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    if found:
        if x < 150:
            h_direction = "left"
            pan_a = linear_scale(pan_a, 500, 2500, pan_a + 20, pan_a - 20)
            pwm.set_servo_pulsewidth(pan, pan_a)
        elif x > 170:
            h_direction = "right"
            pan_a = linear_scale(pan_a, 500, 2500, pan_a - 20, pan_a + 20)
            pwm.set_servo_pulsewidth(pan, pan_a)

        if y > 130:
            v_direction = "up"
            tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a + 20, tilt_a - 20)
            pwm.set_servo_pulsewidth(tilt, tilt_a)
        elif y < 110:
            v_direction = "down"
            tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a - 20, tilt_a + 20)
            pwm.set_servo_pulsewidth(tilt, tilt_a)

    elif v_direction != "none" and h_direction != "none":
        if h_direction == "left":
            pan_a = linear_scale(pan_a, 500, 2500, pan_a + 20, pan_a - 20)
            pwm.set_servo_pulsewidth(pan, pan_a)
        elif h_direction == "right":
            pan_a = linear_scale(pan_a, 500, 2500, pan_a - 20, pan_a + 20)
            pwm.set_servo_pulsewidth(pan, pan_a)

        if v_direction == "up":
            tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a + 20, tilt_a - 20)
            pwm.set_servo_pulsewidth(tilt, tilt_a)
        elif v_direction == "down":
            tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a - 20, tilt_a + 20)
            pwm.set_servo_pulsewidth(tilt, tilt_a)

    cv2.imshow("Feed", frame)  # Shows frame with bounding box

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press q to break the loop and stop moving
        break

pwm.set_servo_pulsewidth(pan, 1500)
pwm.set_servo_pulsewidth(tilt, 1500)
sleep(1)

# Cleanup
pwm.set_PWM_dutycycle(pan, 0)
pwm.set_PWM_dutycycle(tilt, 0)

pwm.set_PWM_frequency(pan, 0)
pwm.set_PWM_frequency(tilt, 0)

cv2.destroyAllWindows()
picam2.stop()
GPIO.cleanup()
