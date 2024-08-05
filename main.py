import RPi.GPIO as GPIO
import pigpio
from gpiozero import DistanceSensor
from picamera2 import Picamera2
import cv2
import time

from motor import forward, stop, sharp_left, sharp_right
from ball import find_color_mask, find_largest_contour

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

sensor_proximity = 10
rerouting_proximity = 17.5

sensor_C = DistanceSensor(echo=18, trigger=22)
sensor_L = DistanceSensor(echo=29, trigger=31)
sensor_R = DistanceSensor(echo=32, trigger=33)

GPIO.setup(16, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

redLower = (150, 140, 1)
redUpper = (190, 255, 255)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()
time.sleep(1)


def linear_scale(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


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

        if radius > 40:
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
        if x < 130:
            h_direction = "left"
            if x < 110:
                pan_a = linear_scale(pan_a, 500, 2500, pan_a + 20, pan_a - 20)
            else:
                pan_a = linear_scale(pan_a, 500, 2500, pan_a + 20, pan_a)
            pwm.set_servo_pulsewidth(pan, pan_a)
        elif x > 190:
            h_direction = "right"
            if x > 210:
                pan_a = linear_scale(pan_a, 500, 2500, pan_a - 20, pan_a + 20)
            else:
                pan_a = linear_scale(pan_a, 500, 2500, pan_a, pan_a + 20)
            pwm.set_servo_pulsewidth(pan, pan_a)

        if y > 150:
            v_direction = "up"
            if y > 170:
                tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a + 20, tilt_a - 20)
            else:
                tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a + 20, tilt_a)
            pwm.set_servo_pulsewidth(tilt, tilt_a)
        elif y < 90:
            v_direction = "down"
            if y < 70:
                tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a - 20, tilt_a + 20)
            else:
                tilt_a = linear_scale(tilt_a, 500, 2500, tilt_a, tilt_a + 20)
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

    cv2.imshow("Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pwm.set_servo_pulsewidth(pan, 1500)
pwm.set_servo_pulsewidth(tilt, 1500)
sleep(1)

pwm.set_PWM_dutycycle(pan, 0)
pwm.set_PWM_dutycycle(tilt, 0)

pwm.set_PWM_frequency(pan, 0)
pwm.set_PWM_frequency(tilt, 0)

cv2.destroyAllWindows()
picam2.stop()
GPIO.cleanup()
