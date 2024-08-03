import pigpio
import time
from picamera2 import Picamera2
import cv2
import time
from ball import find_color_mask, find_largest_contour
import os

redLower = (150, 140, 1)
redUpper = (190, 255, 255)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()
time.sleep(1)

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
time.sleep(1)


def calculate_pwm_values(
    x_ball,
    y_ball,
    frame_width=320,
    frame_height=240,
    horizontal_fov=54,
    vertical_fov=41,
):
    # Constants
    PAN_SERVO_CENTER = 90
    TILT_SERVO_CENTER = 90

    PWM_MIN = 500
    PWM_MAX = 2500
    DEGREE_MIN = 0
    DEGREE_MAX = 180

    # Center of the frame
    x_center = frame_width / 2
    y_center = frame_height / 2

    # Offsets from the center
    delta_x = x_ball - x_center
    delta_y = y_ball - y_center

    # Calculate angular displacement
    theta_pan = (delta_x / frame_width) * horizontal_fov
    theta_tilt = (delta_y / frame_height) * vertical_fov

    # Calculate servo angles
    servo_angle_pan = PAN_SERVO_CENTER + theta_pan
    servo_angle_tilt = TILT_SERVO_CENTER - theta_tilt  # subtract for tilt

    # Map servo angles to PWM values
    pwm_pan = PWM_MIN + (servo_angle_pan - DEGREE_MIN) * (PWM_MAX - PWM_MIN) / (
        DEGREE_MAX - DEGREE_MIN
    )
    pwm_tilt = PWM_MIN + (servo_angle_tilt - DEGREE_MIN) * (PWM_MAX - PWM_MIN) / (
        DEGREE_MAX - DEGREE_MIN
    )

    return pwm_pan, pwm_tilt


found = False

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

    if found:
        if x < 150:
            h_direction = "left"
            if pan_a < 2500:
                pan_a += 10
            pwm.set_servo_pulsewidth(pan, pan_a)
        elif x > 170:
            h_direction = "right"
            if pan_a > 500:
                pan_a -= 10
            pwm.set_servo_pulsewidth(pan, pan_a)

        if y > 130:
            v_direction = "up"
            if tilt_a < 2500:
                tilt_a += 10
            pwm.set_servo_pulsewidth(tilt, tilt_a)
        elif y < 110:
            v_direction = "down"
            if tilt_a > 500:
                tilt_a -= 10
            pwm.set_servo_pulsewidth(tilt, tilt_a)

    elif v_direction != "none" and h_direction != "none":
        if h_direction == "left":
            if pan_a < 2500:
                pan_a += 10
            pwm.set_servo_pulsewidth(pan, pan_a)
        elif h_direction == "right":
            if pan_a > 500:
                pan_a -= 10
            pwm.set_servo_pulsewidth(pan, pan_a)

        if v_direction == "up":
            if tilt_a < 2500:
                tilt_a += 10
            pwm.set_servo_pulsewidth(tilt, tilt_a)
        elif v_direction == "down":
            if tilt_a > 500:
                tilt_a -= 10
            pwm.set_servo_pulsewidth(tilt, tilt_a)

    cv2.imshow("Feed", frame)  # Shows frame with bounding box

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press q to break the loop and stop moving
        break


# Cleanup
print("90 deg")
pwm.set_servo_pulsewidth(pan, 1500)
pwm.set_servo_pulsewidth(tilt, 1500)
time.sleep(1)

pwm.set_PWM_dutycycle(pan, 0)
pwm.set_PWM_dutycycle(tilt, 0)

pwm.set_PWM_frequency(pan, 0)
pwm.set_PWM_frequency(tilt, 0)

cv2.destroyAllWindows()
picam2.stop()
