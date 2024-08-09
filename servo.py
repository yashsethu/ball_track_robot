import pigpio
import time
from picamera2 import Picamera2
import cv2
from ball import find_color_mask, find_largest_contour

# Constants
RED_LOWER = (150, 140, 1)
RED_UPPER = (190, 255, 255)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
HORIZONTAL_FOV = 54
VERTICAL_FOV = 41
PAN_SERVO_PIN = 16
TILT_SERVO_PIN = 26
PAN_SERVO_CENTER = 1500
TILT_SERVO_CENTER = 1500
PWM_MIN = 500
PWM_MAX = 2500
DEGREE_MIN = 0
DEGREE_MAX = 180
DEADZONE = 20

# Initialize pigpio
pwm = pigpio.pi()
pwm.set_mode(PAN_SERVO_PIN, pigpio.OUTPUT)
pwm.set_mode(TILT_SERVO_PIN, pigpio.OUTPUT)
pwm.set_PWM_frequency(PAN_SERVO_PIN, 50)
pwm.set_PWM_frequency(TILT_SERVO_PIN, 50)

# Initialize PiCamera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()
time.sleep(1)

# Initialize variables
pan_angle = 0
tilt_angle = 0
h_direction = "none"
v_direction = "none"


def calculate_pwm_values(x, y):
    # Calculate angular displacement
    theta_pan = (x - FRAME_WIDTH / 2) / FRAME_WIDTH * HORIZONTAL_FOV
    theta_tilt = (y - FRAME_HEIGHT / 2) / FRAME_HEIGHT * VERTICAL_FOV

    # Calculate servo angles
    servo_angle_pan = PAN_SERVO_CENTER + theta_pan
    servo_angle_tilt = TILT_SERVO_CENTER - theta_tilt

    # Map servo angles to PWM values
    pwm_pan = PWM_MIN + (servo_angle_pan - DEGREE_MIN) * (PWM_MAX - PWM_MIN) / (
        DEGREE_MAX - DEGREE_MIN
    )
    pwm_tilt = PWM_MIN + (servo_angle_tilt - DEGREE_MIN) * (PWM_MAX - PWM_MIN) / (
        DEGREE_MAX - DEGREE_MIN
    )

    return pwm_pan, pwm_tilt


def move_servos(x, y):
    global pan_angle, tilt_angle, h_direction, v_direction

    # Calculate PWM values
    pwm_pan, pwm_tilt = calculate_pwm_values(x, y)

    # Update servo angles and directions
    if x < FRAME_WIDTH / 2 - DEADZONE:
        h_direction = "left"
        if pan_angle < 2500:
            pan_angle += int(10 * (FRAME_WIDTH / 2 - DEADZONE - x) / (FRAME_WIDTH / 2))
    elif x > FRAME_WIDTH / 2 + DEADZONE:
        h_direction = "right"
        if pan_angle > 500:
            pan_angle -= int(10 * (x - FRAME_WIDTH / 2 - DEADZONE) / (FRAME_WIDTH / 2))

    if y < FRAME_HEIGHT / 2 - DEADZONE:
        v_direction = "up"
        if tilt_angle < 2500:
            tilt_angle += int(
                10 * (FRAME_HEIGHT / 2 - DEADZONE - y) / (FRAME_HEIGHT / 2)
            )
    elif y > FRAME_HEIGHT / 2 + DEADZONE:
        v_direction = "down"
        if tilt_angle > 500:
            tilt_angle -= int(
                10 * (y - FRAME_HEIGHT / 2 - DEADZONE) / (FRAME_HEIGHT / 2)
            )

    # Move servos
    pwm.set_servo_pulsewidth(PAN_SERVO_PIN, pan_angle)
    pwm.set_servo_pulsewidth(TILT_SERVO_PIN, tilt_angle)


# Main loop
while True:
    frame = picam2.capture_array()

    mask = find_color_mask(frame)
    x, y, radius, center, area = find_largest_contour(mask)

    if radius > 20:
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        move_servos(x, y)

    cv2.imshow("Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
pwm.set_servo_pulsewidth(PAN_SERVO_PIN, PAN_SERVO_CENTER)
pwm.set_servo_pulsewidth(TILT_SERVO_PIN, TILT_SERVO_CENTER)
time.sleep(1)
pwm.set_PWM_dutycycle(PAN_SERVO_PIN, 0)
pwm.set_PWM_dutycycle(TILT_SERVO_PIN, 0)
pwm.set_PWM_frequency(PAN_SERVO_PIN, 0)
pwm.set_PWM_frequency(TILT_SERVO_PIN, 0)
cv2.destroyAllWindows()
picam2.stop()
