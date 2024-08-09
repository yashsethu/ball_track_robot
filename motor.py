import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

MOTOR1B = 16  # LEFT motor
MOTOR1E = 15

MOTOR2B = 11  # RIGHT motor
MOTOR2E = 13

GPIO.setup([MOTOR1B, MOTOR1E, MOTOR2B, MOTOR2E], GPIO.OUT)


def set_motor_pins(motor_b, motor_e, value):
    GPIO.output(motor_b, value)
    GPIO.output(motor_e, not value)


def forward():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.HIGH)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.HIGH)


def reverse():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.LOW)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.LOW)


def leftturn():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.LOW)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.HIGH)


def rightturn():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.HIGH)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.LOW)


def stop():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.LOW)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.LOW)


def sharp_left():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.LOW)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.HIGH)


def sharp_right():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.HIGH)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.LOW)


def back_left():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.LOW)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.LOW)


def back_right():
    set_motor_pins(MOTOR1B, MOTOR1E, GPIO.LOW)
    set_motor_pins(MOTOR2B, MOTOR2E, GPIO.HIGH)


def cleanup():
    GPIO.cleanup()
