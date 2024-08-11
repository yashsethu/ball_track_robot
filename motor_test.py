import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)

MOTOR1B = 16  # LEFT motor
MOTOR1E = 15

MOTOR2B = 11  # RIGHT motor
MOTOR2E = 13

GPIO.setup([MOTOR1B, MOTOR1E, MOTOR2B, MOTOR2E], GPIO.OUT)


def set_motor_pins(motor1b, motor1e, motor2b, motor2e):
    GPIO.output(MOTOR1B, motor1b)
    GPIO.output(MOTOR1E, motor1e)
    GPIO.output(MOTOR2B, motor2b)
    GPIO.output(MOTOR2E, motor2e)


def forward():
    set_motor_pins(GPIO.HIGH, GPIO.LOW, GPIO.HIGH, GPIO.LOW)


def reverse():
    set_motor_pins(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.HIGH)


def leftturn():
    set_motor_pins(GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.LOW)


def rightturn():
    set_motor_pins(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.LOW)


def stop():
    set_motor_pins(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW)


def sharp_left():
    set_motor_pins(GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW)


def sharp_right():
    set_motor_pins(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH)


def back_left():
    set_motor_pins(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.HIGH)


def back_right():
    set_motor_pins(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.LOW)


def cleanup():
    GPIO.cleanup()
