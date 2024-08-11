import RPi.GPIO as GPIO


class MotorController:
    def __init__(self, motor_b, motor_e):
        self.motor_b = motor_b
        self.motor_e = motor_e
        GPIO.setup([self.motor_b, self.motor_e], GPIO.OUT)

    def set_motor_pins(self, value):
        GPIO.output(self.motor_b, value)
        GPIO.output(self.motor_e, not value)

    def move(self, direction):
        if direction == "forward":
            self.set_motor_pins(GPIO.HIGH)
        elif direction == "reverse":
            self.set_motor_pins(GPIO.LOW)
        elif direction == "leftturn":
            self.set_motor_pins(GPIO.LOW)
        elif direction == "rightturn":
            self.set_motor_pins(GPIO.HIGH)
        elif direction == "stop":
            self.set_motor_pins(GPIO.LOW)
        elif direction == "sharp_left":
            self.set_motor_pins(GPIO.LOW)
        elif direction == "sharp_right":
            self.set_motor_pins(GPIO.HIGH)
        elif direction == "back_left":
            self.set_motor_pins(GPIO.LOW)
        elif direction == "back_right":
            self.set_motor_pins(GPIO.HIGH)

    @staticmethod
    def cleanup():
        GPIO.cleanup()


GPIO.setmode(GPIO.BOARD)

MOTOR1B = 16  # LEFT motor
MOTOR1E = 15

MOTOR2B = 11  # RIGHT motor
MOTOR2E = 13

motor1 = MotorController(MOTOR1B, MOTOR1E)
motor2 = MotorController(MOTOR2B, MOTOR2E)
