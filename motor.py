import RPi.GPIO as GPIO


class MotorController:
    def __init__(self, motor_b, motor_e):
        self.motor_b = motor_b
        self.motor_e = motor_e
        GPIO.setup([self.motor_b, self.motor_e], GPIO.OUT)

    def set_motor_pins(self, value):
        GPIO.output(self.motor_b, value)
        GPIO.output(self.motor_e, not value)

    def forward(self):
        self.set_motor_pins(GPIO.HIGH)

    def reverse(self):
        self.set_motor_pins(GPIO.LOW)

    def leftturn(self):
        self.set_motor_pins(GPIO.LOW)

    def rightturn(self):
        self.set_motor_pins(GPIO.HIGH)

    def stop(self):
        self.set_motor_pins(GPIO.LOW)

    def sharp_left(self):
        self.set_motor_pins(GPIO.LOW)

    def sharp_right(self):
        self.set_motor_pins(GPIO.HIGH)

    def back_left(self):
        self.set_motor_pins(GPIO.LOW)

    def back_right(self):
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
