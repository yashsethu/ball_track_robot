import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

# Setup ultrasonic sensor
triggers = [22, 31, 33]
echos = [18, 29, 32]

for trigger, echo in zip(triggers, echos):
    GPIO.setup(trigger, GPIO.OUT)  # Trigger
    GPIO.setup(echo, GPIO.IN)  # Echo

    GPIO.output(trigger, False)


def sonar(trigger, echo):
    GPIO.output(trigger, True)
    time.sleep(0.00001)
    GPIO.output(trigger, False)

    start = time.time()
    while GPIO.input(echo) == 0 and time.time() < start + 0.05:
        pass
    pulse_start = time.time()

    while GPIO.input(echo) == 1 and time.time() < start + 0.1:
        pass
    pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2

    return distance


try:
    while True:
        distances = [sonar(trigger, echo) for trigger, echo in zip(triggers, echos)]
        print(f"Center: {distances[0]}, Left: {distances[1]}, Right: {distances[2]}")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    GPIO.cleanup()
