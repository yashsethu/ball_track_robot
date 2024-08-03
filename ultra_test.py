import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

# Setup ultrasonic sensor
Trigger_C = 22
Echo_C = 18
Trigger_L = 31
Echo_L = 29
Trigger_R = 33
Echo_R = 32

GPIO.setup(Trigger_C, GPIO.OUT)  # Trigger 1
GPIO.setup(Echo_C, GPIO.IN)  # Echo 1

GPIO.setup(Trigger_L, GPIO.OUT)  # Trigger 1
GPIO.setup(Echo_L, GPIO.IN)  # Echo 1

GPIO.setup(Trigger_R, GPIO.OUT)  # Trigger 1
GPIO.setup(Echo_R, GPIO.IN)  # Echo 1

GPIO.output(Trigger_C, False)
GPIO.output(Trigger_L, False)
GPIO.output(Trigger_R, False)


def sonar(GPIO_TRIGGER, GPIO_ECHO):
    start = 0
    stop = 0
    # Set pins as output and input
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # Trigger
    GPIO.setup(GPIO_ECHO, GPIO.IN)  # Echo

    # Set trigger to False (Low)
    GPIO.output(GPIO_TRIGGER, False)

    # Allow module to settle
    time.sleep(0.01)

    # while distance > 5:
    # Send 10us pulse to trigger
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    begin = time.time()
    while GPIO.input(GPIO_ECHO) == 0 and time.time() < begin + 0.05:
        start = time.time()

    while GPIO.input(GPIO_ECHO) == 1 and time.time() < begin + 0.1:
        stop = time.time()

    # Calculate pulse length
    elapsed = stop - start

    # Distance pulse traveled in that time is time multiplied by the speed of sound (cm/s)
    distance = elapsed * 34300

    # That was the distance there and back, so take half of the value
    distance = distance / 2

    # Reset GPIO settings, return distance (in cm) appropriate for robot movements
    return distance


while True:
    distance_c = sonar(Trigger_C, Echo_C)
    distance_l = sonar(Trigger_L, Echo_L)
    distance_r = sonar(Trigger_R, Echo_R)

    print(
        "Center: "
        + str(distance_c)
        + ", Left: "
        + str(distance_l)
        + ", Right: "
        + str(distance_r)
    )
