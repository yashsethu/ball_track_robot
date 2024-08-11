from gpiozero import DistanceSensor
from picamera2 import Picamera2
import time
from ball import find_color_mask, find_largest_contour

# Constants
FRAME_HEIGHT = 240
FRAME_WIDTH = 320

# Initialize sensors
sensor_L = DistanceSensor(echo=29, trigger=31)
sensor_R = DistanceSensor(echo=32, trigger=33)

# Initialize camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()
time.sleep(1)

# Initialize variables outside the loop
h_direction = None

# Main loop
while True:
    try:
        frame = picam2.capture_array()

        if frame is None:
            raise Exception("Error: Frame not captured")

        print(f"{FRAME_HEIGHT} X {FRAME_WIDTH}")

        mask = find_color_mask(frame)
        x, y, radius, center, area = find_largest_contour(mask)

        # Add the rest of your code here...

    except Exception as e:
        print(f"Error: {e}")
        stop()
        break
