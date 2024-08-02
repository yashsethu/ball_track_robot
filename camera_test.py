from picamera2 import PiCamera2
import cv2

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()
time.sleep(2)

while True:
    frame = picam2.capture_array()

    if frame is None:
        print("Error: Frame not captured")
        break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
