Hey there! This is code, testing programs, and iterations of main code for my Ball-tracking robot! 

I used the OpenCV library and the alpha release of PiCamera2 to actually detect the ball from an image.

Hardware: Some motors, servos, ultrasonic sensors, a Raspberry Pi 4B, an Arducam OV5647 5MP Camera, and an H-bridge motor driver

To run on a Raspberry Pi:
1. Transfer all required code, ```requirements.txt```, and the ```ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu``` folder to the Pi through FTP, USB, anything similar
2. Make a Virtual Environment in Python: ```python -m venv --system-site-packages env && source env/bin/activate```
3. Run ```pip install -r requirements.txt```
4. Enjoy!
