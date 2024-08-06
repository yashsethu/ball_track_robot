import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the Edge TPU model
# Load the Edge TPU model
interpreter = tflite.Interpreter(
    model_path="/Users/Yash/Documents/GitHub/ball_track_robot/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
)

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (300, 300))
    frame = frame / 127.5 - 1
    frame = np.expand_dims(frame, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
