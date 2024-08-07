import cv2
import numpy as np
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

# Load the model
model_path = "path/to/your/model.tflite"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input and output shapes
input_shape = input_details[0]["shape"]
output_shape = output_details[0]["shape"]

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the desired resolution and size
width = 320
height = 240

# Set the capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = cv2.resize(frame_rgb, (input_shape[2], input_shape[1]))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run the model
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    # Get the results
    classes = classify.get_classes(interpreter, 1, 0.0)

    # Print the results
    if classes:
        print("Class ID:", classes[0].id, "Score:", classes[0].score)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
