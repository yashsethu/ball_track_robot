from PIL import Image
import cv2
import numpy as np
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

# Load the model
interpreter = make_interpreter("mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).resize((224, 224), Image.ANTIALIAS)
    input_tensor = np.asarray(image).reshape((1, 224, 224, 3))

    # Run the model
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_tensor)
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

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
