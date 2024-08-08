import cv2
import numpy as np
import tensorflow as tf

# Load the trained TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path="/path/to/your/model.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    print("Error loading the model:", str(e))
    exit()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the labels for the classes
class_labels = [
    "Red Ball",
    "Blue Ball",
    "Green Ball",
    "Yellow Ball",
    "Orange Ball",
    "Purple Ball",
    "Pink Ball",
    "Black Ball",
    "White Ball",
    "Gray Ball",
]


# Function to preprocess the input frame
def preprocess_frame(frame):
    try:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Apply adaptive thresholding to enhance edges
        _, thresholded_frame = cv2.threshold(
            blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Preprocess the frame (e.g., resize, normalize, etc.)
        preprocessed_frame = cv2.resize(thresholded_frame, (32, 32))
        preprocessed_frame = preprocessed_frame / 255.0
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        return preprocessed_frame
    except Exception as e:
        print("Error preprocessing frame:", str(e))
        return None


# Function to perform inference on the frame
def perform_inference(frame):
    try:
        # Preprocess the frame
        input_data = preprocess_frame(frame)

        if input_data is None:
            return frame

        # Set the input tensor
        interpreter.set_tensor(input_details[0]["index"], input_data)

        # Perform inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Process the output data
        predicted_class = np.argmax(output_data, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Draw the predicted label on the frame
        cv2.putText(
            frame,
            predicted_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return frame
    except Exception as e:
        print("Error performing inference:", str(e))
        return frame


# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Perform inference on the frame
    output_frame = perform_inference(frame)

    # Display the output frame
    cv2.imshow("Webcam", output_frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
