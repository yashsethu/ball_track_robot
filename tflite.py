import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Initialize lists to store confidence scores over time
confidence_scores = []
time_steps = []

# Initialize lists to store inference times and performance
inference_times = []
performance = []


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

        # Store the confidence score and time step
        confidence_scores.append(output_data[0][predicted_class])
        time_steps.append(len(time_steps) + 1)

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


def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Perform inference on the frame
        start_time = time.time()
        output_frame = perform_inference(frame)
        end_time = time.time()

        # Calculate inference time
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # Calculate performance (frames per second)
        performance.append(1 / inference_time)

        # Display the output frame
        cv2.imshow("Webcam", output_frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()

    # Plot the confidence scores over time
    plt.plot(time_steps, confidence_scores)
    plt.xlabel("Time Step")
    plt.ylabel("Confidence Score")
    plt.title("Confidence Scores Over Time")
    plt.show()

    # Plot the inference times over time
    plt.plot(time_steps, inference_times)
    plt.xlabel("Time Step")
    plt.ylabel("Inference Time (s)")
    plt.title("Inference Times Over Time")
    plt.show()

    # Plot the performance over time
    plt.plot(time_steps, performance)
    plt.xlabel("Time Step")
    plt.ylabel("Performance (fps)")
    plt.title("Performance Over Time")
    plt.show()

    # Plot the confidence scores, inference times, and performance on the same graph
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Confidence Score", color=color)
    ax1.plot(time_steps, confidence_scores, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Inference Time (s)", color=color)
    ax2.plot(time_steps, inference_times, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax3 = ax1.twinx()
    color = "tab:green"
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Performance (fps)", color=color)
    ax3.plot(time_steps, performance, color=color)
    ax3.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Metrics Over Time")
    plt.show()


if __name__ == "__main__":
    main()
