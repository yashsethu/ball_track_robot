import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2
import os
import matplotlib.pyplot as plt


# Function to collect data for the model
def collect_data():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Initialize variables for data collection
    data = []

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Collect data here (e.g., append frame to data list)
        data.append(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data to a file
    save_data(data)

    # Return the collected data
    return data


# Function to save the collected data to a file
def save_data(data):
    # Create a directory for the datasets if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Save each frame as an image file in the datasets directory
    for i, frame in enumerate(data):
        cv2.imwrite(f"datasets/frame_{i}.jpg", frame)


# Define your model architecture
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))  # Additional layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Collect data
collect_data()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

# Create empty lists to store inference times and confidence scores
inference_times = []
confidence_scores = []

# Open the webcam for live inferencing
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Start the timer
    start_time = time.time()

    # Perform inference on the frame using the loaded model
    predictions = loaded_model.predict(frame)

    # Calculate the inference time
    inference_time = time.time() - start_time

    # Append the inference time to the list
    inference_times.append(inference_time)

    # Get the maximum confidence score from the predictions
    max_confidence = np.max(predictions)

    # Append the maximum confidence score to the list
    confidence_scores.append(max_confidence)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()

# Plot the inference times and confidence scores
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(inference_times)
plt.title("Inference Times")
plt.xlabel("Frame")
plt.ylabel("Time (s)")

plt.subplot(1, 2, 2)
plt.plot(confidence_scores)
plt.title("Confidence Scores")
plt.xlabel("Frame")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# Save the model
model.save("model.h5")
# Load the saved model
loaded_model = tf.keras.models.load_model("model.h5")

# Open the webcam for live inferencing
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Perform inference on the frame using the loaded model
    predictions = loaded_model.predict(frame)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()
