import tensorflow as tf
import cv2
import numpy as np

# Call the function to collect 100 samples from the webcam
collected_data = collect_data_from_webcam(100)

# Convert the collected data to a numpy array
x_train = np.array(collected_data)

# Define the labels for the collected data
y_train = np.zeros((x_train.shape[0], 1))
y_train[:50] = 1

# Define the input dimension and number of classes
input_dim = x_train.shape[1]
num_classes = 2

# Define the model architecture
lay_1 = tf.keras.Sequential(
    [tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,))]
)

lay_2 = tf.keras.Sequential([tf.keras.layers.Dense(64, activation="relu")])

lay_3 = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="relu")])

lay_4 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compile the models
lay_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lay_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lay_3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lay_4.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the models
lay_1.fit(x_train, y_train, epochs=10, batch_size=32)
lay_2.fit(x_train, y_train, epochs=10, batch_size=32)
lay_3.fit(x_train, y_train, epochs=10, batch_size=32)
lay_4.fit(x_train, y_train, epochs=10, batch_size=32)

# Save the trained models
lay_1.save("path_to_lay_1_model")
lay_2.save("path_to_lay_2_model")
lay_3.save("path_to_lay_3_model")
lay_4.save("path_to_lay_4_model")

# Load the trained models
model_1 = tf.keras.models.load_model("path_to_lay_1_model")
model_2 = tf.keras.models.load_model("path_to_lay_2_model")
model_3 = tf.keras.models.load_model("path_to_lay_3_model")
model_4 = tf.keras.models.load_model("path_to_lay_4_model")


# Define the function for live inferencing
def live_inference(model):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        normalized_frame = resized_frame / 255.0

        # Perform inference using the trained model
        prediction = model.predict(np.expand_dims(normalized_frame, axis=0))

        # Get the predicted class label
        predicted_class = np.argmax(prediction)

        # Display the frame with predictions
        cv2.putText(
            frame,
            f"Predicted Class: {predicted_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Live Inference", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


# Call the live inference function for each model
live_inference(model_1)
live_inference(model_2)
live_inference(model_3)
live_inference(model_4)


# Define the function for live inferencing
def live_inference():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        normalized_frame = resized_frame / 255.0

        # Perform inference using the trained model
        prediction = model.predict(np.expand_dims(normalized_frame, axis=0))

        # Get the predicted class label
        predicted_class = np.argmax(prediction)

        # Display the frame with predictions
        cv2.putText(
            frame,
            f"Predicted Class: {predicted_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Live Inference", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


# Call the live inference function
# Plot the standard deviation of each image
std_deviation = np.std(x_train, axis=(1, 2))
plt.figure(figsize=(6, 4))
plt.plot(std_deviation, color="gray")
plt.xlabel("Image Index")
plt.ylabel("Standard Deviation")
plt.title("Standard Deviation of Images")
plt.show()

# Plot the maximum intensity of each image
max_intensity = np.max(x_train, axis=(1, 2))
plt.figure(figsize=(6, 4))
plt.plot(max_intensity, color="gray")
plt.xlabel("Image Index")
plt.ylabel("Maximum Intensity")
plt.title("Maximum Intensity of Images")
plt.show()
