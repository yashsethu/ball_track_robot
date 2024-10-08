import cv2
import numpy as np
import tensorflow as tf

# Define the architecture of the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model3 = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

dataset_images = []
dataset_labels = []

while True:
    # Read each frame from the video feed
    ret, frame = video_capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0

    # Display the frame
    cv2.imshow("Dataset Capture", frame)

    # Capture user input for label
    label = input(
        "Enter the label for the image (1 for detected face, 0 for no face): "
    )

    # Append the image and label to the dataset
    dataset_images.append(frame)
    dataset_labels.append(int(label))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Convert the dataset to numpy arrays
dataset_images = np.array(dataset_images)
dataset_labels = np.array(dataset_labels)

# Save the dataset to file
np.savez("dataset.npz", images=dataset_images, labels=dataset_labels)

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()

# Train the model
model.fit(dataset_images, dataset_labels, epochs=10, validation_split=0.2)

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video feed
    ret, frame = video_capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = frame.reshape(1, 64, 64, 3)
    frame = frame / 255.0

    # Make predictions using the model
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions)

    if predicted_class == 1:
        print("Your face is detected!")
    else:
        print("Your face is not detected.")

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
