import cv2
import numpy as np
import tensorflow as tf

# Define the architecture of the models
models = []
models.append(
    tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(64, 64, 3)
            ),
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
)

models.append(
    tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(64, 64, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
)

# Compile the models
for model in models:
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

dataset_images = []
dataset_labels = []

image_number = 1

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
        f"Enter the label for image {image_number}: (1 for detected face, 0 for no face): "
    )

    # Append the image and label to the dataset
    dataset_images.append(frame)
    dataset_labels.append(int(label))

    # Append the image and label to the dataset for the second model
    dataset_images_2.append(frame)
    dataset_labels_2.append(int(label))

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

import matplotlib.pyplot as plt

while True:
    # Read each frame from the video feed
    ret, frame = video_capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = frame.reshape(1, 64, 64, 3)
    frame = frame / 255.0

    # Make predictions using the first model
    predictions_1 = models[0].predict(frame)
    predicted_class_1 = np.argmax(predictions_1)

    # Make predictions using the second model
    predictions_2 = models[1].predict(frame)
    predicted_class_2 = np.argmax(predictions_2)

    if predicted_class_1 == 1:  # Assuming that your face is class 1
        print("Your face is detected by Model 1!")
    else:
        print("Your face is not detected by Model 1.")

    if predicted_class_2 == 1:  # Assuming that your face is class 1
        print("Your face is detected by Model 2!")
    else:
        print("Your face is not detected by Model 2.")

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Plot probability scores
    plt.figure()
    plt.bar(["Class 0", "Class 1"], predictions_1[0])
    plt.title("Model 1 Probability Scores")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()

    plt.figure()
    plt.bar(["Class 0", "Class 1"], predictions_2[0])
    plt.title("Model 2 Probability Scores")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
