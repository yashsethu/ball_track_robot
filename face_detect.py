from keras.preprocessing import image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2

# Define the Face Detection Model architecture
face_detection_model = tf.keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax"),  # Binary classification: face or no face
    ]
)

# Compile the Face Detection Model
face_detection_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Define the Face Recognition Model architecture
face_recognition_model = tf.keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(
            2, activation="softmax"
        ),  # Binary classification: your face or other face
    ]
)

# Compile the Face Recognition Model
face_recognition_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


def load_face_images():
    image_dir = "datasets/face_images"  # Corrected directory

    # Create the directory if it doesn't exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_files = os.listdir(image_dir)
    images = []
    labels = []

    for image_file in image_files:
        img = image.load_img(os.path.join(image_dir, image_file), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Ensure the image has shape (1, 64, 64, 3)
        images.append(img_array)
        labels.append(1)  # Assuming all images are of faces

    return np.vstack(images), np.array(
        labels
    )  # vstack to get shape (num_images, 64, 64, 3)


def load_no_face_images():
    image_dir = "datasets/no_face_images"

    # Create the directory if it doesn't exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_files = os.listdir(image_dir)
    images = []
    labels = []

    for image_file in image_files:
        img = image.load_img(os.path.join(image_dir, image_file), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Ensure the image has shape (1, 64, 64, 3)
        images.append(img_array)
        labels.append(0)  # Assuming all images are not of faces

    return np.vstack(images), np.array(
        labels
    )  # vstack to get shape (num_images, 64, 64, 3)


def load_my_face_images():
    image_dir = "datasets/my_face_images"

    # Create the directory if it doesn't exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_files = os.listdir(image_dir)
    images = []
    labels = []

    for image_file in image_files:
        img = image.load_img(os.path.join(image_dir, image_file), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Ensure the image has shape (1, 64, 64, 3)
        images.append(img_array)
        labels.append(1)  # Assuming all images are of your face

    return np.vstack(images), np.array(
        labels
    )  # vstack to get shape (num_images, 64, 64, 3)


def load_other_face_images():
    image_dir = "datasets/other_face_images"

    # Create the directory if it doesn't exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_files = os.listdir(image_dir)
    images = []
    labels = []

    for image_file in image_files:
        img = image.load_img(os.path.join(image_dir, image_file), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Ensure the image has shape (1, 64, 64, 3)
        images.append(img_array)
        labels.append(1)  # Assuming all images are of other faces

    return np.vstack(images), np.array(
        labels
    )  # vstack to get shape (num_images, 64, 64, 3)


# Load the datasets
face_images, face_labels = load_face_images()  # Images with a face
no_face_images, no_face_labels = load_no_face_images()  # Images without a face
your_face_images, your_face_labels = load_my_face_images()  # Images of your face
other_face_images, other_face_labels = load_other_face_images()  # Images of other faces

# Train the Face Detection Model
face_detection_model.fit(
    np.concatenate((face_images, no_face_images)),
    np.concatenate((face_labels, no_face_labels)),
    epochs=10,
)

# Train the Face Recognition Model
face_recognition_model.fit(
    np.concatenate((your_face_images, other_face_images)),
    np.concatenate((your_face_labels, other_face_labels)),
    epochs=10,
)


# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, original_frame = cap.read()

    # Preprocess the frame for face detection
    frame = cv2.resize(original_frame, (64, 64))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Perform face detection
    face_detection_result = face_detection_model.predict(frame)
    face_label = np.argmax(face_detection_result)

    if face_label == 1:
        # Preprocess the frame for face recognition
        frame = cv2.resize(original_frame, (64, 64))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Perform face recognition
        face_recognition_result = face_recognition_model.predict(frame)
        your_face_label = np.argmax(face_recognition_result)

        # Display the inference on the frame
        if your_face_label == 1:
            cv2.putText(
                original_frame,
                "Your Face",
                (0, 0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                original_frame,
                "Other Face",
                (0, 0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    # Display the original frame
    cv2.imshow("Live Inference", original_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
