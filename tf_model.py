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
    image_dir = "face_images"

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


# Train the model
train_images, train_labels = load_face_images()  # Load the collected face images
model.fit(train_images, train_labels, epochs=10)

# Perform live inference
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        resized_roi = cv2.resize(roi_gray, (64, 64))
        # Normalize, add color channel, and expand dimensions for prediction
        normalized_roi = resized_roi.astype("uint8")

        # Convert grayscale image to RGB
        normalized_roi = cv2.cvtColor(normalized_roi, cv2.COLOR_GRAY2RGB)

        # Normalize again
        normalized_roi = normalized_roi / 255.0

        # Add a new axis to match with the input shape the model expects
        normalized_roi = tf.expand_dims(normalized_roi, axis=0)

        prediction = model.predict(normalized_roi)
        label = "Your Face" if prediction[0][1] > prediction[0][0] else "Unknown"
        cv2.putText(
            frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
