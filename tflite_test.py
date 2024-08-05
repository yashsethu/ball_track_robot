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
        tf.keras.layers.Dense(2, activation="softmax"),  # Change this to 2
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],  # Change loss to binary_crossentropy
)

# Train the model
model.fit(
    train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
)

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video feed
    ret, frame = video_capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = frame.reshape(1, 64, 64, 3)
    frame = frame / 255.0

    # Make predictions using the trained model
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions)

    if predicted_class == 1:  # Assuming that your face is class 1
        print("Your face is detected!")
    else:
        print("Your face is not detected.")

    # Display the frame
    cv2.imshow("Face Detection", frame)
