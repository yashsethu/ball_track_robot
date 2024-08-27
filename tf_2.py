import tensorflow as tf
import cv2
import numpy as np

# Define the model architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Add regularization to the model
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Print the model summary
model.summary()


def process_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Apply erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)

    # Rotate 90 degrees
    rotated_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)

    return rotated_frame


def collect_data(num_samples):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize lists to store training and testing data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Collect data from the webcam
    for i in range(num_samples):
        # Read frame from the webcam
        ret, frame = cap.read()

        # Preprocess the frame (resize, normalize, etc.)
        preprocessed_frame = preprocess(frame)

        # Display the frame
        cv2.imshow("Collecting Data", frame)

        # Collect training data for the first half of the samples
        if i < num_samples // 2:
            train_data.append(preprocessed_frame)
            train_labels.append(label)
        # Collect testing data for the second half of the samples
        else:
            test_data.append(preprocessed_frame)
            test_labels.append(label)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Convert the data lists to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


# Collect training and testing data sets
train_data, train_labels, test_data, test_labels = collect_data(num_samples=100)

# Train the model
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
