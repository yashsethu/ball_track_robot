import tensorflow as tf
import cv2
import numpy as np

# Define the first model architecture
model1 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compile the first model
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print the first model summary
model1.summary()

# Define the second model architecture
model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(input_dim, input_dim, 3)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compile the second model
model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print the second model summary
model2.summary()

# Collect training and testing data sets
train_data, train_labels, test_data, test_labels = collect_data(num_samples=100)

# Train the first model
model1.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

# Evaluate the first model on the testing data
test_loss1, test_accuracy1 = model1.evaluate(test_data, test_labels)
print("Test Loss (Model 1):", test_loss1)
print("Test Accuracy (Model 1):", test_accuracy1)

# Train the second model
model2.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

# Evaluate the second model on the testing data
test_loss2, test_accuracy2 = model2.evaluate(test_data, test_labels)
print("Test Loss (Model 2):", test_loss2)
print("Test Accuracy (Model 2):", test_accuracy2)


def main():
    # Define a video capture object
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("Failed to open video capture.")
        exit()

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow("frame", processed_frame)

        # Calculate color statistics
        color_stats = np.mean(processed_frame, axis=(0, 1))

        # Plot color statistics
        plt.bar(
            range(len(color_stats)),
            color_stats,
            color=["red", "green", "blue", "alpha"],
        )
        plt.xticks(range(len(color_stats)), ["Red", "Green", "Blue", "Alpha"])
        plt.xlabel("Color Channel")
        plt.ylabel("Mean Value")
        plt.title("Color Statistics")
        plt.show()

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Save the processed frame as an image
        cv2.imwrite("processed_frame.jpg", processed_frame)

    # Release the video capture object
    vid.release()

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
