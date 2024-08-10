import tensorflow as tf
import cv2

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

# Train the model
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

# Load the trained model
model = tf.keras.models.load_model("path_to_model.h5")

# Define the class labels
class_labels = ["class1", "class2", "class3", ...]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame (resize, normalize, etc.)
    preprocessed_frame = preprocess(frame)

    # Perform inference
    predictions = model.predict(preprocessed_frame)

    # Get the predicted class label
    predicted_label = class_labels[np.argmax(predictions)]

    # Display the predicted label on the frame
    cv2.putText(
        frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the frame
    cv2.imshow("Live Inference", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
