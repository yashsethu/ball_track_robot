import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import time


# Function to generate data from webcam
def generate_data_from_webcam(num_samples):
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    cap = cv2.VideoCapture(0)

    for i in range(num_samples):
        ret, frame = cap.read()

        frame = cv2.resize(frame, (200, 200))
        frame = frame / 255.0

        filename = f"datasets/sample_{i}.jpg"
        cv2.imwrite(filename, frame)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


# Generate data from webcam and save to /datasets directory
generate_data_from_webcam(10)

# Load and preprocess the dataset
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define the model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(200, 200, 3), include_top=False
)
base_model.trainable = False
model = tf.keras.Sequential([base_model, tf.keras.layers.Dense(1)])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
model.fit(train_data_1, epochs=5)

# Save the model
model.save("model.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Load the TensorFlow Lite model with Edge TPU support
interpreter = interpreter_wrapper.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Capture the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (200, 200))
    frame = frame / 255.0

    # Set the input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], frame[None, ...])

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]["index"])

    # Draw bounding boxes and labels on the frame (this is a placeholder, you need a proper function here)
    # frame = draw_boxes(frame, output)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
