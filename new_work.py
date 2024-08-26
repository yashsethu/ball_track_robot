import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.lite.python import interpreter as interpreter_wrapper
from sklearn.model_selection import train_test_split
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

# Split the first dataset into training and testing sets
train_data_1, test_data_1 = train_test_split(train_generator_1, test_size=0.2)

# Split the second dataset into training and testing sets
train_data_2, test_data_2 = train_test_split(train_generator_2, test_size=0.2)

# Split the filtered second dataset into training and testing sets
train_data_2_filtered, test_data_2_filtered = train_test_split(
    train_generator_2_filtered, test_size=0.2
)


# Lower the resolution, resize the image, and change the colors
def preprocess_image(image):
    # Lower the resolution
    image = cv2.resize(image, (100, 100))

    # Change the colors
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


# Preprocess the images in the datasets
train_data_1 = [(preprocess_image(image), label) for image, label in train_data_1]
test_data_1 = [(preprocess_image(image), label) for image, label in test_data_1]
train_data_2 = [(preprocess_image(image), label) for image, label in train_data_2]
test_data_2 = [(preprocess_image(image), label) for image, label in test_data_2]
train_data_2_filtered = [
    (preprocess_image(image), label) for image, label in train_data_2_filtered
]
test_data_2_filtered = [
    (preprocess_image(image), label) for image, label in test_data_2_filtered
]

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

# Define the second base model
base_model_2 = tf.keras.applications.ResNet50(
    input_shape=(200, 200, 3), include_top=False
)
base_model_2.trainable = False

# Create the second model
model_2 = tf.keras.Sequential([base_model_2, tf.keras.layers.Dense(1)])

# Compile the second model
model_2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the second model
model_2.fit(train_data_2, epochs=5)

# Save the second model
model_2.save("model_2.h5")

# Convert the second model to TensorFlow Lite format
converter_2 = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model_2 = converter_2.convert()

# Save the second TensorFlow Lite model
with open("model_2.tflite", "wb") as f:
    f.write(tflite_model_2)

# Define the third base model
base_model_3 = tf.keras.applications.InceptionV3(
    input_shape=(200, 200, 3), include_top=False
)
base_model_3.trainable = False

# Create the third model
model_3 = tf.keras.Sequential([base_model_3, tf.keras.layers.Dense(1)])

# Compile the third model
model_3.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the third model
model_3.fit(train_data_3, epochs=5)

# Save the third model
model_3.save("model_3.h5")

# Convert the third model to TensorFlow Lite format
converter_3 = tf.lite.TFLiteConverter.from_keras_model(model_3)
tflite_model_3 = converter_3.convert()

# Save the third TensorFlow Lite model
with open("model_3.tflite", "wb") as f:
    f.write(tflite_model_3)
