import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.lite.python import interpreter as interpreter_wrapper


# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# Preprocess train and test data
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

# Create base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(200, 200, 3), include_top=False
)
base_model.trainable = False

# Create model
model = tf.keras.Sequential([base_model, tf.keras.layers.Dense(1)])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train model
model.fit(train_data_1, epochs=5)

# Save model
model.save("model.h5")

# Convert model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# Function to process frame
def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)
    rotated_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)
    return rotated_frame


# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Create interpreter for TFLite model
interpreter = interpreter_wrapper.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Open video capture
cap = cv2.VideoCapture(0)

# Process frames from video capture
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (200, 200))
    frame = frame / 255.0
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], frame[None, ...])
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]["index"])
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# Create another base model
base_model_2 = tf.keras.applications.ResNet50(
    input_shape=(200, 200, 3), include_top=False
)
base_model_2.trainable = False

# Create another model
model_2 = tf.keras.Sequential([base_model_2, tf.keras.layers.Dense(1)])

# Compile another model
model_2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train another model
model_2.fit(train_data_2, epochs=5)

# Save another model
model_2.save("model_2.h5")

# Convert another model to TFLite format
converter_2 = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model_2 = converter_2.convert()

# Save another TFLite model
with open("model_2.tflite", "wb") as f:
    f.write(tflite_model_2)

# Create yet another base model
base_model_3 = tf.keras.applications.InceptionV3(
    input_shape=(200, 200, 3), include_top=False
)
base_model_3.trainable = False

# Create yet another model
model_3 = tf.keras.Sequential([base_model_3, tf.keras.layers.Dense(1)])

# Compile yet another model
model_3.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train yet another model
model_3.fit(train_data_3, epochs=5)

# Save yet another model
model_3.save("model_3.h5")

# Convert yet another model to TFLite format
converter_3 = tf.lite.TFLiteConverter.from_keras_model(model_3)
tflite_model_3 = converter_3.convert()

# Save yet another TFLite model
with open("model_3.tflite", "wb") as f:
    f.write(tflite_model_3)
