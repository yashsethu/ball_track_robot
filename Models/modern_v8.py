import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.lite.python import interpreter as interpreter_wrapper
from sklearn.model_selection import train_test_split
import os
import time


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


generate_data_from_webcam(10)

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data_1, test_data_1 = train_test_split(train_generator_1, test_size=0.2)
train_data_2, test_data_2 = train_test_split(train_generator_2, test_size=0.2)
train_data_2_filtered, test_data_2_filtered = train_test_split(
    train_generator_2_filtered, test_size=0.2
)


def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


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

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(200, 200, 3), include_top=False
)
base_model.trainable = False
model = tf.keras.Sequential([base_model, tf.keras.layers.Dense(1)])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_data_1, epochs=5)

model.save("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)
    rotated_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)
    return rotated_frame


with open("model.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = interpreter_wrapper.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)

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

cap.release()
cv2.destroyAllWindows()

base_model_2 = tf.keras.applications.ResNet50(
    input_shape=(200, 200, 3), include_top=False
)
base_model_2.trainable = False
model_2 = tf.keras.Sequential([base_model_2, tf.keras.layers.Dense(1)])
model_2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model_2.fit(train_data_2, epochs=5)
model_2.save("model_2.h5")
converter_2 = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model_2 = converter_2.convert()
with open("model_2.tflite", "wb") as f:
    f.write(tflite_model_2)

base_model_3 = tf.keras.applications.InceptionV3(
    input_shape=(200, 200, 3), include_top=False
)
base_model_3.trainable = False
model_3 = tf.keras.Sequential([base_model_3, tf.keras.layers.Dense(1)])
model_3.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model_3.fit(train_data_3, epochs=5)
model_3.save("model_3.h5")
converter_3 = tf.lite.TFLiteConverter.from_keras_model(model_3)
tflite_model_3 = converter_3.convert()
with open("model_3.tflite", "wb") as f:
    f.write(tflite_model_3)
