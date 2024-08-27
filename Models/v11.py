import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.lite.python import interpreter as interpreter_wrapper
from sklearn.model_selection import train_test_split
import os
import time

IMAGE_SIZE = (200, 200)


def generate_data_from_webcam(num_samples):
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    cap = cv2.VideoCapture(0)
    for i in range(num_samples):
        ret, frame = cap.read()
        frame = cv2.resize(frame, IMAGE_SIZE)
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


def preprocess_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def train_and_save_model(base_model, train_data, model_name):
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, tf.keras.layers.Dense(1)])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(train_data, epochs=5)
    model.save(f"{model_name}.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f"{model_name}.tflite", "wb") as f:
        f.write(tflite_model)


train_data_1, test_data_1 = train_test_split(train_generator_1, test_size=0.2)
train_data_2, test_data_2 = train_test_split(train_generator_2, test_size=0.2)
train_data_2_filtered, test_data_2_filtered = train_test_split(
    train_generator_2_filtered, test_size=0.2
)

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

base_models = [
    tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False),
    tf.keras.applications.ResNet50(input_shape=(*IMAGE_SIZE, 3), include_top=False),
    tf.keras.applications.InceptionV3(input_shape=(*IMAGE_SIZE, 3), include_top=False),
]

train_data = [train_data_1, train_data_2, train_data_2_filtered]
model_names = ["model", "model_2", "model_3"]

for base_model, train_data, model_name in zip(base_models, train_data, model_names):
    train_and_save_model(base_model, train_data, model_name)

interpreter = interpreter_wrapper.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, IMAGE_SIZE)
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
