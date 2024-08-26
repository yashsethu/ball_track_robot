import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.lite.python import interpreter as interpreter_wrapper


def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
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
    model.save(model_name)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f"{model_name}.tflite", "wb") as f:
        f.write(tflite_model)


def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)
    rotated_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)
    return rotated_frame


def run_inference(interpreter):
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

base_model_1 = tf.keras.applications.MobileNetV2(
    input_shape=(200, 200, 3), include_top=False
)
train_and_save_model(base_model_1, train_data_1, "model")

base_model_2 = tf.keras.applications.ResNet50(
    input_shape=(200, 200, 3), include_top=False
)
train_and_save_model(base_model_2, train_data_2, "model_2")

base_model_3 = tf.keras.applications.InceptionV3(
    input_shape=(200, 200, 3), include_top=False
)
train_and_save_model(base_model_3, train_data_3, "model_3")

interpreter = interpreter_wrapper.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
run_inference(interpreter)
