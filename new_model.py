import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def collect_data():
    cap = cv2.VideoCapture(0)
    data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture frame from webcam")
            cv2.imshow("Webcam", frame)
            data.append(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print("An error occurred during data collection:", str(e))

    cap.release()
    cv2.destroyAllWindows()

    save_data(data)

    return data


def save_data(data):
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    for i, frame in enumerate(data):
        cv2.imwrite(f"datasets/frame_{i}.jpg", frame)


def train_model(model, X_train, y_train):
    try:
        history = model.fit(X_train, y_train, epochs=10, batch_size=32)
    except Exception as e:
        print("An error occurred during training:", str(e))


def save_load_model(model):
    try:
        model.save("model.h5")
        loaded_model = tf.keras.models.load_model("model.h5")
    except Exception as e:
        print("An error occurred while saving/loading the model:", str(e))

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            try:
                predictions = loaded_model.predict(frame)
            except Exception as e:
                print("An error occurred during inference:", str(e))
                break

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print("An error occurred during inference:", str(e))

    cap.release()
    cv2.destroyAllWindows()


def train_additional_model(model, X_train, y_train, input_dim, output_dim):
    additional_model = Sequential()
    additional_model.add(Dense(128, activation="relu", input_shape=(input_dim,)))
    additional_model.add(Dense(64, activation="relu"))
    additional_model.add(Dense(32, activation="relu"))
    additional_model.add(Dense(output_dim, activation="softmax"))

    additional_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    try:
        additional_model.fit(X_train, y_train, epochs=10, batch_size=32)
    except Exception as e:
        print("An error occurred during training of additional model:", str(e))


def perform_inference(model):
    inference_times = []
    confidence_scores = []

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            start_time = time.time()

            try:
                predictions = model.predict(frame)
            except Exception as e:
                print("An error occurred during inference:", str(e))
                break

            inference_time = time.time() - start_time

            inference_times.append(inference_time)

            max_confidence = np.max(predictions)

            confidence_scores.append(max_confidence)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print("An error occurred during inference:", str(e))

    cap.release()
    cv2.destroyAllWindows()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(inference_times)
    plt.title("Inference Times")
    plt.xlabel("Frame")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.plot(confidence_scores)
    plt.title("Confidence Scores")
    plt.xlabel("Frame")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.show()


# Run the models
input_dim = 100  # Update with the appropriate input dimension
output_dim = 10  # Update with the appropriate output dimension

model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
model.add(Dense(64, activation="relu"))

X_train = ...
y_train = ...

train_model(model, X_train, y_train)

additional_X_train = ...
additional_y_train = ...

train_additional_model(
    model, additional_X_train, additional_y_train, input_dim, output_dim
)

save_load_model(model)

perform_inference(model)
