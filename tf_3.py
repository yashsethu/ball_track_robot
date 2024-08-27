import cv2
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


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


# Define the architecture of the models
models = []
models.append(
    tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(64, 64, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
)

models.append(
    tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(64, 64, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
)

# Compile the models
for model in models:
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

dataset_images = []
dataset_labels = []
dataset_images_2 = []
dataset_labels_2 = []

image_number = 1

while True:
    # Read each frame from the video feed
    ret, frame = video_capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0

    # Display the frame
    cv2.imshow("Dataset Capture", frame)

    # Capture user input for label
    label = input(
        f"Enter the label for image {image_number}: (1 for detected face, 0 for no face): "
    )

    # Append the image and label to the dataset
    dataset_images.append(frame)
    dataset_labels.append(int(label))

    # Append the image and label to the dataset for the second model
    dataset_images_2.append(frame)
    dataset_labels_2.append(int(label))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Convert the dataset to numpy arrays
dataset_images = np.array(dataset_images)
dataset_labels = np.array(dataset_labels)
dataset_images_2 = np.array(dataset_images_2)
dataset_labels_2 = np.array(dataset_labels_2)

# Save the dataset to file
np.savez("dataset.npz", images=dataset_images, labels=dataset_labels)

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()

# Train the model
model.fit(dataset_images, dataset_labels, epochs=10, validation_split=0.2)

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video feed
    ret, frame = video_capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = frame.reshape(1, 64, 64, 3)
    frame = frame / 255.0

    # Make predictions using the first model
    start_time = time.time()
    predictions_1 = models[0].predict(frame)
    end_time = time.time()
    inference_time_1 = end_time - start_time
    predicted_class_1 = np.argmax(predictions_1)

    # Make predictions using the second model
    start_time = time.time()
    predictions_2 = models[1].predict(frame)
    end_time = time.time()
    inference_time_2 = end_time - start_time
    predicted_class_2 = np.argmax(predictions_2)

    if predicted_class_1 == 1:  # Assuming that your face is class 1
        print("Your face is detected by Model 1!")
    else:
        print("Your face is not detected by Model 1.")

    if predicted_class_2 == 1:  # Assuming that your face is class 1
        print("Your face is detected by Model 2!")
    else:
        print("Your face is not detected by Model 2.")

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Plot probability scores
    plt.figure()
    plt.bar(["Class 0", "Class 1"], predictions_1[0])
    plt.title("Model 1 Probability Scores")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()

    plt.figure()
    plt.bar(["Class 0", "Class 1"], predictions_2[0])
    plt.title("Model 2 Probability Scores")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()

    # Print inference times
    print("Inference Time for Model 1:", inference_time_1)
    print("Inference Time for Model 2:", inference_time_2)

    # Calculate mean, maximum, and minimum probability scores
    mean_prob_1 = np.mean(predictions_1)
    mean_prob_2 = np.mean(predictions_2)
    max_prob_1 = np.max(predictions_1)
    max_prob_2 = np.max(predictions_2)
    min_prob_1 = np.min(predictions_1)
    min_prob_2 = np.min(predictions_2)

    print("Mean Probability for Model 1:", mean_prob_1)
    print("Mean Probability for Model 2:", mean_prob_2)
    print("Maximum Probability for Model 1:", max_prob_1)
    print("Maximum Probability for Model 2:", max_prob_2)
    print("Minimum Probability for Model 1:", min_prob_1)
    print("Minimum Probability for Model 2:", min_prob_2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
