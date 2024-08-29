import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import tkinter as tk
from PIL import ImageTk, Image


def create_model(input_shape, num_classes):
    model = tf.keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def compile_model(model):
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_images(image_dir, target_size, label):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_files = os.listdir(image_dir)
    images = []
    labels = []

    for image_file in image_files:
        img = image.load_img(
            os.path.join(image_dir, image_file), target_size=target_size
        )
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        images.append(img_array)
        labels.append(label)

    return np.vstack(images), np.array(labels)


def train_model(model, images, labels, epochs):
    model.fit(images, labels, epochs=epochs)


def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


def perform_inference(frame, model):
    result = model.predict(frame)
    label = np.argmax(result)
    return label


def display_inference(frame, label):
    if label == 1:
        cv2.putText(
            frame, "Your Face", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            frame, "Other Face", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    cv2.imshow("Live Inference", frame)


def create_and_compile_models(input_shape, num_classes):
    face_detection_model = create_model(input_shape, num_classes)
    face_detection_model = compile_model(face_detection_model)

    face_recognition_model = create_model(input_shape, num_classes)
    face_recognition_model = compile_model(face_recognition_model)

    return face_detection_model, face_recognition_model


def train_face_detection_model(
    face_detection_model,
    face_images,
    no_face_images,
    face_labels,
    no_face_labels,
    epochs,
):
    train_model(
        face_detection_model,
        np.concatenate((face_images, no_face_images)),
        np.concatenate((face_labels, no_face_labels)),
        epochs,
    )


def train_face_recognition_model(
    face_recognition_model,
    your_face_images,
    other_face_images,
    your_face_labels,
    other_face_labels,
    epochs,
):
    train_model(
        face_recognition_model,
        np.concatenate((your_face_images, other_face_images)),
        np.concatenate((your_face_labels, other_face_labels)),
        epochs,
    )


def capture_and_process_frames(cap, face_detection_model, face_recognition_model):
    while True:
        ret, original_frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        frame = preprocess_frame(original_frame, (64, 64))
        face_label = perform_inference(frame, face_detection_model)

        if face_label == 1:
            frame = preprocess_frame(original_frame, (64, 64))
            your_face_label = perform_inference(frame, face_recognition_model)
            display_inference(original_frame, your_face_label)

        cv2.imshow("Live Inference", original_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    input_shape = (64, 64, 3)
    num_classes = 2
    epochs = 10

    face_detection_model, face_recognition_model = create_and_compile_models(
        input_shape, num_classes
    )

    face_images, face_labels = load_images("datasets/face_images", (64, 64), 1)
    no_face_images, no_face_labels = load_images("datasets/no_face_images", (64, 64), 0)
    your_face_images, your_face_labels = load_images(
        "datasets/my_face_images", (64, 64), 1
    )
    other_face_images, other_face_labels = load_images(
        "datasets/other_face_images", (64, 64), 1
    )

    train_face_detection_model(
        face_detection_model,
        face_images,
        no_face_images,
        face_labels,
        no_face_labels,
        epochs,
    )

    train_face_recognition_model(
        face_recognition_model,
        your_face_images,
        other_face_images,
        your_face_labels,
        other_face_labels,
        epochs,
    )

    cap = cv2.VideoCapture(0)

    # Create a GUI window
    root = tk.Tk()
    root.title("Face Detection and Recognition")
    root.geometry("800x600")

    # Create a canvas to display the video feed
    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack()

    def update_frame():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((640, 480), Image.ANTIALIAS)
        frame = ImageTk.PhotoImage(frame)
        canvas.create_image(0, 0, anchor=tk.NW, image=frame)
        canvas.image = frame
        canvas.after(1, update_frame)

    update_frame()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
