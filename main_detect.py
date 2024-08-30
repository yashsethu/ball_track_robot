import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image


def create_model(input_shape, num_classes):
    """
    Create a convolutional neural network model for image classification.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of classes for classification.

    Returns:
        tf.keras.Sequential: The created model.
    """
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
    """
    Compile the given model with optimizer, loss function, and metrics.

    Args:
        model (tf.keras.Sequential): The model to be compiled.

    Returns:
        tf.keras.Sequential: The compiled model.
    """
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_images(image_dir, target_size, label):
    """
    Load images from the given directory and resize them to the target size.

    Args:
        image_dir (str): The directory containing the images.
        target_size (tuple): The target size for image resizing.
        label (int): The label for the loaded images.

    Returns:
        numpy.ndarray: The loaded images as a numpy array.
        numpy.ndarray: The labels for the loaded images as a numpy array.
    """
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
    """
    Train the given model with the provided images and labels for the specified number of epochs.

    Args:
        model (tf.keras.Sequential): The model to be trained.
        images (numpy.ndarray): The input images for training.
        labels (numpy.ndarray): The labels for the input images.
        epochs (int): The number of epochs for training.
    """
    model.fit(images, labels, epochs=epochs)


def preprocess_frame(frame, target_size):
    """
    Preprocess a frame by resizing and normalizing it.

    Args:
        frame (numpy.ndarray): The input frame.
        target_size (tuple): The target size for frame resizing.

    Returns:
        numpy.ndarray: The preprocessed frame.
    """
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


def perform_inference(frame, model):
    """
    Perform inference on a frame using the given model.

    Args:
        frame (numpy.ndarray): The input frame.
        model (tf.keras.Sequential): The model for inference.

    Returns:
        int: The predicted label for the frame.
    """
    result = model.predict(frame)
    label = np.argmax(result)
    return label


def display_inference(frame, label):
    """
    Display the inference result on the frame.

    Args:
        frame (numpy.ndarray): The input frame.
        label (int): The label for the frame.
    """
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
    """
    Create and compile two models for face detection and face recognition.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of classes for classification.

    Returns:
        tf.keras.Sequential: The face detection model.
        tf.keras.Sequential: The face recognition model.
    """
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
    """
    Train the face detection model with face and no-face images.

    Args:
        face_detection_model (tf.keras.Sequential): The face detection model.
        face_images (numpy.ndarray): The face images for training.
        no_face_images (numpy.ndarray): The no-face images for training.
        face_labels (numpy.ndarray): The labels for the face images.
        no_face_labels (numpy.ndarray): The labels for the no-face images.
        epochs (int): The number of epochs for training.
    """
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
    """
    Train the face recognition model with your face and other face images.

    Args:
        face_recognition_model (tf.keras.Sequential): The face recognition model.
        your_face_images (numpy.ndarray): Your face images for training.
        other_face_images (numpy.ndarray): Other face images for training.
        your_face_labels (numpy.ndarray): The labels for your face images.
        other_face_labels (numpy.ndarray): The labels for other face images.
        epochs (int): The number of epochs for training.
    """
    train_model(
        face_recognition_model,
        np.concatenate((your_face_images, other_face_images)),
        np.concatenate((your_face_labels, other_face_labels)),
        epochs,
    )


def capture_and_process_frames(cap, face_detection_model, face_recognition_model):
    """
    Capture and process frames from the camera feed.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        face_detection_model (tf.keras.Sequential): The face detection model.
        face_recognition_model (tf.keras.Sequential): The face recognition model.
    """
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
    """
    The main function to run the face detection and recognition system.
    """
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

    capture_and_process_frames(cap, face_detection_model, face_recognition_model)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
