import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pathlib import Path
import matplotlib.pyplot as plt


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Call the functions
collect_data()
preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)
train_models(X_train, y_train)


# Function to train models
def train_models(X_train, y_train):
    # Model 1: Linear Regression
    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    # Model 2: Decision Tree Regressor
    model2 = DecisionTreeRegressor()
    model2.fit(X_train, y_train)

    # Model 3: Random Forest Regressor
    model3 = RandomForestRegressor()
    model3.fit(X_train, y_train)

    # Model 4: Support Vector Regressor
    model4 = SVR()
    model4.fit(X_train, y_train)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color="blue", label="Actual")
    plt.plot(X_train, model1.predict(X_train), color="red", label="Linear Regression")
    plt.plot(X_train, model2.predict(X_train), color="green", label="Decision Tree")
    plt.plot(X_train, model3.predict(X_train), color="orange", label="Random Forest")
    plt.plot(X_train, model4.predict(X_train), color="purple", label="Support Vector")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Regression Models")
    plt.legend()
    plt.show()


# Call the functions
collect_data()
preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)
train_models(X_train, y_train)
live_inference()


# Function for live inferencing
def live_inference():
    cap = cv2.VideoCapture(0)  # Open the default camera
    while True:
        ret, frame = cap.read()  # Read the frame from the camera
        # Preprocess the frame if needed
        # Perform inference using the trained models
        # Display the results on the frame
        cv2.imshow("Live Inference", frame)  # Display the frame with results
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit if 'q' is pressed
            break
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows


# Call the functions
collect_data()
preprocess_data()
X_train, y_train = split_data()
train_models(X_train, y_train)
live_inference()


def load_model(model_path):
    """
    Load the TensorFlow model from the given path.

    Args:
        model_path (Path): Path to the saved model directory.

    Returns:
        Tuple: Tuple containing the TensorFlow session and other required tensors.
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        model = tf.saved_model.load(str(model_path))
        with tf.Session(graph=detection_graph) as sess:
            image_np_expanded = np.empty((1, 800, 600, 3), dtype=np.uint8)
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            scores = detection_graph.get_tensor_by_name("detection_scores:0")
            classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")
            return (
                sess,
                image_np_expanded,
                image_tensor,
                boxes,
                scores,
                classes,
                num_detections,
            )


def filter_detections(boxes, scores, classes, num_detections, min_score_thresh):
    """
    Filter the detected objects based on the confidence score threshold.

    Args:
        boxes (np.ndarray): Detected bounding boxes.
        scores (np.ndarray): Confidence scores of the detections.
        classes (np.ndarray): Class labels of the detections.
        num_detections (np.ndarray): Number of detections.
        min_score_thresh (float): Minimum confidence score threshold.

    Returns:
        List: List of filtered detections.
    """
    detections = []
    for i in range(int(num_detections[0])):
        if scores[0][i] > min_score_thresh:
            detections.append(
                {
                    "box": boxes[0][i],
                    "score": scores[0][i],
                    "class": classes[0][i],
                }
            )
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections


def visualize_results(image_np, boxes, classes, scores, category_index):
    """
    Visualize the detected objects on the image.

    Args:
        image_np (np.ndarray): Input image.
        boxes (np.ndarray): Detected bounding boxes.
        classes (np.ndarray): Class labels of the detections.
        scores (np.ndarray): Confidence scores of the detections.
        category_index (dict): Mapping of class IDs to class names.
    """
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
    )


def plot_scores(scores):
    """
    Plot the object detection scores.

    Args:
        scores (np.ndarray): Confidence scores of the detections.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(len(scores[0])), scores[0])
    plt.xlabel("Object Index")
    plt.ylabel("Score")
    plt.title("Object Detection Scores")


def plot_classes(classes, category_index):
    """
    Plot the object detection classes.

    Args:
        classes (np.ndarray): Class labels of the detections.
        category_index (dict): Mapping of class IDs to class names.
    """
    plt.subplot(1, 3, 2)
    class_names = [category_index[class_id]["name"] for class_id in classes[0]]
    plt.barh(range(len(class_names)), scores[0])
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Score")
    plt.ylabel("Class")
    plt.title("Object Detection Classes")


def plot_bounding_boxes(image_np, boxes, classes, scores, category_index):
    """
    Plot the object detection bounding boxes.

    Args:
        image_np (np.ndarray): Input image.
        boxes (np.ndarray): Detected bounding boxes.
        classes (np.ndarray): Class labels of the detections.
        scores (np.ndarray): Confidence scores of the detections.
        category_index (dict): Mapping of class IDs to class names.
    """
    plt.subplot(1, 3, 3)
    image_with_boxes = image_np.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_with_boxes,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
    )
    plt.imshow(image_with_boxes)
    plt.title("Object Detection Bounding Boxes")


# Load the pre-trained model
model_path = Path(
    "/Users/Yash/Documents/GitHub/ball_track_robot/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
)
label_path = Path(
    "/Users/Yash/Documents/GitHub/ball_track_robot/mscoco_label_map.pbtxt"
)
num_classes = 90

# Load the label map
label_map = label_map_util.load_labelmap(str(label_path))
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

# Open the webcam
with cv2.VideoCapture(0) as cap:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # Load the model and create the session
    sess, image_np_expanded, image_tensor, boxes, scores, classes, num_detections = (
        load_model(model_path)
    )

    while True:
        ret, image_np = cap.read()

        # Copy the image to the fixed-size numpy array
        np.copyto(image_np_expanded[0], image_np)

        # Perform object detection
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded},
        )

        # Filter objects based on confidence score
        min_score_thresh = 0.5
        detections = filter_detections(
            boxes, scores, classes, num_detections, min_score_thresh
        )

        # Print the top 3 detected objects
        top_3_detections = detections[:3]
        for i, detection in enumerate(top_3_detections):
            class_name = category_index[detection["class"]]["name"]
            print(f"Top {i+1} detection: {class_name} (Score: {detection['score']})")

        # Visualize the results
        visualize_results(image_np, boxes, classes, scores, category_index)

        # Display the resulting image
        cv2.imshow("Object Detection", image_np)

        # Plot the scores
        plot_scores(scores)

        # Plot the classes
        plot_classes(classes, category_index)

        # Plot the bounding boxes
        plot_bounding_boxes(image_np, boxes, classes, scores, category_index)

        plt.tight_layout()
        plt.show()

        if cv2.waitKey(1) == ord("q"):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
