import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pathlib import Path
import matplotlib.pyplot as plt


def load_model(model_path):
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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(len(scores[0])), scores[0])
    plt.xlabel("Object Index")
    plt.ylabel("Score")
    plt.title("Object Detection Scores")


def plot_classes(classes, category_index):
    plt.subplot(1, 3, 2)
    class_names = [category_index[class_id]["name"] for class_id in classes[0]]
    plt.barh(range(len(class_names)), scores[0])
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Score")
    plt.ylabel("Class")
    plt.title("Object Detection Classes")


def plot_bounding_boxes(image_np, boxes, classes, scores, category_index):
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
cap = cv2.VideoCapture(0)
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
