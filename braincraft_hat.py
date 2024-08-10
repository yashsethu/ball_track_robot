import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the pre-trained model
model_path = "/Users/Yash/Documents/GitHub/ball_track_robot/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
label_path = "/Users/Yash/Documents/GitHub/ball_track_robot/mscoco_label_map.pbtxt"
num_classes = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    model = tf.saved_model.load(model_path)

# Load the label map
label_map = label_map_util.load_labelmap(label_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

# Open the webcam
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            scores = detection_graph.get_tensor_by_name("detection_scores:0")
            classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            # Perform object detection
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )

            # Visualize the results
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
            )

            # Display the resulting image
            cv2.imshow("Object Detection", cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
