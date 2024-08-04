import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph and label map
PATH_TO_MODEL = "path/to/frozen_inference_graph.pb"
PATH_TO_LABELS = "path/to/label_map.pbtxt"

# Load the frozen detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

# Load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=1, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


# Function to detect objects in an image
def detect_objects(image):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Preprocess the image
            image_np = np.array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Get input and output tensors
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            scores = detection_graph.get_tensor_by_name("detection_scores:0")
            classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            # Run the inference
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

            return image_np


# Load and process an image
image = cv2.imread("path/to/image.jpg")
output_image = detect_objects(image)

# Display the output image
cv2.imshow("Object Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
