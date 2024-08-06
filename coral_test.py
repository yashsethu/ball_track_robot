import cv2
import numpy as np
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the Edge TPU model
model_path = "/Users/Yash/Documents/GitHub/ball_track_robot/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load the pre-trained YOLO model
model_path = "/path/to/yolo.weights"
config_path = "/path/to/yolo.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, model_path)

# Load the class labels
labels_path = "/path/to/yolo.names"
with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

# Load the images from the directory
image_dir = "/datasets/ball_images"
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

for image_file in image_files:
    # Read the image
    image = cv2.imread(image_file)

    # Perform object detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Process the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Draw bounding box and label
                box = detection[0:4] * np.array(
                    [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
                )
                (x, y, w, h) = box.astype("int")
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{labels[class_id]}: {confidence:.2f}"
                cv2.putText(
                    image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # Display the image
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (300, 300))
    frame = frame / 127.5 - 1
    frame = np.expand_dims(frame, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
