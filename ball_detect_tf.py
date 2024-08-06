import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import os


# Function to capture and save images
def capture_data():
    # Create the directory if it doesn't exist
    if not os.path.exists("dataset/ball_images"):
        os.makedirs("dataset/ball_images")

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Capture and save images
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow("Capture", frame)

        # Press 's' to save the image
        if cv2.waitKey(1) & 0xFF == ord("s"):
            filename = f"dataset/ball_images/image_{count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image {count} saved.")
            count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


# Load a pre-trained model
base_model = MobileNetV2(weights="imagenet", include_top=False)

# Add a new top layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation="sigmoid")(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First, we will only train the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
)

# Prepare the dataset
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), class_mode="binary", subset="training"
)
validation_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), class_mode="binary", subset="validation"
)

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save("path/to/save/model")

# Load the saved model
loaded_model = model

# Capture and add data to the dataset
capture_data()

# Live inferencing
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = tf.expand_dims(normalized_frame, axis=0)

    # Perform inference
    predictions = loaded_model.predict(input_data)
    confidence = predictions[0][0]

    # Draw box around ball
    if confidence > 0.5:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

    # Display confidence percentage
    confidence_percentage = round(confidence * 100, 2)
    cv2.putText(
        frame,
        f"Confidence: {confidence_percentage}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display the frame
    cv2.imshow("Live Inference", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
