import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import os
import matplotlib.pyplot as plt


def capture_data():
    os.makedirs("dataset/ball_images", exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            filename = f"dataset/ball_images/image_{count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image {count} saved.")
            count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


base_model = MobileNetV2(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
)

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), class_mode="binary", subset="training"
)
validation_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), class_mode="binary", subset="validation"
)

history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Plotting the training and validation accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Plotting the training and validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

model.save("path/to/save/model")

loaded_model = tf.keras.models.load_model("path/to/save/model")

capture_data()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = tf.expand_dims(normalized_frame, axis=0)

    predictions = loaded_model.predict(input_data)
    confidence = predictions[0][0]

    if confidence > 0.5:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

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

    cv2.imshow("Live Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
