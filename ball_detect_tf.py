import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    "path/to/dataset", target_size=(224, 224), class_mode="binary", subset="training"
)
validation_generator = datagen.flow_from_directory(
    "path/to/dataset", target_size=(224, 224), class_mode="binary", subset="validation"
)

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save("path/to/save/model")
