import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import requests

# Define dataset path
dataset_path = "/content/dataset/2750"

# Define image size and batch size
IMG_SIZE = (64, 64)  # Reduce image size to save memory
BATCH_SIZE = 16  # Reduce batch size to prevent memory crashes

# Use ImageDataGenerator to load images in batches
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training")

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation")

num_classes = len(train_generator.class_indices)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using generators
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save the trained model
model.save("/content/drive/MyDrive/satellite_land_classification.h5")

print("Model training complete and saved to Google Drive!")

# Streamlit App
st.title("🌍 Satellite Land Classification")
st.write("Upload an image to classify its land type.")

MODEL_URL = "https://github.com/yourusername/yourrepo/raw/main/satellite_land_classification.h5"
MODEL_PATH = "satellite_land_classification.h5"

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_labels = ['Forest', 'Industrial', 'Highway', 'Residential', 'SeaLake', 
                    'Pasture', 'AnnualCrop', 'River', 'HerbaceousVegetation', 'PermanentCrop']
    predicted_class = class_labels[np.argmax(prediction)]

    st.subheader(f"🏷️ Predicted Land Type: {predicted_class}")
