import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import streamlit as st
import os
from PIL import Image
import gdown

# Download model from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1nT6T210944mp5zDFd67qwQhKOXE5Zg40"
MODEL_PATH = "land_classification_model.h5"
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load trained model
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['Forest', 'Industrial', 'Highway', 'Residential', 'SeaLake', 'Pasture', 'AnnualCrop', 'River', 'HerbaceousVegetation', 'PermanentCrop']

# Streamlit UI
st.title("Land Classification from Satellite Images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    img = Image.open(image_path)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
