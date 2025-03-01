import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image

# Load model from GitHub
MODEL_URL = "https://github.com/yourusername/yourrepo/raw/main/satellite_land_classification.h5"
MODEL_PATH = "satellite_land_classification.h5"

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

st.title("🌍 Satellite Land Classification")
st.write("Upload an image to classify its land type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_labels = ['Forest', 'Industrial', 'Highway', 'Residential', 'SeaLake', 
                    'Pasture', 'AnnualCrop', 'River', 'HerbaceousVegetation', 'PermanentCrop']
    predicted_class = class_labels[np.argmax(prediction)]

    st.subheader(f"🏷️ Predicted Land Type: {predicted_class}")
