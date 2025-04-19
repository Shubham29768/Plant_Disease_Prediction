import streamlit as st
import joblib
import numpy as np
from PIL import Image
import json
import os

# Load the model and class indices
@st.cache_resource
def load_model():
    model = joblib.load("plant_disease_prediction_model.joblib")
    return model

@st.cache_data
def load_class_indices():
    with open("class_indices.json", "r") as f:
        return json.load(f)

model = load_model()
class_indices = load_class_indices()

# Preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    class_index = int(np.argmax(prediction, axis=1)[0])
    return class_indices[str(class_index)]

# Streamlit UI
st.title("🌿 Plant Disease Classifier")
uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("🔍 Classify"):
        result = predict(image)
        st.success(f"🌱 Prediction: **{result}**")
