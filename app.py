import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import gdown
import os

file_id="1fwNiM1QIKy6MGCeeSNE9kLdIpAp0sHTw"
url="https://drive.google.com/file/d/1fwNiM1QIKy6MGCeeSNE9kLdIpAp0sHTw/view?usp=sharing"
model_path="potato_leaf_detection_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path,quiet=False)
    

# Load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Class labels
CLASS_NAMES = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

# Prediction function
def predict(image):
    try:
        image = image.convert("RGB")
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (128, 128))
        img_expanded = np.expand_dims(img_resized, axis=0)
        
        predictions = model.predict(img_expanded)
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        
        return f"{predicted_class} ({confidence:.2f}% Confidence)", confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error", 0

# Streamlit UI
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
    st.write("📸 **Or capture a live image:**")
    camera_img = st.camera_input("Take a Photo")
    
    image_source = None
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)
    elif camera_img is not None:
        image_source = Image.open(camera_img)
    
    if image_source is not None:
        st.image(image_source, caption="📌 Uploaded Image", use_column_width=True)
        if st.button("🔍 Predict Disease 🩺"):
            with st.spinner("Analyzing the image... ⏳"):
                time.sleep(2)
                result, confidence = predict(image_source)
                st.success(f"✅ Prediction: {result}")
