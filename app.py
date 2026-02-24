import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("multiclass_model.h5")

labels = ["Car 🚗", "Human 👤", "Cat 🐱", "Dog 🐶"]

st.title("Multi-Class Image Classifier")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    st.success(f"Prediction: {labels[class_index]}")

    # streamlit run app.py