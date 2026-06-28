import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "mnist_model.keras"
)

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Handwritten Digit Recognizer")

st.markdown("""
Upload an image containing a handwritten digit (0–9).

**Model:** Neural Network (TensorFlow/Keras)  
**Dataset:** MNIST Dataset  
**Accuracy:** 98.66%
""")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")

    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    image = image.resize((28, 28))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = image_array.reshape(
        1,
        28,
        28
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(
        f"Predicted Digit: {digit}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

st.divider()

st.subheader("About")

st.write(
    "This project uses a deep learning model trained on the MNIST handwritten digit dataset."
)