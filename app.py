import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢"
)
st.title("Handwritten Digit Classifier")
st.write(
    "Upload an image of a handwritten digit (0–9) and let the AI predict it."
)
model = load_model("models/mnist_ann.keras")
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)
    # Automatically detect background
    # If the image is mostly bright, invert it
    if image_array.mean() > 127:
       image_array = 255 - image_array
    # Display processed image
    st.write("Processed Image (28×28):")
    st.image(image_array, width=150)