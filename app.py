import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢"
)

# Main title
st.title("🔢 Handwritten Digit Classifier")

st.write(
    "Upload an image of a handwritten digit (0–9) and let the AI predict it."
)

# Sidebar
st.sidebar.header("About")

st.sidebar.write(
    """
    **Dataset:** MNIST

    **Model:** Artificial Neural Network (ANN)

    **Framework:** TensorFlow / Keras

    **Frontend:** Streamlit
    """
)

# Load trained model
model = load_model("models/mnist_ann.keras")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

# Prediction block
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display original image
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
    st.image(image_array, width=150, clamp=True)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension for the model
    image_array = image_array.reshape(1, 28, 28)
    # Make prediction
    prediction = model.predict(image_array)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"🎯 Predicted Digit: {predicted_digit}")

    st.metric("Confidence", f"{confidence:.2f}%")

    # Probability chart
    st.subheader("📊 Prediction Probabilities")

    probabilities = prediction[0]

    st.bar_chart(probabilities)