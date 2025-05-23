import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = joblib.load("model.pkl")

st.title("Digit Classifier: Is it a 5 or not?")
st.write("Upload an image of a handwritten digit (either 3 or 5) to check if it's a 5.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    # Convert to grayscale and resize to 28x28
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image_np = np.array(image)
    # Invert colors (assuming white background, black digit)
    image_np = 255 - image_np
    # Flatten the image and scale to match training data
    image_flat = image_np.reshape(1, -1) / 255.0 * 255
    return image_flat

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_data = preprocess_image(image)
    prediction = model.predict(image_data)
    result = "It's a 5!" if prediction[0] == 1 else "It's not a 5."

    st.subheader("Prediction:")
    st.write(result)
