import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache the model loading for performance
@st.cache_data()
def load_model():
    # Replace 'waste_model.keras' with your model file name and format if needed.
    model = tf.keras.models.load_model('C:/Users/welcome/OneDrive/Desktop/waste classfication/my_model.keras')

    return model

# Load the saved model
model = load_model()

# Define your two class labels
class_labels = {
    0: 'Organic Waste',
    1: 'Recycle Waste'
}

# ----------------------------
# Sidebar: About the Project
# ----------------------------
st.sidebar.title("About the Project")
st.sidebar.info(
    """
    This project is a waste management classifier that uses a Convolutional Neural Network (CNN) 
    to differentiate between Organic Waste and Recycle Waste. Upload an image of waste, and the classifier 
    will predict which category it belongs to.
    """
)
st.sidebar.title("Created BY")
st.sidebar.info("  JEEVA J ,BTech Artificial Intelligence and Data Science")

# ----------------------------
# Main Page: Title and Instructions
# ----------------------------
# Set the title
st.title("Waste Management Classifier")


st.write("Upload an image of waste to classify it as either **Organic Waste** or **Recycle Waste**.")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width =True)
    st.write("Classifying...")

    # ----------------------------
    # Preprocessing
    # ----------------------------
    # Resize the image to match the input size expected by your CNN.
    target_size = (224, 224)  # Adjust this based on your model's expected input size.
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized)

    # Convert images with an alpha channel (RGBA) to RGB.
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    # Normalize the image if your model was trained on normalized data.
    image_array = image_array / 255.0

    # Add a batch dimension since the model expects a batch of images.
    image_array = np.expand_dims(image_array, axis=0)

    # ----------------------------
    # Prediction
    # ----------------------------
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels.get(predicted_index, "Unknown")
    confidence = np.max(predictions)

    # Display the prediction and confidence level
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Prediction Confidence:** {confidence:.2f}")

# ----------------------------
# Footer: Created By
# ----------------------------

