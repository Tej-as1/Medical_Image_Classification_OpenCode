import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('CNN_model.h5')

# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((256,256))  # Resize image to match model input size
    img_array = np.array(img)  # Convert image to NumPy array
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    return predictions

# Streamlit app code
st.title("Image Classification with Deep Learning")
st.write("Upload an image and let the model predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Perform prediction
    with st.spinner('Predicting...'):
        result = predict(uploaded_file)

    # Define classes (change these according to your model)
    classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    
    # Display the prediction results
    st.subheader("Prediction Results:")
    class_index = np.argmax(result)
    st.write(f"Predicted Class: {classes[class_index]}")
    st.write(f"Confidence: {result[0][class_index]*100:.2f}%")
