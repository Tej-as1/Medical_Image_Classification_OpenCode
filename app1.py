import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

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
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        margin-bottom: 30px;
    }
    .upload-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 30px;
    }
    .prediction-results {
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app code
st.title("MedZ: Deep Learning based Medical Image Classifier")
st.markdown("---")

st.markdown('<p class="title">Upload an image to predict normal or infected.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image with border
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    # Perform prediction
    if st.button("Predict"):
        result = predict(uploaded_file)
        progress_bar = st.progress(0)
        status_text = st.empty()
          
         
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f'Progress: {i}%')
            time.sleep(0.00)
        
        status_text.text('Done!')
        
        # Display the prediction results
        st.write("Prediction Results:")
        prediction_label = "Normal" if result >= 0.5 else "Infected"
        if prediction_label == "Normal":
            st.balloons()
            st.success(f"The image is predicted as {prediction_label}")
        else :
            st.snow()
            st.warning(f'Warning : The image is predicted as {prediction_label}', icon="⚠️")
            
st.markdown("---")
st.write("Developed by Tejas Sharma")
st.write("Copyright © 2023. All rights reserved.")
