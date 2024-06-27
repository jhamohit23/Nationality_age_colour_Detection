import streamlit as st
import torch
from app.utils import load_model, preprocess_image, predict

# Configuration for the Streamlit app
st.title("Multi-Task Prediction")
st.write("This app predicts nationality, emotion, age, and dress color from an image.")

# Load the pre-trained model
model_path = 'models/multitask_model.pth'
num_nationalities = 5  # Replace with actual number
num_emotions = 7       # Replace with actual number
num_dress_colors = 10  # Replace with actual number

model = load_model(model_path, num_nationalities, num_emotions, num_dress_colors)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image_tensor = preprocess_image(uploaded_file)

    # Make predictions
    nationality, emotion, age, dress_color = predict(model, image_tensor)

    # Decode the labels (replace with actual labels)
    nationalities = ['Nationality1', 'Nationality2', 'Nationality3', 'Nationality4', 'Nationality5']
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Disgust', 'Fear']
    dress_colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'White', 'Purple', 'Orange', 'Brown', 'Pink']

    # Display results
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Predicted Nationality:", nationalities[nationality])
    st.write("Predicted Emotion:", emotions[emotion])
    st.write("Predicted Age:", int(age))
    st.write("Predicted Dress Color:", dress_colors[dress_color])
