import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np


st.title("Face Recognition Demo")

activities = ["Webcam", "Upload Image"]
choice = st.sidebar.selectbox("Choose Activity", activities)


if choice == 'Webcam':
    st.header("Using Webcam")

    video_capture = st.camera_input("Capture Video")

    if video_capture is not None:

        img = Image.open(video_capture).convert("RGB")
        img_array = np.array(img)
        result = DeepFace.find(img_array, db_path='data/',
                               enforce_detection=False)[0]
        label = "Unknow"
        if result.empty:
            st.write("No faces found")
        else:
            label = result.iloc[0]['identity']
            st.write("Face recognition: ", label)
else:
    st.header("Upload Image")

    img_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        img = np.array(img)
        result = DeepFace.find(img, db_path='data/',
                               enforce_detection=False)[0]
        label = "Unknow"
        if result.empty:
            st.write("No faces found")
        else:
            label = result.iloc[0]['identity']
            st.write("Face recognition: ", label)
