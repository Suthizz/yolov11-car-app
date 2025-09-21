import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.title("YOLO Car Detection App :)")

# Load YOLO model
# Check if the model file exists at the specified path
model_path = "runs/detect/train4/weights/best.pt"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    st.info(f"Loading model from: {os.path.abspath(model_path)}")
    model = YOLO(model_path)

    # Upload image
    uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
      # Show original image
      st.image(uploaded_image , caption="Uploaded Image", use_container_width=True)

      # Read image and convert to numpy array
      image = Image.open(uploaded_image)
      image_np = np.array(image)

      # Run YOLO inference
      st.info("Running YOLO object detection...")
      results = model.predict(image_np , conf=0.4)

      # Draw results on image
      result_image = results[0].plot()
      st.image(result_image , caption="YOLO Detection Result", use_container_width=True)
      st.success("Detection completed!")

      # Extract detection results
      boxes = results[0].boxes
      class_ids = boxes.cls.cpu().numpy().astype(int)
      class_names = [model.names[i] for i in class_ids]

      # Count people
      car_count = class_names.count("car")
      st.write(f"Number of cars detected: **{car_count}**")
