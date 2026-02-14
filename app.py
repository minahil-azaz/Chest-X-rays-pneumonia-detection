# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# st.sidebar.title("About")
# st.sidebar.info("AI Pneumonia Detection using Deep Learning")

# # Load model
# model = tf.keras.models.load_model("pneumonia_model.keras", compile=False)

# IMG_SIZE = (224, 224)

# st.title("Pneumonia Detection from Chest X-ray")
# st.write("Upload a Chest X-ray image to detect Pneumonia")

# uploaded_file = st.file_uploader("Choose an X-ray Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess
#     img = np.array(image.resize(IMG_SIZE)) / 255.0
#     img = np.expand_dims(img, axis=0)

#     # Predict
#     prediction = model.predict(img)[0][0]
#     confidence = float(prediction)

#     if confidence > 0.5:
#         st.error(f"Prediction: PNEUMONIA ({confidence*100:.2f}% confidence)")
#         st.progress(confidence)
#     else:
#         st.success(f"Prediction: NORMAL ({(1-confidence)*100:.2f}% confidence)")
#         st.progress(1-confidence)

import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("pneumonia_model.keras", compile=False)
