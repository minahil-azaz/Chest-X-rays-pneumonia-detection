import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About this App")
st.sidebar.info(
    """
    **AI Pneumonia Detection**  
    Upload a Chest X-ray image, and the AI model will predict if it shows **Pneumonia** or **Normal** lungs.  
    Predictions are saved to a local SQLite database.
    """
)

# -----------------------------
# Main Title
# -----------------------------
st.markdown("<h1 style='text-align: center; color: teal;'>🫁 Pneumonia Detection from Chest X-ray</h1>", unsafe_allow_html=True)
st.write("Upload a Chest X-ray image below:")

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("pneumonia_model.h5", compile=False)
IMG_SIZE = (224,224)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an X-ray Image (jpg, png, jpeg)",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image.resize(IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    confidence = float(prediction)

    # -----------------------------
    # Display result with color and confidence
    # -----------------------------
    if confidence > 0.5:
        result = "PNEUMONIA"
        st.markdown(f"<h3 style='color:red;'>Prediction: {result}</h3>", unsafe_allow_html=True)
        st.progress(confidence)
    else:
        result = "NORMAL"
        st.markdown(f"<h3 style='color:green;'>Prediction: {result}</h3>", unsafe_allow_html=True)
        st.progress(1-confidence)

    # -----------------------------
    # Save to SQLite
    # -----------------------------
    conn = sqlite3.connect("pneumonia_app.db")
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL
    )
    ''')

    # Insert prediction
    c.execute("INSERT INTO predictions (filename, prediction, confidence) VALUES (?, ?, ?)",
              (uploaded_file.name, result, confidence))
    conn.commit()
    conn.close()

    st.success("✅ Prediction saved to database!")

# -----------------------------
# Show past predictions
# -----------------------------
if st.sidebar.checkbox("Show past predictions"):
    conn = sqlite3.connect("pneumonia_app.db")
    import pandas as pd
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    st.sidebar.write(df)
    conn.close()
