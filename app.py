import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import sqlite3
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CONSTANTS
IMG_SIZE = (224, 224)
MODEL_PATH = "final_model.h5"
DB_PATH = "pneumonia_predictions.db"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
PREDICTION_THRESHOLD = 0.65

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1e88e5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .normal-box {
        background-color: #e8f5e9;
        border: 3px solid #4caf50;
    }
    .pneumonia-box {
        background-color: #ffebee;
        border: 3px solid #f44336;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e88e5;
    }
    .info-box {
        background-color: #E328;
        border: 2px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Database functions
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            raw_probability REAL NOT NULL,
            threshold_used REAL NOT NULL,
            image_mode TEXT,
            image_size TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(filename, prediction, confidence, raw_prob, threshold, img_mode, img_size):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """INSERT INTO predictions 
            (filename, prediction, confidence, raw_probability, threshold_used, image_mode, image_size) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (filename, prediction, float(confidence), float(raw_prob), float(threshold), img_mode, img_size)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def get_all_predictions():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def get_statistics():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]
        c.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
        counts = dict(c.fetchall())
        c.execute("SELECT AVG(confidence) FROM predictions")
        avg_conf = c.fetchone()[0] or 0
        conn.close()
        return {
            'total': total,
            'normal': counts.get('NORMAL', 0),
            'pneumonia': counts.get('PNEUMONIA', 0),
            'avg_confidence': avg_conf
        }
    except:
        return {'total': 0, 'normal': 0, 'pneumonia': 0, 'avg_confidence': 0}

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.stop()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def diagnose_image(image):
    """Diagnose image properties for debugging"""
    info = {
        'mode': image.mode,
        'size': f"{image.size[0]}x{image.size[1]}",
        'format': image.format,
        'aspect_ratio': image.size[0] / image.size[1]
    }
    
    # Check brightness (for inversion detection)
    gray = image.convert('L')
    mean_brightness = np.array(gray).mean()
    info['mean_brightness'] = mean_brightness
    info['is_inverted'] = mean_brightness > 128
    
    # Check if grayscale
    info['is_grayscale'] = image.mode in ['L', 'LA']
    
    # Check aspect ratio
    info['is_square'] = 0.9 < info['aspect_ratio'] < 1.1
    
    return info

def preprocess_image_robust(image, show_steps=False):
    """
    Robust preprocessing for X-ray images
    Handles: grayscale, different sizes, aspect ratios, color inversions
    """
    steps_info = {}
    
    try:
        # Store original info
        original_mode = image.mode
        original_size = image.size
        steps_info['original'] = f"{original_mode}, {original_size}"
        
        # Step 1: Handle different color modes
        if image.mode == 'L':  # Grayscale
            image = image.convert('RGB')
            steps_info['conversion'] = "Grayscale → RGB"
        elif image.mode == 'LA':  # Grayscale with alpha
            image = image.convert('RGB')
            steps_info['conversion'] = "Grayscale+Alpha → RGB"
        elif image.mode == 'RGBA':  # RGB with alpha
            # Create black background and paste image
            background = Image.new('RGB', image.size, (0, 0, 0))
            background.paste(image, mask=image.split()[3])
            image = background
            steps_info['conversion'] = "RGBA → RGB (removed alpha)"
        elif image.mode == 'CMYK':
            image = image.convert('RGB')
            steps_info['conversion'] = "CMYK → RGB"
        elif image.mode == 'P':  # Palette mode
            image = image.convert('RGB')
            steps_info['conversion'] = "Palette → RGB"
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            steps_info['conversion'] = f"{original_mode} → RGB"
        else:
            steps_info['conversion'] = "Already RGB"
        
        # Step 2: Check for color inversion (X-rays should be dark background)
        gray = image.convert('L')
        mean_brightness = np.array(gray).mean()
        
        if mean_brightness > 128:
            # Image is inverted (bright background)
            image = ImageOps.invert(image.convert('RGB'))
            steps_info['inversion'] = f"Inverted (brightness: {mean_brightness:.1f})"
        else:
            steps_info['inversion'] = f"Not inverted (brightness: {mean_brightness:.1f})"
        
        # Step 3: Resize with aspect ratio preservation and padding
        width, height = image.size
        aspect_ratio = width / height
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = 224
            new_height = int((224 / width) * height)
        else:
            new_height = 224
            new_width = int((224 / height) * width)
        
        # Ensure minimum size
        if new_width < 1:
            new_width = 1
        if new_height < 1:
            new_height = 1
        
        # Resize with high-quality filter
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
        steps_info['resize'] = f"{original_size} → {new_width}x{new_height}"
        
        # Step 4: Add padding to make it square
        padded_image = Image.new('RGB', (224, 224), (0, 0, 0))
        paste_x = (224 - new_width) // 2
        paste_y = (224 - new_height) // 2
        padded_image.paste(image_resized, (paste_x, paste_y))
        steps_info['padding'] = f"Padded to 224x224 (offset: {paste_x}, {paste_y})"
        
        # Step 5: Convert to array and normalize for MobileNetV2
        img_array = np.array(padded_image)
        
        # MobileNetV2 preprocessing: scale to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        steps_info['normalization'] = "Scaled to [-1, 1]"
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        steps_info['batch'] = f"Shape: {img_array.shape}"
        
        if show_steps:
            return img_array, steps_info, padded_image
        return img_array
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def make_prediction(image, model, threshold=PREDICTION_THRESHOLD):
    """Make prediction with robust preprocessing"""
    try:
        # Get original image info
        img_info = diagnose_image(image)
        
        # Preprocess with diagnostic info
        processed, steps_info, processed_image = preprocess_image_robust(image, show_steps=True)
        
        if processed is None:
            return None, None, None, None, None, None
        
        # Make prediction
        raw_prediction = model.predict(processed, verbose=0)[0][0]
        
        # Determine class and confidence
        if raw_prediction > threshold:
            prediction = "PNEUMONIA"
            confidence = raw_prediction
        else:
            prediction = "NORMAL"
            confidence = 1 - raw_prediction
        
        return prediction, confidence, raw_prediction, img_info, steps_info, processed_image
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None, None, None

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': '#ffcdd2'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def main():
    init_database()
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
        st.title("About This App")
        
        st.markdown("""
        ### 🫁 AI Pneumonia Detection
        
        **Enhanced with Robust Image Processing**
        
        This app now handles:
        - ✅ Grayscale images
        - ✅ Different sizes/shapes
        - ✅ Color inversions
        - ✅ Different formats
        
        ⚠️ **For educational purposes only**
        """)
        
        st.divider()
        
        st.subheader("⚙️ Settings")
        custom_threshold = st.slider(
            "Prediction Threshold",
            min_value=0.50,
            max_value=0.80,
            value=0.65,
            step=0.05,
            help="Higher = fewer pneumonia predictions"
        )
        st.session_state['threshold'] = custom_threshold
        
        st.divider()
        
        stats = get_statistics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats['total'])
            st.metric("Normal", stats['normal'])
        with col2:
            st.metric("Confidence", f"{stats['avg_confidence']:.1%}")
            st.metric("Pneumonia", stats['pneumonia'])
    
    # Main content
    st.markdown('<div class="main-header">🫁 Pneumonia Detection AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Robust preprocessing for all X-ray image formats</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>✨ Enhanced Image Processing</strong><br>
        This app automatically handles grayscale images, different sizes, color inversions, 
        and maintains aspect ratios. Upload any chest X-ray image format!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supports JPG, PNG, BMP, TIFF formats"
        )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Get threshold
            threshold = st.session_state.get('threshold', PREDICTION_THRESHOLD)
            
            # Make prediction
            with st.spinner("Analyzing image with robust preprocessing..."):
                prediction, confidence, raw_prob, img_info, steps_info, processed_img = make_prediction(
                    image, model, threshold
                )
            
            if prediction is not None:
                # Create layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
                    
                    # Show image diagnostics
                    with st.expander("📊 Image Diagnostics", expanded=False):
                        st.write("**Original Image Properties:**")
                        st.write(f"- Format: `{img_info['mode']}`")
                        st.write(f"- Size: `{img_info['size']}`")
                        st.write(f"- Aspect Ratio: `{img_info['aspect_ratio']:.2f}`")
                        st.write(f"- Mean Brightness: `{img_info['mean_brightness']:.1f}`")
                        
                        if img_info['is_grayscale']:
                            st.warning("⚠️ Grayscale image detected (converted to RGB)")
                        if img_info['is_inverted']:
                            st.warning("⚠️ Inverted colors detected (corrected)")
                        if not img_info['is_square']:
                            st.info("ℹ️ Non-square image (padded to maintain aspect ratio)")
                    
                    # Show preprocessing steps
                    with st.expander("🔧 Preprocessing Steps", expanded=False):
                        for step, info in steps_info.items():
                            st.write(f"**{step.title()}:** {info}")
                        
                        st.subheader("Processed Image:")
                        st.image(processed_img, caption="Ready for model input (224x224)", 
                                use_column_width=True)
                
                with col2:
                    st.subheader("🔬 Analysis Results")
                    
                    # Display prediction
                    if prediction == "NORMAL":
                        st.markdown(f"""
                        <div class="prediction-box normal-box">
                            <h2>✅ NORMAL</h2>
                            <p style="font-size: 1.2rem;">No signs of pneumonia detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box pneumonia-box">
                            <h2>⚠️ PNEUMONIA DETECTED</h2>
                            <p style="font-size: 1.2rem;">Please consult a medical professional</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Prediction", prediction)
                    with metric_col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with metric_col3:
                        st.metric("Raw Prob", f"{raw_prob:.3f}")
                    
                    # Gauge
                    st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
                    
                    # Explanation
                    st.info(f"""
                    **Decision Logic:**
                    - Raw output: `{raw_prob:.4f}`
                    - Threshold: `{threshold:.2f}`
                    - Result: `{prediction}` (because {raw_prob:.4f} {'>' if raw_prob > threshold else '≤'} {threshold:.2f})
                    """)
                    
                    # Save to database
                    if save_prediction(
                        uploaded_file.name, prediction, confidence, raw_prob, 
                        threshold, img_info['mode'], img_info['size']
                    ):
                        st.success("✅ Prediction saved!")
                    
                    # Download report
                    result_text = f"""
PNEUMONIA DETECTION REPORT
{'='*60}

ORIGINAL IMAGE:
  Filename: {uploaded_file.name}
  Format: {img_info['mode']}
  Size: {img_info['size']}
  Aspect Ratio: {img_info['aspect_ratio']:.2f}
  Mean Brightness: {img_info['mean_brightness']:.1f}

PREPROCESSING:
  Grayscale Detected: {img_info['is_grayscale']}
  Inverted Colors: {img_info['is_inverted']}
  Square Image: {img_info['is_square']}

PREDICTION RESULTS:
  Prediction: {prediction}
  Confidence: {confidence:.2%}
  Raw Probability: {raw_prob:.4f}
  Threshold Used: {threshold:.2f}
  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
This prediction used robust image preprocessing to handle
various image formats and ensure accurate results.

DISCLAIMER: For educational purposes only.
Consult a medical professional for accurate diagnosis.
                    """
                    
                    st.download_button(
                        label="📥 Download Detailed Report",
                        data=result_text,
                        file_name=f"xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            st.info("Please ensure you uploaded a valid image file.")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")
        
        st.divider()
        
        st.subheader("📋 Supported Image Formats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Supported:**
            - ✅ Grayscale images (L, LA)
            - ✅ RGB images
            - ✅ RGBA (with transparency)
            - ✅ Different sizes (any resolution)
            - ✅ Non-square aspect ratios
            - ✅ Inverted colors
            """)
        with col2:
            st.markdown("""
            **Formats:**
            - ✅ JPEG / JPG
            - ✅ PNG
            - ✅ BMP
            - ✅ TIFF
            - ✅ Any PIL-supported format
            """)
    
    # History
    st.divider()
    st.subheader("📜 Prediction History")
    
    df = get_all_predictions()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions yet. Upload an X-ray to get started!")
    
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Powered by TensorFlow & MobileNetV2 | Enhanced Image Processing v2.0</p>
        <p style="font-size: 0.9rem;">⚠️ For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
