# Chest-X-rays-pneumonia-detection
🫁 Pneumonia Detection from Chest X-Ray Images
📌 Project Overview

This project is an end-to-end deep learning system for detecting pneumonia from chest X-ray images. The system uses a MobileNetV2-based convolutional neural network to classify X-ray images into Pneumonia or Normal categories.

The project also includes a Streamlit web application that allows users to upload X-ray images and receive real-time predictions along with confidence scores and downloadable reports.

🚀 Features

Deep learning model using MobileNetV2 architecture

Real-time X-ray image prediction

Confidence score visualization

Robust preprocessing for:

Grayscale images

Inverted X-rays

Different image sizes

Prediction logging using SQLite database

Interactive analytics using Plotly

Downloadable prediction reports

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Pandas

PIL (Python Imaging Library)

Streamlit

SQLite

Plotly

🧠 Model Details

Architecture: MobileNetV2 (Transfer Learning)

Task: Binary Classification (Pneumonia vs Normal)

Validation Accuracy: ~92%

Input Size: 224 × 224

📂 Project Structure
project/
│
├── app.py                  # Streamlit application
├── pneumonia_model.keras   # Trained model
├── database.db             # SQLite database
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── utils/
│   ├── preprocess.py       # Image preprocessing
│   ├── database.py         # Database operations
│
└── reports/
    ├── prediction_reports  # Generated reports

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
streamlit run app.py


The app will open in your browser (usually http://localhost:8501
).

📊 How It Works

User uploads a chest X-ray image

Image is preprocessed (resize, normalize, format handling)

Model predicts pneumonia probability

Result + confidence score displayed

Prediction stored in SQLite database

Report can be downloaded

🗄️ Database Logging

The system stores:

Prediction result

Confidence score

Timestamp

Image reference

📈 Visualization

Plotly is used to generate:

Prediction confidence charts

Historical prediction trends

Performance analytics

🧪 Future Improvements

Multi-class lung disease classification

Model explainability (Grad-CAM visualization)

Cloud deployment

Doctor feedback integration

Batch X-ray processing
