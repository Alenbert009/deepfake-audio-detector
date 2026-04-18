import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from app.utils import extract_spectrogram,load_audio,extract_audio_features
from app.config import MODEL_PATH, THRESHOLD
from app.app_logger import setup_logger
# Page config
st.set_page_config(page_title="Deepfake Audio Detector",layout='wide')
# Setup logger
logger=setup_logger()
# Load model
@st.cache_resource
def load_my_model():
    with st.spinner("Loading model... please wait ⏳"):
        
        return load_model(MODEL_PATH)

model = load_my_model()

# Sidebar
st.sidebar.title('⚙️ Settings')
threshold= st.sidebar.slider("Detection Threshold", 0.1, 0.5, THRESHOLD)
st.title("🎧 Deepfake Audio Detection System")
st.markdown("Analyze audio and detect whether it's **Real or Fake**")
uploaded_file=st.file_uploader('Upload WAV file',type=['wav','mp3','ogg'])
if uploaded_file:
    
    st.audio(uploaded_file)
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name
    # Load audio
    
    st.write("Processing audio...")
    audio, sr = load_audio(file_path)
    # Extract features
    features = extract_audio_features(audio, sr)
    # Model prediction
    spec = extract_spectrogram(file_path)
    spec = np.expand_dims(spec, axis=0)
    prob = model.predict(spec)[0][0]
    prediction = 1 if prob > threshold else 0
    logger.info(f"Prediction: {prediction}, Probability: {prob:.4f}")
    # RESULT SECTION
    st.subheader("🔍 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        if prediction==1:
            st.error(f"⚠️ FAKE Audio\nConfidence: {prob:.2f}")
        else:
            st.success(f"✅ REAL Audio\nConfidence: {1 - prob:.2f}")
    with col2:
        st.metric("Fake Probability", f"{prob:.2f}")
    st.progress(float(prob))
    # AUDIO FEATURES
    st.subheader("📊 Audio Features")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{features['duration']} sec")
    c2.metric("Sample Rate", features['sample_rate'])
    c3.metric("RMS Energy", features['rms'])
    st.metric("Zero Crossing Rate", features['zcr'])
     # WAVEFORM
    st.subheader("📈 Waveform")
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)
    # SPECTROGRAM
    st.subheader("🎧 Mel Spectrogram")
    spec_vis = librosa.feature.melspectrogram(y=audio, sr=sr)
    spec_db = librosa.power_to_db(spec_vis, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)
    