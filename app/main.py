import os
import numpy as np
import tempfile
import librosa
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model

from app.utils import extract_spectrogram, load_audio, extract_audio_features
from app.config import MODEL_PATH, THRESHOLD
from app.app_logger import setup_logger
app = FastAPI()
# Logger
logger = setup_logger()
# Load model once
model = load_model(MODEL_PATH)
@app.get("/")
def home():
    return {"message": "Deepfake Audio Detection API is running 🚀"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save temp file-Step 1: Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            file_path = tmp.name
            # Load audio
        audio, sr = load_audio(file_path)

        # Extract features (optional but useful)
        features = extract_audio_features(audio, sr)

        # Spectrogram
        spec = extract_spectrogram(file_path)
        spec = np.expand_dims(spec, axis=0)

        # Prediction
        prob = model.predict(spec)[0][0]
        prediction = 1 if prob > THRESHOLD else 0

        logger.info(f"Prediction: {prediction}, Probability: {prob:.4f}")

        return {
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "confidence": float(prob if prediction == 1 else 1 - prob),
            "probability_fake": float(prob),
            "features": {
                "duration": features["duration"],
                "sample_rate": features["sample_rate"],
                "rms": features["rms"],
                "zcr": features["zcr"]
            }
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}