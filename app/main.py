import os
import numpy as np
import tempfile
import librosa
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import keras
from keras.layers import BatchNormalization
import tensorflow as tf
from app.utils import (
    extract_spectrogram,
    load_audio,
    extract_audio_features
)
from app.config import MODEL_PATH, THRESHOLD
from app.app_logger import setup_logger


# Logger
logger = setup_logger()
# Load model once
model = None
# def CustomBN(**kwargs):
#     # 🔥 Remove unsupported args
#     kwargs.pop("renorm", None)
#     kwargs.pop("renorm_clipping", None)
#     kwargs.pop("renorm_momentum", None)
#     return BatchNormalization(**kwargs)
def get_model():
    global model
    if model is None:
        logger.info(f"Loading model from: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ MODEL LOAD FAILED: {str(e)}")
            model=None
            raise e   # 🔥 VERY IMPORTANT
    return model
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting application..."
    )

    get_model()

    logger.info(
        "Deepfake model preloaded successfully"
    )

    yield

    logger.info(
        "Shutting down application..."
    )
app = FastAPI(
    title="Deepfake Audio Detection API",
    description=(
        "API for detecting whether an uploaded audio file "
        "is real or AI-generated/fake using a CNN."
    ),
    version="1.0.0",
    lifespan=lifespan
)
@app.get("/")
def home():
    return {"message": "Deepfake Audio Detection API is running 🚀"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded audio file is REAL or FAKE.

    Supported audio formats:
        WAV
        MP3
        FLAC
        M4A
        OGG
    """
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
        model=get_model()
        if model is None:
            return {"error": "Model not loaded"}
        prediction_output = model.predict(
            spec,
            verbose=0
        )
        prob = float(
            prediction_output[0][0]
        )
        threshold = float(THRESHOLD)
        prediction = 1 if prob > threshold else 0
    
        fake_probability = prob

        real_probability = 1.0 - prob
        if prediction == 1:

            label = "FAKE"

            confidence = fake_probability

        else:

            label = "REAL"

            confidence = real_probability
        # STEP 11: Calculate risk
        if label == "FAKE":

            if confidence >= 0.80:
                risk = "HIGH"

            elif confidence >= 0.60:
                risk = "MEDIUM"

            else:
                risk = "LOW"

        else:

            if confidence >= 0.80:
                risk = "LOW"

            elif confidence >= 0.60:
                risk = "MEDIUM"

            else:
                risk = "HIGH"
        duration = float(
            features["duration"]
        )

        sample_rate = int(
            features["sample_rate"]
        )

        rms = float(
            features["rms"]
        )

        zcr = float(
            features["zcr"]
        )

        logger.info(
            f"File: {file.filename} | "
            f"Prediction: {label} | "
            f"Fake Probability: {fake_probability:.4f} | "
            f"Confidence: {confidence:.4f} | "
            f"Risk: {risk}"
            )

        return {

            "filename": str(
                file.filename
            ),

            "prediction": str(
                label
            ),

            "confidence": float(
                round(confidence, 4)
            ),

            "probability_fake": float(
                round(fake_probability, 4)
            ),

            "probability_real": float(
                round(real_probability, 4)
            ),

            "threshold": float(
                round(threshold, 4)
            ),

            "risk": str(
                risk
            ),

            "features": {

                "duration": float(
                    round(duration, 2)
                ),

                "sample_rate": int(
                    sample_rate
                ),

                "rms": float(
                    round(rms, 4)
                ),

                "zcr": float(
                    round(zcr, 4)
                )
            }
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
    finally:
        # STEP 14: Delete temporary file

        if file_path is not None:

            try:

                if os.path.exists(
                    file_path
                ):

                    os.remove(
                        file_path
                    )

            except Exception as cleanup_error:

                logger.warning(
                    f"Could not delete temporary "
                    f"file: {cleanup_error}"
                )
# uvicorn app.main:app --reload -> backend api run command 