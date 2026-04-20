import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH=os.path.join(BASE_DIR,"model","deepfake_audio_model.keras")
LOG_PATH=os.path.join(BASE_DIR,'logs','app.log')
THRESHOLD=0.2
SAMPLE_RATE=22050
MAX_PAD_LEN=120
