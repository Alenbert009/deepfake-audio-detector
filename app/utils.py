import librosa
import numpy as np
from app.config import SAMPLE_RATE,MAX_PAD_LEN
def load_audio(file_path):
    audio,sr=librosa.load(file_path,sr=SAMPLE_RATE)
    return audio,sr
def extract_spectrogram(file_path):
    audio,sr=load_audio(file_path)
    spec=librosa.feature.melspectrogram(y=audio,sr=sr)
    spec_db=librosa.power_to_db(spec,ref=np.max)
    if spec_db.shape[1]<MAX_PAD_LEN:
        pad_width=MAX_PAD_LEN-spec_db.shape[1]
        spec_db=np.pad(spec_db,((0,0),(0,pad_width)),mode='constant')
    else:
        spec_db = spec_db[:128, :128]
    spec_db = spec_db / (np.max(np.abs(spec_db)) + 1e-6)
    return spec_db[...,np.newaxis]
def extract_audio_features(audio,sr):
    #duration-calculate the total length of an audio signal in seconds
    duration=librosa.get_duration(y=audio,sr=sr)
    #RMS (Root Mean Square)- RMS represents the average power or loudness of an audio signal over time
    rms=float(np.mean(librosa.feature.rms(y=audio)))
    # ZCR (Zero Crossing Rate)- ZCR is the rate at which an audio signal crosses the zero axis (changes sign from positive to negative or vice versa)
    zcr=float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    return{
        "duration":round(duration,2),
        "sample_rate":sr,
        "rms":round(rms,4),
        "zcr":round(zcr,4)
    }