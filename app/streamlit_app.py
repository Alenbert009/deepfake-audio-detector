import os
import sys
import tempfile
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
from report_generator import (
    generate_detection_report,
    generate_batch_detection_report
)
import tempfile
from streamlit_mic_recorder import mic_recorder
import time

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.utils import load_audio
from app.app_logger import setup_logger


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "https://deepfake-audio-detector-6.onrender.com/predict"

REQUEST_TIMEOUT = 120


# =========================================================
# LOGGER
# =========================================================

logger = setup_logger()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VoxAuth – AI-Generated Speech Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ==============================
       MAIN APPLICATION
       ============================== */

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ==============================
       HERO HEADER
       ============================== */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e1b4b 50%,
            #312e81 100%
        );

        padding: 30px 35px;
        border-radius: 20px;
        margin-bottom: 25px;

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 10px 35px rgba(0,0,0,0.35);
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 18px;
    }

    .status-pill {
        display: inline-block;

        padding: 7px 14px;

        border-radius: 999px;

        background: rgba(34,197,94,0.15);

        border: 1px solid rgba(34,197,94,0.35);

        color: #86efac;

        font-size: 14px;

        font-weight: 600;
    }


    /* ==============================
       RESULT CARDS
       ============================== */

    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.15);
        background: rgba(15,23,42,0.75);
        margin-bottom: 1rem;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.20);
    }


    /* ==============================
       SECTION HEADERS
       ============================== */

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }


    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a,
            #0b1120
        );

        border-right: 1px solid rgba(148,163,184,0.12);
    }


    /* ==============================
       FILE UPLOADER
       ============================== */

    [data-testid="stFileUploader"] {
        background: rgba(15,23,42,0.65);

        border-radius: 16px;

        padding: 0.5rem;

        border: 1px dashed rgba(148,163,184,0.30);
    }


    /* ==============================
       METRICS
       ============================== */

    [data-testid="stMetric"] {
        background: rgba(15,23,42,0.65);

        padding: 1rem;

        border-radius: 14px;

        border: 1px solid rgba(148,163,184,0.10);
    }


    /* ==============================
       DOWNLOAD BUTTON
       ============================== */

    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
    }


    /* ==============================
       NORMAL BUTTONS
       ============================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
"""
<div class="hero">
<div class="hero-title">
🎙️ VoxAuth
</div>
<div class="hero-subtitle">
AI-Generated Speech Detection System and authenticity analysis
using CNN-based Mel Spectrogram classification.
</div>
<div class="status-pill">
🟢 Detection Engine Online
</div>
</div>
""",
unsafe_allow_html=True
)
# =========================================================
# SIDEBAR
# =========================================================

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Detection Settings")

    st.caption("SYSTEM CONFIGURATION")

    # Inference Engine
    st.markdown("### 🚀 Inference Engine")
    st.info("FastAPI Server\n\nRemote model inference")

    st.divider()

    # Model
    st.markdown("### 🧠 Detection Model")
    st.info("CNN Audio Classifier\n\nSpectrogram-based deep learning")

    st.divider()

    # Input representation
    st.markdown("### 🎵 Feature Extraction")
    st.info("Mel Spectrogram\n\nCNN input representation")

    st.divider()

    # Supported formats
    st.markdown("### 📁 Supported Formats")

    st.markdown(
        """
        **WAV** • **MP3** • **OGG** • **FLAC** • **M4A**
        """
    )

    st.divider()

    # API status
    st.markdown("### 🌐 API Status")

    # Since you already have the API URL
    st.success("🟢 FastAPI Connected")

    st.divider()

    # Threshold
    st.markdown("### 🎯 Detection Threshold")

    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.20,
        step=0.05,
        help=(
            "Probability above this value is classified as FAKE."
        )
    )
    st.sidebar.markdown("---")

    detection_mode = st.sidebar.radio(
        "Detection Mode",
        ["Single Audio", "Batch Detection","Live Microphone"],
        index=0
    )

    st.caption(
        f"Current threshold: **{threshold:.2f}**"
    )

    st.divider()

    st.caption(
        "VoxAuth – AI-Generated Speech Detection System"
    )




st.sidebar.info(
    "The uploaded audio is sent to the deployed "
    "FastAPI model for prediction."
)


# =========================================================
# HEADER
# =========================================================


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    accept_multiple_files=True,
    help="Supported formats: WAV, MP3, OGG, FLAC and M4A"
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if detection_mode == "Single Audio" and uploaded_files: # as uploaded_files is now a list of files, we need to iterate over it

    # -----------------------------------------------------
    # File information
    # -----------------------------------------------------
    if detection_mode == "Single Audio":  
        uploaded_file = uploaded_files[0]
        file_name = uploaded_file.name

        file_extension = os.path.splitext(
            file_name
        )[1].lower()

        st.success(
            f"File selected: **{file_name}**"
        )

        # -----------------------------------------------------
        # Audio player
        # -----------------------------------------------------

        st.subheader("🔊 Audio Preview")

        # Read the uploaded audio as bytes
        audio_bytes = uploaded_file.getvalue()

        st.audio(
            audio_bytes,
            format=uploaded_file.type
        )

        # -----------------------------------------------------
        # Save temporary file
        # -----------------------------------------------------

        file_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                file_path = tmp.name

            # -------------------------------------------------
            # API REQUEST
            # -------------------------------------------------

            st.subheader("🤖 AI Analysis")
            api_start_time = time.time()
            with st.spinner(
                "Analyzing audio with the deepfake detection model..."
            ):

                try:

                    with open(
                        file_path,
                        "rb"
                    ) as audio_file:

                        response = requests.post(
                            API_URL,
                            files={
                                "file": (
                                    file_name,
                                    audio_file,
                                    uploaded_file.type
                                )
                            },
                            timeout=REQUEST_TIMEOUT
                        )
                    api_end_time = time.time()
                    api_time = api_end_time - api_start_time
                    st.info(
                        f"⚡ API response time: "
                        f"{api_time:.2f} seconds"
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ The prediction server took too long "
                        "to respond. The Render server may be waking up."
                    )

                    st.stop()

                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Could not connect to the prediction API."
                    )

                    st.stop()

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"❌ API request failed: {str(e)}"
                    )

                    st.stop()

            # -------------------------------------------------
            # HTTP ERROR
            # -------------------------------------------------

            if response.status_code != 200:

                st.error(
                    f"❌ API returned HTTP "
                    f"{response.status_code}"
                )

                try:
                    st.json(
                        response.json()
                    )
                except Exception:
                    st.code(
                        response.text
                    )

                st.stop()

            # -------------------------------------------------
            # JSON RESPONSE
            # -------------------------------------------------

            try:

                result = response.json()

            except ValueError:

                st.error(
                    "❌ The API returned an invalid response."
                )

                st.stop()

            # -------------------------------------------------
            # API ERROR
            # -------------------------------------------------

            if "error" in result:

                st.error(
                    f"❌ {result['error']}"
                )

                st.stop()

            # =================================================
            # EXTRACT API RESULTS
            # =================================================

            prediction = result.get(
                "prediction",
                "UNKNOWN"
            )

            confidence = float(
                result.get(
                    "confidence",
                    0
                )
            )

            fake_probability = float(
                result.get(
                    "probability_fake",
                    0
                )
            )

            real_probability = float(
                result.get(
                    "probability_real",
                    1 - fake_probability
                )
            )

            risk = result.get(
                "risk",
                "UNKNOWN"
            )

            api_threshold = float(
                result.get(
                    "threshold",
                    0.2
                )
            )

            features = result.get(
                "features",
                {}
            )

            # =================================================
            # LOG
            # =================================================

            logger.info(
                f"Frontend result | "
                f"File: {uploaded_file} | "
                f"Prediction: {prediction} | "
                f"Confidence: {confidence:.4f}"
            )

            # =================================================
            # RESULT SECTION
            # =================================================

            st.divider()

            st.subheader(
                "🔍 Detection Result"
            )

            result_col1, result_col2, result_col3 = st.columns(
                3
            )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            with result_col1:

                if prediction == "FAKE":

                    st.error(
                        "⚠️ FAKE AUDIO"
                    )

                elif prediction == "REAL":

                    st.success(
                        "✅ REAL AUDIO"
                    )

                else:

                    st.warning(
                        "⚠️ UNKNOWN"
                    )

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            with result_col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # -------------------------------------------------
            # Risk
            # -------------------------------------------------

            with result_col3:

                if risk == "HIGH":

                    st.error(
                        f"Risk: {risk}"
                    )

                elif risk == "MEDIUM":

                    st.warning(
                        f"Risk: {risk}"
                    )

                else:

                    st.success(
                        f"Risk: {risk}"
                    )

            # =================================================
            # PROBABILITY SECTION
            # =================================================

            st.subheader(
                "📊 Model Probabilities"
            )

            prob_col1, prob_col2 = st.columns(
                2
            )

            with prob_col1:

                st.metric(
                    "Fake Probability",
                    f"{fake_probability * 100:.2f}%"
                )

                st.progress(
                    fake_probability
                )

            with prob_col2:

                st.metric(
                    "Real Probability",
                    f"{real_probability * 100:.2f}%"
                )

                st.progress(
                    real_probability
                )

            st.caption(
                f"Model decision threshold: "
                f"{api_threshold:.2f}"
            )

            # =================================================
            # AUDIO FEATURES
            # =================================================

            st.divider()

            st.subheader(
                "🎵 Audio Signal Analysis"
            )

            feature_col1, feature_col2, feature_col3, feature_col4 = (
                st.columns(4)
            )

            with feature_col1:

                st.metric(
                    "Duration",
                    f"{features.get('duration', 0):.2f} sec"
                )

            with feature_col2:

                st.metric(
                    "Sample Rate",
                    f"{features.get('sample_rate', 0):,} Hz"
                )

            with feature_col3:

                st.metric(
                    "RMS Energy",
                    f"{features.get('rms', 0):.4f}"
                )

            with feature_col4:

                st.metric(
                    "Zero Crossing Rate",
                    f"{features.get('zcr', 0):.4f}"
                )

            # =================================================
            # LOAD AUDIO FOR VISUALIZATION
            # =================================================
            visualization_start_time = time.time()
            with st.spinner(
                "Generating audio visualizations..."
            ):

                audio, sr = load_audio(
                    file_path
                )

            # =================================================
            # WAVEFORM
            # =================================================

            st.divider()

            st.subheader(
                "📈 Waveform Analysis"
            )

            fig_wave, ax_wave = plt.subplots(
                figsize=(12, 4)
            )
            display_audio=audio[::10]
            display_sr=sr/10

            librosa.display.waveshow(
                display_audio,
                sr=display_sr,
                ax=ax_wave,
                color="#6366f1"
            )

            ax_wave.set_title(
                "Audio Waveform"
            )

            ax_wave.set_xlabel(
                "Time (seconds)"
            )

            ax_wave.set_ylabel(
                "Amplitude"
            )

            fig_wave.tight_layout()

        
            # SAVE WAVEFORM FOR PDF REPORT
            waveform_path = os.path.join(
                tempfile.gettempdir(),
                "deepfake_waveform.png"
            )
            fig_wave.savefig(
                waveform_path,
                dpi=150,
                bbox_inches="tight"
            )
            st.pyplot(
                fig_wave,
                clear_figure=True
            )

            plt.close(
                fig_wave
            )

            # =================================================
            # MEL SPECTROGRAM
            # =================================================

            st.subheader(
                "🎧 Mel Spectrogram"
            )

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=64,
                n_fft=1024,
                hop_length=512

            )

            mel_db = librosa.power_to_db(
                mel_spec,
                ref=np.max
            )

            fig_spec, ax_spec = plt.subplots(
                figsize=(12, 5)
            )

            img = librosa.display.specshow(
                mel_db,
                sr=sr,
                x_axis="time",
                y_axis="mel",
                ax=ax_spec,
                cmap="magma"
            )

            fig_spec.colorbar(
                img,
                ax=ax_spec,
                format="%+2.0f dB"
            )

            ax_spec.set_title(
                "Mel Spectrogram"
            )

            ax_spec.set_xlabel(
                "Time (seconds)"
            )

            ax_spec.set_ylabel(
                "Mel Frequency"
            )

            fig_spec.tight_layout()
            #saving the mel spectrogram in a temporary directory so that we can use it for pdf report generation
            spectrogram_path = os.path.join(
                tempfile.gettempdir(),
                "deepfake_mel_spectrogram.png"
            )
            fig_spec.savefig(
                spectrogram_path,
                dpi=150,
                bbox_inches="tight"
            )
            

            st.pyplot(
                fig_spec,
                clear_figure=True
            )

            plt.close(fig_spec)
            visualization_end_time = time.time()

            visualization_time = (
                visualization_end_time -
                visualization_start_time
            )

            st.info(
                f"📊 Visualization time: "
                f"{visualization_time:.2f} seconds"
)
            # =================================================
            # INTERPRETATION
            # =================================================

            st.divider()

            st.subheader(
                "🧠 Analysis Summary"
            )

            if prediction == "FAKE":

                st.warning(
                    f"""
                    The model classified this audio as **FAKE**
                    with **{confidence * 100:.2f}% confidence**.

                    The estimated probability of synthetic/manipulated
                    audio is **{fake_probability * 100:.2f}%**.

                    Risk level: **{risk}**.
                    """
                )

            elif prediction == "REAL":

                st.success(
                    f"""
                    The model classified this audio as **REAL**
                    with **{confidence * 100:.2f}% confidence**.

                    The estimated probability of fake/manipulated
                    audio is **{fake_probability * 100:.2f}%**.

                    Risk level: **{risk}**.
                    """
                )

            st.caption(
                "Note: Model predictions are probabilistic and should "
                "not be treated as definitive forensic evidence."
            )
        # PDF DETECTION REPORT
            st.divider()
            st.subheader(
                "📄Detection Report"
            )
            if st.button(
                "Generate PDF Report"
            ):
                with st.spinner(
                    "Generating PDF report..."
                ):
                    base_filename = os.path.splitext(
                        uploaded_file.name
                    )[0]
                    pdf_file = generate_detection_report(
                        filename=file_name,
                        prediction=prediction,
                        confidence=confidence,
                        probability_fake=fake_probability,
                        probability_real=real_probability,
                        threshold=threshold,
                        risk=risk,
                        features=features,
                        waveform_path=waveform_path,
                        spectrogram_path=spectrogram_path,
                    )
                    st.download_button(
                        label="📥 Download Detection Report",
                        data=pdf_file,
                        file_name=(
                            f"deepfake_detection_"
                            f"{base_filename}.pdf"
                        ),
                        mime="application/pdf",
                    )
        finally:

            # -------------------------------------------------
            # Clean temporary file
            # -------------------------------------------------

            if file_path and os.path.exists(
                file_path
            ):

                try:

                    os.remove(
                        file_path
                    )

                except Exception as cleanup_error:

                    logger.warning(
                        f"Could not delete temporary file: "
                        f"{cleanup_error}"
                    )
elif detection_mode == "Batch Detection":
     st.title("📦 Batch Audio Detection")
     if not uploaded_files:

        st.info(
            "📁 Please upload audio files to start batch detection."
        )
     else:

        st.markdown(
            """
            Upload multiple audio files and analyze them
            using the CNN-based deepfake audio detection model.
            """
        )

        st.info(
            f"📁 {len(uploaded_files)} files selected"
        )

        # -------------------------------------------------
        # Batch results will be stored here
        # -------------------------------------------------

        batch_results = []

        # -------------------------------------------------
        # Process each uploaded file
        # -------------------------------------------------

        for uploaded_file in uploaded_files:

            file_name = uploaded_file.name

            status_placeholder = st.empty()

            status_placeholder.info(
                f"🔍 Analyzing **{file_name}**..."
            )
            file_extension = os.path.splitext(
                file_name
            )[1].lower()

            file_path = None

            try:

                # Read uploaded file once
                file_bytes = uploaded_file.getvalue()

                if not file_bytes:

                    st.error(
                        f"❌ {file_name}: Empty audio file."
                    )

                    continue

                # Save temporary copy
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_extension
                ) as tmp:

                    tmp.write(file_bytes)

                    file_path = tmp.name
                with st.spinner(
                    f"Analyzing {file_name}..."
                ):

                    with open(
                        file_path,
                        "rb"
                    ) as audio_file:

                        response = requests.post(
                            API_URL,
                            files={
                                "file": (
                                    file_name,
                                    audio_file,
                                    uploaded_file.type
                                )
                            },
                            timeout=REQUEST_TIMEOUT
                        )
                 # HTTP ERROR
                if response.status_code != 200:

                    st.error(
                        f"❌ {file_name}: "
                        f"API returned HTTP "
                        f"{response.status_code}"
                    )

                    continue

                # -----------------------------------------
                # JSON RESPONSE
                # -----------------------------------------

                try:

                    result = response.json()

                except ValueError:

                    st.error(
                        f"❌ {file_name}: "
                        "Invalid API response."
                    )

                    continue

                # -----------------------------------------
                # API ERROR
                # -----------------------------------------

                if "error" in result:

                    st.error(
                        f"❌ {file_name}: "
                        f"{result['error']}"
                    )

                    continue
                # Extract results
                # -----------------------------------------

                prediction = result.get(
                    "prediction",
                    "UNKNOWN"
                )

                confidence = float(
                    result.get(
                        "confidence",
                        0
                    )
                )

                fake_probability = float(
                    result.get(
                        "probability_fake",
                        0
                    )
                )

                real_probability = float(
                    result.get(
                        "probability_real",
                        1 - fake_probability
                    )
                )

                risk = result.get(
                    "risk",
                    "UNKNOWN"
                )

                # -----------------------------------------
                # Store result
                # -----------------------------------------

                batch_results.append(
                    {
                        "filename": file_name,
                        "prediction": prediction,
                        "confidence": confidence,
                        "fake_probability": fake_probability,
                        "real_probability": real_probability,
                        "risk": risk
                    }
                )

                logger.info(
                    f"Batch result | "
                    f"File: {file_name} | "
                    f"Prediction: {prediction} | "
                    f"Confidence: {confidence:.4f}"
                )
#removing tthe temporary file after processing each file in the batch
            finally:
                # Cleanup
                if file_path and os.path.exists(
                    file_path
                ):

                    try:

                        os.remove(
                            file_path
                        )

                    except Exception as cleanup_error:

                        logger.warning(
                            f"Could not delete temporary "
                            f"file {file_name}: "
                            f"{cleanup_error}"
                        )
                status_placeholder.empty()
        if batch_results:

                    st.divider()

                    st.subheader("📊 Batch Detection Results")

                    # Create table data
                    table_data = []

                    for result in batch_results:

                        table_data.append({
                            "File": result["filename"],
                            "Prediction": result["prediction"],
                            "Confidence": f"{result['confidence'] * 100:.2f}%",
                            "Fake Probability": f"{result['fake_probability'] * 100:.2f}%",
                            "Risk": result["risk"]
                        })

                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=True
                    )
                        # =====================================================
                    # BATCH PDF REPORT
                    # =====================================================

                    st.divider()

                    st.subheader(
                        "📄 Batch Detection Report"
                    )

                    batch_pdf = generate_batch_detection_report(
                        results=batch_results,
                        threshold=threshold
                    )
                    st.download_button(
                        label="📥 Download Batch Detection Report",
                        data=batch_pdf,
                        file_name="deepfake_batch_detection_report.pdf",
                        mime="application/pdf"
                    )
elif detection_mode == "Live Microphone":

        st.title("🎙️ Live Microphone Detection")

        st.markdown(
            """
            Record a short audio sample using your microphone
            and analyze it using the CNN-based deepfake audio
            detection model.
            """
        )

        st.info(
            "🎙️ Click **Start Recording**, speak into your microphone, "
            "then click **Stop Recording**."
        )

        # =========================================================
        # MICROPHONE RECORDER
        # =========================================================

        audio_recording = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            format="wav",
            key="live_microphone"
        )

        # CHECK WHETHER RECORDING EXISTS

        if audio_recording:

            st.success(
                "✅ Recording captured successfully!"
            )
            recording_bytes = audio_recording["bytes"]

            st.write(
                "Recording size:",
                len(recording_bytes),
                "bytes"
            )

            st.audio(
                recording_bytes,
                format="audio/wav"
            )

            # -----------------------------------------------------
            # AUDIO PREVIEW
            # -----------------------------------------------------

            st.subheader(
                "🔊 Recorded Audio"
            )

            st.audio(
                audio_recording["bytes"],
                format="audio/wav"
            )

            # -----------------------------------------------------
            # SAVE RECORDING
            # -----------------------------------------------------

            file_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as tmp:

                    tmp.write(
                        audio_recording["bytes"]
                    )

                    file_path = tmp.name

                # =================================================
                # SEND TO FASTAPI
                # =================================================

                st.subheader(
                    "🤖 AI Analysis"
                )

                with st.spinner(
                    "Analyzing recorded audio with the CNN model..."
                ):

                    try:

                        with open(
                            file_path,
                            "rb"
                        ) as audio_file:

                            response = requests.post(
                                API_URL,
                                files={
                                    "file": (
                                        "microphone_recording.wav",
                                        audio_file,
                                        "audio/wav"
                                    )
                                },
                                timeout=(30,180)
                            )


                    except requests.exceptions.Timeout:

                        st.error(
                            "⏱️ The prediction server took too long "
                            "to respond."
                        )

                        st.stop()

                    except requests.exceptions.ConnectionError:

                        st.error(
                            "❌ Could not connect to the prediction API."
                        )

                        st.stop()

                    except requests.exceptions.RequestException as e:

                        st.error(
                            f"❌ API request failed: {str(e)}"
                        )

                        st.stop()

                # =================================================
                # HTTP ERROR
                # =================================================

                if response.status_code != 200:

                    st.error(
                        f"❌ API returned HTTP "
                        f"{response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except Exception:

                        st.code(
                            response.text
                        )

                    st.stop()

                # =================================================
                # PARSE JSON
                # =================================================

                try:

                    result = response.json()

                except ValueError:

                    st.error(
                        "❌ The API returned an invalid response."
                    )

                    st.stop()

                # =================================================
                # API ERROR
                # =================================================

                if "error" in result:

                    st.error(
                        f"❌ {result['error']}"
                    )

                    st.stop()

                # =================================================
                # EXTRACT RESULTS
                # =================================================

                prediction = result.get(
                    "prediction",
                    "UNKNOWN"
                )

                confidence = float(
                    result.get(
                        "confidence",
                        0
                    )
                )

                fake_probability = float(
                    result.get(
                        "probability_fake",
                        0
                    )
                )

                real_probability = float(
                    result.get(
                        "probability_real",
                        1 - fake_probability
                    )
                )

                risk = result.get(
                    "risk",
                    "UNKNOWN"
                )

                api_threshold = float(
                    result.get(
                        "threshold",
                        0.2
                    )
                )

                features = result.get(
                    "features",
                    {}
                )

                # =================================================
                # LOG
                # =================================================

                logger.info(
                    f"Microphone result | "
                    f"Prediction: {prediction} | "
                    f"Confidence: {confidence:.4f}"
                )

                # =================================================
                # DETECTION RESULT
                # =================================================

                st.divider()

                st.subheader(
                    "🔍 Live Detection Result"
                )

                result_col1, result_col2, result_col3 = st.columns(
                    3
                )

                # -------------------------------------------------
                # Prediction
                # -------------------------------------------------

                with result_col1:

                    if prediction == "FAKE":

                        st.error(
                            "⚠️ FAKE AUDIO"
                        )

                    elif prediction == "REAL":

                        st.success(
                            "✅ REAL AUDIO"
                        )

                    else:

                        st.warning(
                            "⚠️ UNKNOWN"
                        )

                # -------------------------------------------------
                # Confidence
                # -------------------------------------------------

                with result_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                # -------------------------------------------------
                # Risk
                # -------------------------------------------------

                with result_col3:

                    if risk == "HIGH":

                        st.error(
                            f"Risk: {risk}"
                        )

                    elif risk == "MEDIUM":

                        st.warning(
                            f"Risk: {risk}"
                        )

                    else:

                        st.success(
                            f"Risk: {risk}"
                        )

                # =================================================
                # PROBABILITIES
                # =================================================

                st.subheader(
                    "📊 Model Probabilities"
                )

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:

                    st.metric(
                        "Fake Probability",
                        f"{fake_probability * 100:.2f}%"
                    )

                    st.progress(
                        fake_probability
                    )

                with prob_col2:

                    st.metric(
                        "Real Probability",
                        f"{real_probability * 100:.2f}%"
                    )

                    st.progress(
                        real_probability
                    )

                st.caption(
                    f"Model decision threshold: "
                    f"{api_threshold:.2f}"
                )

                # =================================================
                # AUDIO FEATURES
                # =================================================

                st.divider()

                st.subheader(
                    "🎵 Audio Signal Analysis"
                )

                feature_col1, feature_col2, feature_col3, feature_col4 = (
                    st.columns(4)
                )

                with feature_col1:

                    st.metric(
                        "Duration",
                        f"{features.get('duration', 0):.2f} sec"
                    )

                with feature_col2:

                    st.metric(
                        "Sample Rate",
                        f"{features.get('sample_rate', 0):,} Hz"
                    )

                with feature_col3:

                    st.metric(
                        "RMS Energy",
                        f"{features.get('rms', 0):.4f}"
                    )

                with feature_col4:

                    st.metric(
                        "Zero Crossing Rate",
                        f"{features.get('zcr', 0):.4f}"
                    )

                # =================================================
                # LOAD AUDIO
                # =================================================

                with st.spinner(
                    "Generating audio visualizations..."
                ):

                    audio, sr = load_audio(
                        file_path
                    )

                # =================================================
                # WAVEFORM
                # =================================================

                st.divider()

                st.subheader(
                    "📈 Waveform Analysis"
                )

                fig_wave, ax_wave = plt.subplots(
                    figsize=(12, 4)
                )

                librosa.display.waveshow(
                    audio,
                    sr=sr,
                    ax=ax_wave,
                    color="#6366f1"
                )

                ax_wave.set_title(
                    "Recorded Audio Waveform"
                )

                ax_wave.set_xlabel(
                    "Time (seconds)"
                )

                ax_wave.set_ylabel(
                    "Amplitude"
                )
                fig_wave.tight_layout()
                waveform_path = os.path.join(
                    tempfile.gettempdir(),
                    "microphone_waveform.png"
                )

                fig_wave.savefig(
                    waveform_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close(fig_wave)

                # =================================================
                # MEL SPECTROGRAM
                # =================================================

                st.subheader(
                    "🎧 Mel Spectrogram"
                )

                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=128
                )

                mel_db = librosa.power_to_db(
                    mel_spec,
                    ref=np.max
                )

                fig_spec, ax_spec = plt.subplots(
                    figsize=(12, 5)
                )

                img = librosa.display.specshow(
                    mel_db,
                    sr=sr,
                    x_axis="time",
                    y_axis="mel",
                    ax=ax_spec,
                    cmap="magma"
                )

                fig_spec.colorbar(
                    img,
                    ax=ax_spec,
                    format="%+2.0f dB"
                )

                ax_spec.set_title(
                    "Recorded Audio Mel Spectrogram"
                )

                ax_spec.set_xlabel(
                    "Time (seconds)"
                )

                ax_spec.set_ylabel(
                    "Mel Frequency"
                )

                fig_spec.tight_layout()

                # SAVE MICROPHONE SPECTROGRAM FOR PDF
                spectrogram_path = os.path.join(
                    tempfile.gettempdir(),
                    "microphone_mel_spectrogram.png"
                )

                fig_spec.savefig(
                    spectrogram_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                st.pyplot(
                    fig_spec,
                    clear_figure=True
                )

                plt.close(
                    fig_spec
                )

                # =================================================
                # ANALYSIS SUMMARY
                # =================================================

                st.divider()

                st.subheader(
                    "🧠 Analysis Summary"
                )

                if prediction == "FAKE":

                    st.warning(
                        f"""
                        The CNN model classified the recorded audio
                        as **FAKE** with
                        **{confidence * 100:.2f}% confidence**.

                        The estimated probability of synthetic or
                        manipulated audio is
                        **{fake_probability * 100:.2f}%**.

                        Risk level: **{risk}**.
                        """
                    )

                elif prediction == "REAL":

                    st.success(
                        f"""
                        The CNN model classified the recorded audio
                        as **REAL** with
                        **{confidence * 100:.2f}% confidence**.

                        The estimated probability of fake or
                        manipulated audio is
                        **{fake_probability * 100:.2f}%**.

                        Risk level: **{risk}**.
                        """
                    )

                st.caption(
                    "Note: Model predictions are probabilistic and "
                    "should not be treated as definitive forensic evidence."
                )

                # =================================================
                # MICROPHONE PDF REPORT
                # =================================================

                st.divider()

                st.subheader(
                    "📄 Microphone Detection Report"
                )

                microphone_pdf = generate_detection_report(
                    filename="microphone_recording.wav",
                    prediction=prediction,
                    confidence=confidence,
                    probability_fake=fake_probability,
                    probability_real=real_probability,
                    threshold=api_threshold,
                    risk=risk,
                    features=features,
                    waveform_path=waveform_path,
                    spectrogram_path=spectrogram_path
                )

                st.download_button(
                    label="📥 Download Microphone Detection Report",
                    data=microphone_pdf.getvalue(),
                    file_name="deepfake_microphone_detection_report.pdf",
                    mime="application/pdf",
                    key="download_microphone_report"
                )

            finally:

                # =================================================
                # CLEAN TEMPORARY FILE
                # =================================================

                if file_path and os.path.exists(
                    file_path
                ):

                    try:

                        os.remove(
                            file_path
                        )

                    except Exception as cleanup_error:

                        logger.warning(
                            f"Could not delete microphone "
                            f"temporary file: {cleanup_error}"
                        )
# streamlit run app/streamlit_app.py -> run command to start the streamlit app
