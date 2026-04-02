# 🎧 Deepfake Audio Detection System

A Deep Learning-based web application that detects whether an audio clip is **Real or Fake (Deepfake)** using a Convolutional Neural Network (CNN) trained on Mel Spectrogram features.

---

## 🚀 Live Demo

👉 *(Add your Streamlit link here after deployment)*

---

## 📌 Features

* 🎧 Upload audio files (`.wav`, `.mp3`, `.ogg`)
* 🤖 Detect **Real vs Fake** audio using CNN
* 📊 Visualize:

  * Waveform
  * Mel Spectrogram
* 📈 Display prediction confidence
* ⚙️ Adjustable detection threshold
* 📁 Demo audio support for testing

---

## 🧠 Model Architecture (CNN)

The model uses a **Convolutional Neural Network (CNN)** trained on Mel Spectrogram inputs.

### 🔹 Input

* Shape: **(128 × 128 × 1)**

### 🔹 Architecture

* Conv2D (16 filters) + BatchNorm + MaxPooling
* Conv2D (32 filters) + BatchNorm + MaxPooling
* Conv2D (64 filters) + BatchNorm + MaxPooling
* Flatten
* Dense (64) + Dropout
* Output Layer (Sigmoid)

---

## 📊 Model Summary

```
Model: "sequential"

Conv2D (16) → (128,128,16)
BatchNormalization
MaxPooling2D → (64,64,16)

Conv2D (32) → (64,64,32)
BatchNormalization
MaxPooling2D → (32,32,32)

Conv2D (64) → (30,30,64)
BatchNormalization
MaxPooling2D → (15,15,64)

Flatten → 14400
Dense (64) → 921,664 params
Dropout
Dense (1)

Total Parameters: 945,475
```

---

## ⚙️ Tech Stack

* Python 🐍
* Streamlit 🌐
* TensorFlow / Keras 🤖
* Librosa 🎧
* NumPy 📊
* Matplotlib 📈

---

## 📁 Project Structure

```
deepfake-audio-detector/
│
├── app/
│   ├── main.py
│   ├── utils.py
│   ├── config.py
│   ├── app_logger.py
│
├── model/
│   └── deepfake_audio_model.h5
│
├── data/
│   └── demo/
│       ├── real/
│       └── fake/
│
├── logs/
│   └── app.log
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/deepfake-audio-detector.git
cd deepfake-audio-detector
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the app

```
streamlit run app/main.py
```

---

## 🎯 How It Works

1. Upload an audio file
2. Audio is converted to a **Mel Spectrogram**
3. Spectrogram is resized to **128×128**
4. Passed through CNN model
5. Output:

   * **0 → Real**
   * **1 → Fake**

---

## 📊 Model Performance

* Accuracy: ~50% (current version)
* Observations:

  * Model shows bias toward real audio
  * Needs better class balancing and tuning

---

## 🔮 Future Improvements

* Improve accuracy using balanced dataset
* Use advanced architectures (CNN + LSTM)
* Real-time audio detection
* Noise-robust feature extraction
* Support variable-length audio

---

## 📸 Screenshots

*(Add screenshots of your Streamlit UI here)*

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* ASVspoof Dataset
* Librosa Documentation
* TensorFlow/Keras

---

## 👨‍💻 Author

**Sujoy Dass**

---

⭐ If you like this project, consider giving it a star!
