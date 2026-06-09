# 🧠 Deepfake Detection System

<div align="center">


### 🚀 AI-Powered Deepfake Detection using MobileNetV2 & Explainable AI

Detect whether an image or video is **Real** or **AI Generated** with visual explanations powered by **Grad-CAM Heatmaps**.

</div>

---

# 📌 Project Overview

Deepfake media has become increasingly sophisticated due to advances in Artificial Intelligence and Generative Models.

This project introduces a **Deepfake Detection System** that can analyze:

✅ Images

✅ Videos

✅ AI Generated Content

✅ Real Human Media

The system uses **Transfer Learning with MobileNetV2** and **Grad-CAM Explainable AI** to provide transparent predictions.

---

# 🎯 Key Features

### 🔍 Deepfake Detection

- Detects Real vs Fake Images
- Detects Real vs Fake Videos
- Confidence Score Prediction

### 🧠 Explainable AI

- Grad-CAM Heatmaps
- Visual Decision Explanation
- Pixel-Level Attention Maps

### ⚡ Fast Inference

- MobileNetV2 Architecture
- Lightweight Deployment
- Real-Time Predictions

### 🎥 Video Processing

- Frame Extraction
- Multi-frame Analysis
- Voting-Based Final Prediction

### 🎨 Modern Dashboard

- Interactive Streamlit UI
- Dark Theme Interface
- Responsive Design

---

# 🏗️ System Architecture

```text
                +----------------+
                | User Uploads   |
                | Image / Video  |
                +--------+-------+
                         |
                         v
                +----------------+
                | Preprocessing  |
                | Resize/Normalize|
                +--------+-------+
                         |
                         v
                +----------------+
                | MobileNetV2    |
                | Deep Learning  |
                +--------+-------+
                         |
              +----------+----------+
              |                     |
              v                     v
      Prediction            Grad-CAM Heatmap
      (Real/Fake)           Explainability
              |                     |
              +----------+----------+
                         |
                         v
                 Streamlit Dashboard
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Model | MobileNetV2 |
| Explainability | Grad-CAM |
| Frontend | Streamlit |
| Image Processing | PIL |
| Data Loading | TorchVision |

---

# 🧠 Model Details

## Base Architecture

📌 MobileNetV2

Transfer Learning was applied by:

- Loading pretrained ImageNet weights
- Freezing feature extraction layers
- Replacing final classifier
- Fine-tuning on Deepfake Dataset

### Advantages

✅ Fast

✅ Lightweight

✅ High Accuracy

✅ Suitable for Real-Time Detection

---

# 🔥 Explainable AI (Grad-CAM)

Unlike traditional classifiers, this project explains:

👉 Why the prediction was made

👉 Which image regions were important

👉 How the model reached its decision

Heatmaps highlight important areas using:

🔴 Red = Highly Important

🟠 Orange = Important

🔵 Blue = Less Important

---

# 📊 Results

| Input | Prediction |
|---------|------------|
| Real Image | REAL ✅ |
| Fake Image | FAKE ❌ |
| Real Video | REAL ✅ |
| Fake Video | FAKE ❌ |

### Example Confidence Scores

- REAL → 70%+
- FAKE → 80%+

---



# 📂 Project Structure

```text
Deepfake-Detection-System
│
├── app.py
├── train_model.py
├── requirements.txt
├── deepfake_model.pth
│
├── screenshots
│   ├── homepage.png
│   ├── prediction.png
│   ├── heatmap.png
│   └── video-analysis.png
│
└── README.md
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Deepfake-Detection-System.git
```

### Move into Project

```bash
cd Deepfake-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

# 📈 Future Enhancements

- 🎥 Live Webcam Detection
- ☁️ Cloud Deployment
- 🤖 Vision Transformers (ViT)
- 📱 Mobile Application
- 🌐 REST API Support
- 📊 Detection Analytics Dashboard

---

# 🎓 Academic Contribution

This project was developed as a **Final Year Engineering Project** to explore:

- Deep Learning
- Computer Vision
- Explainable AI (XAI)
- Transfer Learning
- AI-Based Media Authentication

---

# 👨‍💻 Author

### Sudarshan Birajdar

🎓 Final Year Computer Engineering Student

💡 Interested in:

- Artificial Intelligence
- Machine Learning
- Data Analytics
- Cloud Computing
- Software Development

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
