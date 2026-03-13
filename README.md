<div align="center">

# 🤖 AI Internship Tasks — Kodbud
### *4 Hands-on Artificial Intelligence Projects built with Python*

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.x-4A9F6F?style=for-the-badge&logo=python&logoColor=white)

<br/>

> **Internship Provider:** [Kodbud](https://kodbud.com) &nbsp;|&nbsp; **Tasks Completed:** 4 / 8 &nbsp;|&nbsp; **Status:** ✅ Completed

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tasks Completed](#-tasks-completed)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Task 1 — Spam Email Classifier](#-task-1--spam-email-classifier)
- [Task 2 — Face Detection App](#-task-2--face-detection-app)
- [Task 3 — Sentiment Analysis](#-task-3--sentiment-analysis)
- [Task 4 — Handwritten Digit Recognizer](#-task-4--handwritten-digit-recognizer)
- [Results Summary](#-results-summary)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [LinkedIn Submission](#-linkedin-submission)

---

## 🧠 Overview

This repository contains **4 AI/ML projects** completed as part of the **Kodbud Artificial Intelligence Internship**.

Each project demonstrates a core area of Artificial Intelligence:

| Domain | Task |
|---|---|
| 🔤 NLP + Machine Learning | Spam Email Classifier |
| 👁️ Computer Vision | Face Detection App |
| 💬 Natural Language Processing | Sentiment Analysis |
| 🧠 Deep Learning | Handwritten Digit Recognizer |

---

## ✅ Tasks Completed

```
[✅] Task 1 — Spam Email Classifier        (NLP + Naive Bayes)
[✅] Task 2 — Face Detection App           (OpenCV + Haar Cascade)
[✅] Task 3 — Sentiment Analysis           (NLTK + Logistic Regression)
[✅] Task 4 — Handwritten Digit Recognizer (TensorFlow + Keras + MNIST)
[ ] Task 5 — Chatbot (Rule-Based)
[ ] Task 6 — Rock Paper Scissors AI
[ ] Task 7 — Stock Price Prediction
[ ] Task 8 — AI Virtual Assistant
```

---

## 📁 Project Structure

```
📦 ai-internship-kodbud/
│
├── 📄 task1_spam_classifier.py       # Spam Email Classifier
├── 📄 task2_face_detection.py        # Face Detection App
├── 📄 task3_sentiment_analysis.py    # Sentiment Analysis
├── 📄 task4_digit_recognizer.py      # Handwritten Digit Recognizer
│
├── 📊 outputs/
│   ├── task4_training_history.png    # Training accuracy/loss graph
│   ├── task4_predictions.png         # 5x5 digit prediction grid
│   ├── face_detected_output.jpg      # Face detection sample output
│   └── mnist_digit_model.h5          # Saved trained model
│
└── 📄 README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install All Dependencies at Once

```bash
pip install numpy pandas scikit-learn nltk opencv-python tensorflow matplotlib
```

### Or Install Per Task

```bash
# Task 1 & 3
pip install scikit-learn pandas nltk

# Task 2
pip install opencv-python

# Task 4
pip install tensorflow matplotlib
```

---

## 📧 Task 1 — Spam Email Classifier

### 📖 What it Does
Classifies SMS/Email messages as **SPAM** or **HAM (Not Spam)** using Natural Language Processing and Machine Learning.

### 🔧 Approach
```
Raw Text  →  TF-IDF Vectorizer  →  Multinomial Naive Bayes  →  Spam / Ham
```

### 🛠️ Tools Used
- **Dataset:** SMS Spam Collection (UCI Repository) — 5,572 messages
- **Vectorizer:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Model:** Multinomial Naive Bayes
- **Library:** scikit-learn

### ▶️ Run

```bash
python task1_spam_classifier.py
```

### 📊 Sample Output

```
🚨 SPAM  (99.1%) | Congratulations! You've won a FREE iPhone! Click now...
✅ HAM   (98.4%) | Hey, can we reschedule our meeting to 3pm?
🚨 SPAM  (97.2%) | URGENT: Your account has been compromised. Call us NOW!
✅ HAM   (99.0%) | Don't forget mom's birthday is this Sunday!
```

### 🎯 Accuracy: **97.58%**

### 📸 Output Screenshot
![Task 1] <img width="1829" height="1010" alt="Image" src="https://github.com/user-attachments/assets/93834665-cb70-4bb1-b89c-d4528c10c099" />[](url)

---

## 👁️ Task 2 — Face Detection App

### 📖 What it Does
Detects human faces in **real-time using webcam** or in **static image files** using Computer Vision.

### 🔧 Approach
```
Webcam / Image  →  Grayscale Conversion  →  Haar Cascade  →  Draw Rectangles
```

### 🛠️ Tools Used
- **Algorithm:** Haar Cascade Classifier (frontal face + eyes)
- **Library:** OpenCV (`cv2`)
- **Input:** Live webcam OR image file

### ▶️ Run

```bash
python task2_face_detection.py
```

```
Choose mode:
  1 - Webcam (live face detection)
  2 - Image file (detect faces in a photo)
```

### 🎮 Controls (Webcam Mode)
| Key | Action |
|-----|--------|
| `Q` | Quit webcam |
| `S` | Save screenshot |

### ✨ Features
- Detects multiple faces simultaneously
- Draws blue rectangles around faces
- Detects eyes inside each face region
- Shows face count + frame number on screen
- Saves output image automatically (image mode)

### 📸 Output Screenshot
![Task 2] <img width="1920" height="1017" alt="Image" src="https://github.com/user-attachments/assets/fe713545-8c27-4428-85ad-fc3a228c1438" />[](url)

---

## 💬 Task 3 — Sentiment Analysis

### 📖 What it Does
Classifies product reviews and tweets as **POSITIVE 😊** or **NEGATIVE 😠** using NLP techniques.

### 🔧 Approach
```
Raw Text  →  Clean (remove URLs, mentions, stopwords)  →  Stemming
         →  TF-IDF (bigrams)  →  Logistic Regression  →  Positive / Negative
```

### 🛠️ Tools Used
- **Dataset:** Twitter US Airline Sentiment (~2,000 tweets)
- **Text Cleaning:** Regex, NLTK Stopwords, Porter Stemmer
- **Vectorizer:** TF-IDF with bigrams (`ngram_range=(1,2)`)
- **Model:** Logistic Regression
- **Libraries:** NLTK, scikit-learn

### ▶️ Run

```bash
python task3_sentiment_analysis.py
```

### 📊 Sample Output

```
😊 POSITIVE (94.1%) | "This product is absolutely fantastic! Love it..."
😠 NEGATIVE (97.3%) | "Worst experience ever. Completely broken on arrival."
😊 POSITIVE (88.5%) | "I am so happy with this purchase, exceeded expectations!"
😠 NEGATIVE (95.0%) | "Terrible customer service, waited weeks and got nothing."
```

### 🎯 Accuracy: **~83%**

### 📸 Output Screenshot
![Task 3]<img width="1920" height="1015" alt="Image" src="https://github.com/user-attachments/assets/6de612cc-c9d2-4f41-867d-da9a67b1bdd9" />[](url)

---

## 🔢 Task 4 — Handwritten Digit Recognizer

### 📖 What it Does
Trains a **Deep Neural Network** to recognize handwritten digits (0–9) from the famous **MNIST dataset** with ~98% accuracy.

### 🔧 Neural Network Architecture
```
Input (784)  →  Dense(256) + BatchNorm + Dropout(0.3)
             →  Dense(128) + BatchNorm + Dropout(0.2)
             →  Dense(64)
             →  Output(10) Softmax  →  Digit 0–9
```

### 🛠️ Tools Used
- **Dataset:** MNIST — 70,000 handwritten digit images (28×28 px)
- **Framework:** TensorFlow 2.x / Keras
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

### ▶️ Run

```bash
python task4_digit_recognizer.py
```

> ⚠️ First run downloads MNIST (~11MB). Training takes ~2–5 minutes.

### 📊 Training Progress

```
Epoch  1/20 — loss: 0.2341 — accuracy: 0.9301 — val_accuracy: 0.9612
Epoch  5/20 — loss: 0.0821 — accuracy: 0.9756 — val_accuracy: 0.9798
Epoch 10/20 — loss: 0.0512 — accuracy: 0.9845 — val_accuracy: 0.9861
EarlyStopping triggered — restoring best weights...

🎯 Final Test Accuracy: 98.62%
```

### 📁 Generated Output Files
| File | Description |
|------|-------------|
| `task4_training_history.png` | Accuracy & loss curves over epochs |
| `task4_predictions.png` | 5×5 grid of predicted digits |
| `mnist_digit_model.h5` | Saved trained model (reusable) |

### 🎯 Accuracy: **98.62%**

### 📸 Output Screenshot
![Task 4]<img width="1920" height="924" alt="Image" src="https://github.com/user-attachments/assets/d53a6744-9075-489f-84ca-06e153e86d7d" />[](url)

---

## 📈 Results Summary

| # | Task | Algorithm | Accuracy | Status |
|---|------|-----------|----------|--------|
| 1 | Spam Email Classifier | Naive Bayes + TF-IDF | **97.58%** | ✅ Done |
| 2 | Face Detection App | Haar Cascade (OpenCV) | Real-time | ✅ Done |
| 3 | Sentiment Analysis | Logistic Regression + TF-IDF | **~83%** | ✅ Done |
| 4 | Digit Recognizer | Deep Neural Network (Keras) | **98.62%** | ✅ Done |

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| Machine Learning | scikit-learn |
| Deep Learning | TensorFlow 2.x / Keras |
| Computer Vision | OpenCV 4.x |
| NLP | NLTK, TF-IDF |
| Data | NumPy, Pandas |
| Visualization | Matplotlib |
| Environment | Local / Google Colab |

</div>

---

## ▶️ How to Run (Full Setup)

```bash
# Step 1: Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-internship-kodbud.git
cd ai-internship-kodbud

# Step 2: Install all dependencies
pip install numpy pandas scikit-learn nltk opencv-python tensorflow matplotlib

# Step 3: Run any task
python task1_spam_classifier.py
python task2_face_detection.py
python task3_sentiment_analysis.py
python task4_digit_recognizer.py
```

---

## 🎥 LinkedIn Submission

All tasks have been submitted as video explanations on LinkedIn as per internship guidelines.

**Video Title:**
> *"Completed AI Internship Tasks – Spam Classifier, Face Detection, Sentiment Analysis & Digit Recognizer | Python AI Projects @Kodbud"*

**Tags:** `#Python` `#MachineLearning` `#ArtificialIntelligence` `#OpenCV` `#TensorFlow` `#NLP` `#DeepLearning` `#Kodbud` `#AIInternship`

---

## 📜 License

This project was built for educational purposes as part of the **Kodbud AI Internship Program**.

---

<div align="center">

**Made with ❤️ during Kodbud AI Internship**

⭐ *If you found this helpful, drop a star!* ⭐

</div>
