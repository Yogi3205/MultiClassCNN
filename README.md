# 🧠 Multi-Class Image Classifier using CNN

A deep learning project that classifies images into multiple categories using a Convolutional Neural Network (CNN) with a Streamlit web interface.

## 🚀 Features

- Multi-class image classification (Cat, Dog, Human, Car)
- CNN-based deep learning model
- Real-time prediction via Streamlit app
- Upload image → Get instant result
- Confidence-based prediction
- Lightweight and beginner-friendly

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Streamlit

---

## 📂 Project Structure
MultiClassCNN/
│
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── predict.py # Prediction script
├── test_image.jpg # Sample test image
├── requirements.txt # Dependencies
├── README.md # Project documentation

---

## 📊 Dataset

The model is trained on images from the following classes:

- 🐱 Cat  
- 🐶 Dog  
- 👤 Human  
- 🚗 Car  

> ⚠️ Dataset not included due to size limitations.  
Download datasets from Kaggle or other sources and place them in:
dataset/train/
dataset/test/

---

## ⚙️ Installation

1. Clone the repository
git clone https://github.com/YOUR_USERNAME/MultiClassCNN.git

cd MultiClassCNN

2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

---

## 🧠 Train the Model
python train_model.py
---

## 🔎 Run Prediction Script
python predict.py
---

## 🌐 Run Streamlit Web App
streamlit run app.py

Open in browser:
http://localhost:8501

Upload an image and get prediction instantly 🎉

---

## 📈 Model Details

- CNN architecture with convolution + pooling layers
- Softmax output for multi-class classification
- Image size: 150 × 150
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## 🧑‍💻 Author

**Yogi Kevadiya**

---

## ⭐ Future Improvements

- Transfer Learning (MobileNet / ResNet)
- More classes
- Real-time webcam detection
- Deployment to cloud

---

## 📜 License

This project is for educational purposes.