# 🔢 Handwritten Digit Classifier using ANN

A deep learning web application that classifies single handwritten digits from 0 to 9 using an Artificial Neural Network (ANN) trained on the MNIST dataset.

The project combines:

- 🧠 Model training in Python with TensorFlow and Keras
- 🌐 A Streamlit web interface for real-time predictions

---

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow:

1. Load and explore the MNIST dataset
2. Preprocess handwritten digit images
3. Build and train an ANN model
4. Evaluate model performance
5. Save the trained model
6. Deploy the model through a Streamlit application

Users can upload a handwritten digit image and receive:

- The predicted digit
- A confidence score
- A probability distribution for all classes from 0 to 9

---

## 🚀 Features

- Upload handwritten digit images in PNG, JPG, or JPEG format
- Automatic grayscale conversion
- Resize images to 28×28 pixels
- Automatic background inversion for better MNIST-style preprocessing
- Real-time prediction using the trained ANN
- Confidence score display
- Probability chart for digits 0 through 9
- Clean and interactive Streamlit UI

---

## 🧠 Model Architecture

The model is a simple feedforward neural network built with Keras:

- Input layer: 28×28 grayscale image
- Flatten layer
- Dense layer with 128 neurons and ReLU activation
- Output layer with 10 neurons and Softmax activation

This architecture is suitable for classifying the 10 handwritten digit classes.

---

## 📊 Dataset

The project uses the MNIST dataset:

- 60,000 training images
- 10,000 testing images
- 10 classes: digits 0–9

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶️ Installation and Run

### 1. Clone the repository

```bash
git clone https://github.com/harshita10sharma/mnist_digit_classifier_ann.git
cd mnist_digit_classifier_ann
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
mnist_digit_classifier_ann/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── mnist_ann.keras
└── notebooks/
    └── mnist_ann.ipynb
```

---

## ⚠️ Limitations

This application is designed specifically for recognizing single handwritten digits.

- ✅ Supported: digits 0–9
- ❌ Not supported: multi-digit numbers or complex handwriting patterns

---

## 👩‍💻 Author

Harshita Sharma

- LinkedIn: https://linkedin.com/in/harshita10sharma
- GitHub: https://github.com/harshita10sharma

---

## 💡 Notes

For best results, upload clear, centered handwritten digits with a simple background. The model performs best when the input image closely resembles the MNIST format.
