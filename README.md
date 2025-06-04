# 🧠 AI-Based Pneumonia Detection Using Deep Learning

This project focuses on developing an intelligent system using a Convolutional Neural Network (CNN) to automatically detect pneumonia from chest X-ray images. The aim is to assist healthcare professionals by providing fast and accurate predictions, reducing diagnostic workload and human error.

---

## 📁 Dataset

The dataset used is a publicly available chest X-ray dataset, which is structured into three directories:

* **Training Set (`train/`)**
* **Validation Set (`val/`)**
* **Testing Set (`test/`)**

Each directory contains two categories:

* `NORMAL` — Chest X-rays with no signs of pneumonia
* `PNEUMONIA` — Chest X-rays showing signs of pneumonia

---

## 🔧 Data Preprocessing & Augmentation

To improve the model's ability to generalize to new data, several preprocessing and augmentation techniques were used:

* Rescaling image pixel values to the range \[0, 1]
* Resizing all images to **150x150** pixels
* Applying **zoom**, **horizontal flipping**, and **shearing** transformations
* Organizing data using TensorFlow's `ImageDataGenerator`

---

## 🧱 CNN Model Architecture

The model was built using **TensorFlow** and **Keras Sequential API**, which allows easy stacking of layers. The architecture includes:

* Multiple **Conv2D** layers with ReLU activation
* **MaxPooling2D** layers to reduce spatial dimensions
* **Flatten** layer to convert 2D feature maps into 1D feature vectors
* Fully connected **Dense** layers for classification
* **Sigmoid** output layer for binary classification

---

## 🏋️ Model Training

* Loss Function: `binary_crossentropy`
* Optimizer: `Adam`
* Evaluation Metric: `accuracy`
* Number of Epochs: **10**
* Batch Size: `32`

The model was trained on the augmented dataset using the training and validation sets to prevent overfitting and improve generalization.

---

## 📈 Model Evaluation

After training, the model was evaluated on the test dataset using various performance metrics:

* ✅ **Accuracy**
* 🎯 **Precision**
* 🔁 **Recall**
* 📊 **F1-Score**
* 🔍 **Confusion Matrix**
* 📄 **Classification Report**

These metrics provided a detailed insight into how well the model performed on unseen data.

---

## 🌐 Web Deployment with Flask

The model was deployed as a **Flask web application**. Key features include:

* Uploading a chest X-ray image via the browser
* Preprocessing the image in real-time
* Generating a prediction ("Pneumonia Detected" or "Normal")
* Displaying the result along with the uploaded image

To run the app:

```bash
git clone https://github.com/mhd-sadiq/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/`

---

## 🗂️ Project Structure

pneumonia_detection_project/
├── chest_xray/              # Dataset
│   ├── train/
│   ├── test/
│   └── val/
│
├── src/                     # Source code
│   ├── data_preprocessing.py
│   └── model_training.py
│
├── app/                     # Flask app
│   └── app.py
│
└── requirements.txt         # Dependencies

## 📌 Conclusion

This project demonstrates a complete end-to-end deep learning pipeline — from preprocessing medical imaging data and training a CNN model to evaluating its performance using classification metrics and deploying it as a web application with Flask. The model, trained over 10 epochs, achieved high accuracy and is capable of making real-time pneumonia predictions from chest X-rays. It shows how AI can be practically applied to enhance diagnostic processes in the healthcare sector.

> **Developed by:** Sadiq  
> **Institution:** Learn Logic AI  
> **Date:** May 15, 2025
---



