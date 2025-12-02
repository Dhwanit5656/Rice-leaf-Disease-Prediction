# Rice Leaf Disease Prediction (Deep Learning – CNN)

## 🌾 Project Overview

This project develops a deep learning-based image classification system designed to accurately identify major rice leaf diseases from plant images. By leveraging a custom Convolutional Neural Network (CNN) and robust data augmentation techniques, the system achieves high accuracy, providing a valuable tool for early agricultural intervention and crop management.

The system is capable of detecting and classifying the following three common rice leaf diseases:
1.  **Bacterial Leaf Blight**
2.  **Brown Spot**
3.  **Leaf Smut**

## ✨ Key Features

* **Custom CNN Architecture:** Implementation of a simple, efficient CNN tailored for image classification.
* **Data Augmentation:** Techniques (rotation, shifting, shearing, zooming) were heavily used to artificially expand the small dataset (119 images) and improve model generalization and robustness.
* **Transfer Learning & Hyperparameter Tuning:** Experiments with pre-trained models and tuning were conducted to maximize performance and minimize overfitting on the limited data.
* **High Accuracy:** Achieved a classification accuracy of **98.33%** with strong performance metrics across all disease categories, enabling reliable deployment for real-world agricultural assistance tools.
* **Deployment Ready:** The model is designed for reliable deployment in real-world agricultural assistance tools.

## 🚀 Results

| Metric | Value |
| :--- | :--- |
| **Classification Accuracy** | **98.33%** |
| **Model Type** | Custom CNN / Transfer Learning |
| **Target Classes** | 3 (Bacterial Blight, Brown Spot, Leaf Smut) |

## ⚙️ Technologies Used

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Image Processing:** OpenCV
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib

## 📂 Dataset

The model was trained on a small initial dataset of 119 images across the three disease categories and healthy leaves.

**Download the Dataset:**
You can download the original image dataset used for this project from the following Google Drive link:

[Rice Leaf Disease Dataset](https://drive.google.com/drive/folders/1JancWacS_i6Y0DvPlJ1ptotcGOYwubf6?usp=drive_link)

---

