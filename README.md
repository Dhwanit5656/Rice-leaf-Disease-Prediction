# 🌾 Rice Leaf Disease Prediction (Deep Learning – CNN + Transfer Learning)

## 📌 Project Overview

This project develops a deep learning-based image classification system to accurately identify major rice leaf diseases from plant images. Given an extremely small dataset of just **119 images**, the core challenge was building a model that generalizes well without overfitting.

The approach followed an **iterative development strategy** — progressing from a simple baseline CNN to a robust Transfer Learning solution using **EfficientNetB0** — ultimately achieving near-perfect classification accuracy.

The system classifies rice leaf images into **3 disease categories:**

| # | Disease Class |
|---|--------------|
| 1 | 🔴 Bacterial Leaf Blight |
| 2 | 🟤 Brown Spot |
| 3 | 🟢 Leaf Smut |

---

## 🧩 Key Challenge

> **Extremely small dataset (N = 119 images)** — the primary obstacle throughout this project, causing severe overfitting in early models and requiring aggressive regularization, data augmentation, and ultimately Transfer Learning to overcome.

---

## 🔬 Iterative Model Development

Three distinct modeling attempts were made, each building on the failures of the previous:

### Attempt 1 — Baseline Custom CNN (No Augmentation)

A simple Sequential CNN with 3 convolutional blocks, BatchNormalization, MaxPooling, Flatten, and two Dense layers with Dropout.

**Architecture:**
```
Rescaling → Conv2D(8) → BN → MaxPool
          → Conv2D(16) → BN → MaxPool
          → Conv2D(32) → BN → MaxPool
          → Flatten → Dense(32) → Dropout(0.3)
          → Dense(16) → Dropout(0.5)
          → Dense(3, softmax)
```

| Result | Value |
|--------|-------|
| Validation Accuracy | ~22% |
| Outcome | Severe overfitting — training accuracy rose steadily while validation accuracy collapsed |

---

### Attempt 2 — CNN + Data Augmentation

The same CNN architecture, now paired with aggressive in-model data augmentation layers to artificially expand the training distribution:

- `RandomFlip` (horizontal)
- `RandomRotation`
- `RandomZoom`
- `RandomContrast`
- `RandomBrightness`

Trained with `Adam (lr=1e-4)`, `sparse_categorical_crossentropy`, and **Early Stopping** (patience monitoring validation accuracy).

| Result | Value |
|--------|-------|
| Validation Accuracy | ~78–80% |
| Improvement over Baseline | +56% |
| Outcome | Significant improvement; overfitting reduced but not eliminated |

---

### Attempt 3 — EfficientNetB0 Transfer Learning + Augmentation ✅ *(Final Model)*

Replaced the custom feature extractor with a **pre-trained EfficientNetB0** backbone (weights from ImageNet), frozen during initial training, combined with the same augmentation pipeline.

**Architecture:**
```
Data Augmentation Layers
→ EfficientNetB0 (frozen, pre-trained ImageNet weights)
→ GlobalAveragePooling2D
→ Dense layers + Dropout
→ Dense(3, softmax)
```

| Result | Value |
|--------|-------|
| Validation Accuracy | ~96–98% |
| Improvement over Model 2 | +18–20% |
| Improvement over Baseline | +74–76% |
| Outcome | Training and validation curves closely overlapped — no overfitting. Selected as the **final production model**. |

---

## 📊 Final Model Evaluation (Test Set)

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bacterial Leaf Blight | 1.00 | 1.00 | 1.00 | 14 |
| Brown Spot | 0.88 | 1.00 | 0.93 | 7 |
| Leaf Smut | 1.00 | 0.91 | 0.95 | 11 |

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.87%** |
| Macro Average F1-Score | 0.96 |
| Weighted Average F1-Score | 0.97 |

### Highlights
- **Bacterial Leaf Blight** — Perfect precision and recall (1.00 / 1.00)
- **Brown Spot** — 100% recall even with the fewest samples (7 support)
- **Leaf Smut** — Balanced performance (precision 1.00, recall 0.91)
- Final validation accuracy improved by **more than 74%** over the initial baseline

---

## ⚙️ Tech Stack

```
Python 3.12 (Google Colab – GPU T4)
├── Deep Learning    : TensorFlow / Keras
├── Backbone         : EfficientNetB0 (ImageNet pre-trained)
├── Augmentation     : Keras RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
├── Evaluation       : scikit-learn (classification_report, confusion_matrix, accuracy_score)
└── Visualization    : Matplotlib
```

---

## 📂 Dataset

| Attribute | Details |
|-----------|---------|
| Total Images | 119 |
| Classes | 3 (Bacterial Leaf Blight, Brown Spot, Leaf Smut) |
| Image Size | 224 × 224 px |
| Train / Val Split | 70% / 30% |
| Batch Size | 16 |

📥 **Download Dataset:**  
[Rice Leaf Disease Dataset – Google Drive](https://drive.google.com/drive/folders/1JancWacS_i6Y0DvPlJ1ptotcGOYwubf6?usp=drive_link)

---

## 🚀 How to Run

1. **Open in Google Colab** (recommended — uses GPU T4 for faster training)

2. **Mount Google Drive and place the dataset:**
   ```
   /content/drive/MyDrive/Colab Notebooks/Data/Train/
   ├── Bacterial_leaf_blight/
   ├── Brown_spot/
   └── Leaf_smut/
   ```

3. **Install dependencies** (pre-installed in Colab):
   ```bash
   pip install tensorflow scikit-learn matplotlib
   ```

4. **Run the notebook cells in order:**
   - Section 1: Setup & Data Loading
   - Section 2: Iterative Model Development (Attempts 1 → 2 → 3)
   - Section 3: Final Model Evaluation & Summary

---

## 📁 Project Structure

```
rice-leaf-disease-prediction/
│
├── Rice_Leaf_Disease_Prediction.ipynb   # Main notebook (all 3 model attempts)
└── README.md                            # Project documentation
```

---

## 💡 Key Takeaways

- With only 119 training images, a custom CNN from scratch was fundamentally insufficient.
- **Data Augmentation alone** improved accuracy by ~56% but wasn't enough for deployment-grade performance.
- **Transfer Learning (EfficientNetB0)** was the critical breakthrough — leveraging pre-learned ImageNet features to compensate for the tiny dataset.
- The final model is suitable for real-world agricultural applications including disease diagnosis apps, farmer-assistance systems, and web-based crop disease prediction platforms.

---

## 🔮 Future Scope

- Expand the dataset with more annotated rice leaf images for further improvement
- Experiment with fine-tuning the EfficientNetB0 backbone layers
- Deploy as a mobile or web application for field use by farmers
- Extend classification to additional rice diseases beyond the current 3 classes
