# Handwritten Digit Classification using Support Vector Machine (SVM)

This project implements a **machine learning model that recognizes handwritten digits (0–9)** using a **Support Vector Machine (SVM)** classifier. The model is trained on the **Scikit-learn Digits dataset**, which contains small grayscale images of handwritten numbers.

The project demonstrates a complete **machine learning workflow**, including data preprocessing, model training, prediction, evaluation, and visualization.

---

# Project Overview

Handwritten digit recognition is a classic **computer vision and machine learning problem** used in applications such as:

- Postal code recognition
- Bank check processing
- Document digitization
- Automated form processing

In this project, the model learns to classify digits by analyzing the **pixel values of handwritten images**.

---

# Dataset

The dataset used is the **Digits Dataset** from Scikit-learn.

Dataset characteristics:

| Property | Value |
|--------|------|
| Total Samples | 1797 |
| Image Size | 8 × 8 pixels |
| Features per Image | 64 |
| Classes | Digits 0–9 |

Each image is flattened into **64 numerical pixel features** before being used by the machine learning model.

---

# Technologies Used

**Programming Language**

- Python

**Libraries**

- NumPy
- Scikit-learn
- Matplotlib

---

# Machine Learning Model

This project uses a **Support Vector Machine (SVM)** classifier with an **RBF kernel**.

SVM works by finding the optimal **decision boundary** that separates classes in a dataset while maximizing the margin between them.

The trained model learns patterns in the pixel values of images to correctly classify handwritten digits.

---

## Project Workflow

1. Load Dataset
The handwritten digits dataset is loaded from Scikit-learn.  
Each image is an **8×8 grayscale image** representing a handwritten digit.

2. Data Visualization
Sample images from the dataset are displayed using Matplotlib to understand how the handwritten digits look.
```python
ax.imshow(digits.images[i], cmap='gray')
```
3. Train-Test Split
The dataset is split into training and testing sets to evaluate model performance.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```
-70% training data
-30% testing data

4. Feature Scaling
Pixel values are standardized using StandardScaler to improve SVM performance.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
5. Model Training
A Support Vector Machine (SVM) classifier with an RBF kernel is trained on the scaled data.
```python
model = SVC(kernel='rbf', gamma=0.001, C=10)
model.fit(X_train_scaled, y_train)
```

6. Prediction
 The trained model predicts digit labels for the test dataset.
```python
y_pred = model.predict(X_test_scaled)
```
7.Model Evaluation
Model performance is evaluated using a classification report and confusion matrix.
```python
print(classification_report(y_test, y_pred))
```
These metrics include:
Precision
Recall
F1-score
Accuracy


## **Results**

## Sample Digits

![Digits](sample_images.png)

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


