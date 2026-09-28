# Task 5: Credit Card Fraud Detection

This repository contains the solution for **Task 5 (Credit Card Fraud Detection)** of the CodSoft Machine Learning Internship.

## Project Description
The goal of this project is to build a machine learning model that detects fraudulent credit card transactions. 
Because fraud cases are extremely rare, the dataset is highly imbalanced. We address this using **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority class during training.

Models compared:
- **Logistic Regression**
- **Random Forest Classifier**

The best model is evaluated using Precision, Recall, and F1-score due to the imbalanced nature of the data.

## Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the `creditcard.csv` dataset in this directory (downloaded from Kaggle). If the dataset is not found, the script will automatically generate a synthetic dataset for demonstration purposes.
3. Run the script:
   ```bash
   python credit_card_fraud_detection.py
   ```

## Files Generated
- `fraud_detection_model.joblib`: The trained and serialized best model.
- `fraud_confusion_matrix.png`: A visual heatmap of the confusion matrix for the best model.
