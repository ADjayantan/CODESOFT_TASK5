"""
Credit Card Fraud Detection - CodSoft Machine Learning Internship (Task 5)
========================================================================

Builds a classifier that identifies fraudulent credit card transactions.

Pipeline
--------
    load/generate data -> scale features -> handle imbalance (SMOTE) -> 
    train models (Logistic Regression, Random Forest) -> evaluate (Precision, Recall, F1) -> 
    save best model.

Usage
-----
    python credit_card_fraud_detection.py

Note: If 'creditcard.csv' is not found in the current directory, a synthetic 
imbalanced dataset simulating credit card transactions will be generated for demonstration.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

DATA_FILE = "creditcard.csv"
MODEL_FILE = "fraud_detection_model.joblib"
CONFUSION_MATRIX_FILE = "fraud_confusion_matrix.png"

def load_or_generate_data():
    if os.path.exists(DATA_FILE):
        print(f"[data] Loading dataset from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
    else:
        print(f"[data] {DATA_FILE} not found. Generating synthetic dataset for demonstration...")
        # Simulating the highly imbalanced PCA-transformed credit card dataset
        X, y = make_classification(n_samples=10000, n_features=30, n_informative=20, 
                                   n_redundant=10, weights=[0.99, 0.01], 
                                   random_state=42, class_sep=0.8)
        
        feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = pd.DataFrame(X, columns=feature_cols)
        
        # Scaling amount/time roughly to look like realistic data
        df['Amount'] = np.abs(df['Amount'] * 100)
        df['Time'] = np.abs(df['Time'] * 10000)
        df['Class'] = y
        
    print("[data] Class balance:")
    print(df['Class'].value_counts())
    return df

def preprocess_data(df):
    print("[preprocess] Scaling Time and Amount features...")
    df = df.copy()
    
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    scores = {
        "name": name,
        "model": model,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "y_pred": y_pred
    }
    
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    return scores

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Genuine", "Fraud"], 
                yticklabels=["Genuine", "Fraud"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close()
    print(f"[save] Confusion matrix saved to {CONFUSION_MATRIX_FILE}")

def main():
    df = load_or_generate_data()
    X, y = preprocess_data(df)
    
    print("[split] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[smote] Handling class imbalance using SMOTE on training data...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"[smote] Resampled training class balance:\n{pd.Series(y_train_resampled).value_counts()}")
    
    results = []
    print("\n[train] Training models...\n")
    for name, model in build_models().items():
        model.fit(X_train_resampled, y_train_resampled)
        results.append(evaluate_model(name, model, X_test, y_test))
        
    best = max(results, key=lambda r: r["f1"])
    print(f"\n[best] The best model is {best['name']} (F1 = {best['f1']:.4f})")
    
    plot_confusion_matrix(y_test, best["y_pred"], best["name"])
    
    joblib.dump(best["model"], MODEL_FILE)
    print(f"[save] Best model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
