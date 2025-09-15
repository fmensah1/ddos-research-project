"""
DDoS Detection Analysis using Logistic Regression
Author: Felix Mensah
Date: 9/03/2025

Description:
This script analyzes a network traffic dataset (CIC-DDoS2019) to build a baseline model for DDoS detection.
It focuses on interpreting the model to identify key differentiating features between benign and DDoS traffic.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb

# 1. Load and Prepare Data
print("Loading and preparing data...")
file_path = 'ddos_dataset.csv'
df = pd.read_csv(file_path, low_memory=False)

# Identify the target label column
target_column = 'Label'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# 2. Preprocessing
# Separate features and target, then convert labels to numerical values
y = df[target_column]
X = df.drop(columns=[target_column])
if y.dtype == 'object':
    y = y.astype('category').cat.codes  # Convert text labels to numbers

# Keep only numeric features for this baseline model
X = X.select_dtypes(include=[np.number])

# 3. Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True,
    "early_stopping_round": 50,
}

model = lgb.train(
    params,
    d_train,
    10000,
    valid_sets=[d_test],
)

# 4. Model Evaluation
# Get probability predictions
y_pred_proba = model.predict(X_test)

y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

print(shap_values)
shap.summary_plot(shap_values, X)
plt.show()

"https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20LightGBM.html#Explain-predictions"