
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import xgboostAnalysis
import shap

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
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
}
model = xgboost.train(
    params,
    d_train,
    5000,
    evals=[(d_test, "test")],
    verbose_eval=100,
    early_stopping_rounds=20,
)

# 4. Model Evaluation
y_pred_proba = model.predict(d_test)
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

# 6. Feature Importance from XGBoost
xgb_importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': list(xgb_importance.keys()),
    'importance': list(xgb_importance.values())
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (XGBoost):")
print(importance_df.head(10))