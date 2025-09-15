import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

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

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data (important for neural network convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Class distribution: {np.bincount(y_train)} (train), {np.bincount(y_test)} (test)")

# 4. Prepare data types for the neural network
dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
print(f"\nData types: {dtypes}")

# 5. Build the Neural Network Model
print("\nBuilding neural network model...")
input_els = []
encoded_els = []

for k, dtype in dtypes:
    input_els.append(Input(shape=(1,), name=k))
    if "int" in dtype:
        # For integer features, use embedding
        max_val = int(X_train[k].max() + 1)
        e = Flatten()(Embedding(max_val, 10)(input_els[-1]))
    else:
        # For float features, use directly
        e = input_els[-1]
    encoded_els.append(e)

# Concatenate all encoded features
encoded_els = concatenate(encoded_els)

# Build the network architecture
layer1 = Dense(128, activation="relu")(encoded_els)
layer1 = Dropout(0.3)(layer1)
layer2 = Dense(64, activation="relu")(layer1)
layer2 = Dropout(0.3)(layer2)
layer3 = Dense(32, activation="relu")(layer2)
out = Dense(1, activation="sigmoid")(layer3)

# Create and compile the model
model = Model(inputs=input_els, outputs=out)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

print(model.summary())

# 6. Train the model
print("\nTraining neural network...")
history = model.fit(
    [X_train[k].values for k, dtype in dtypes],
    y_train,
    epochs=50,
    batch_size=512,
    shuffle=True,
    validation_data=([X_test[k].values for k, dtype in dtypes], y_test),
    verbose=1
)

# 7. Model Evaluation
print("\nEvaluating model...")
y_pred_proba = model.predict([X_test[k].values for k, dtype in dtypes])
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

# 8. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('nn_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'DDoS'],
            yticklabels=['Benign', 'DDoS'])
plt.title('Neural Network Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. SHAP Analysis (using a sample for efficiency)
print("\nGenerating SHAP explanations (using sample for efficiency)...")
# Take a sample for SHAP analysis to make it faster
sample_size = min(1000, len(X_test))
X_test_sample = X_test.iloc[:sample_size]


# Create a wrapper function for the model predictions
def model_predict(X_array):
    # Convert the array back to the expected input format
    if isinstance(X_array, pd.DataFrame):
        X_array = X_array.values

    # Prepare input in the format expected by the model
    input_data = []
    for i, (k, dtype) in enumerate(dtypes):
        input_data.append(X_array[:, i:i + 1])

    return model.predict(input_data)


# Create SHAP explainer
explainer = shap.KernelExplainer(model_predict, X_test_sample.values[:100])  # Use small background dataset

# Calculate SHAP values for a sample
shap_values = explainer.shap_values(X_test_sample.values[:100])

# Summary plot
shap.summary_plot(shap_values, X_test_sample.values[:100], feature_names=X.columns.tolist(), show=False)
plt.title("Neural Network SHAP Feature Importance")
plt.tight_layout()
plt.savefig('nn_shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Feature Importance Analysis
# Get the weights from the first dense layer to understand feature importance
first_layer_weights = model.layers[-5].get_weights()[0]  # Get weights from the first Dense layer

# Calculate feature importance by averaging absolute weights
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.mean(np.abs(first_layer_weights), axis=1)
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (from Neural Network weights):")
print(feature_importance.head(10))

print("Neural network analysis complete!")
