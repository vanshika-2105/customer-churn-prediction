
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the encoded dataset
data = pd.read_csv("data/processed/encoded_churn.csv")

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data loaded successfully!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create a pipeline that scales features and trains the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Train using only the training data
model.fit(X_train, y_train)

print("\nModel training completed successfully!")
print("Model trained on", len(X_train), "customers.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Predict churn for customers in the test dataset
y_pred = model.predict(X_test)

# Calculate evaluation metrics
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

import joblib
from pathlib import Path

# Create the models folder if it doesn't exist
models_folder = Path("models")
models_folder.mkdir(parents=True, exist_ok=True)

# Save the trained pipeline
model_path = models_folder / "logistic_regression_model.joblib"
joblib.dump(model, model_path)

print("\nTrained model saved successfully!")
print("Model location:", model_path)
