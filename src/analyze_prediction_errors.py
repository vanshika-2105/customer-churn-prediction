
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the encoded customer churn dataset
data = pd.read_csv("data/processed/encoded_churn.csv")

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Use the same train-test split as previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Calculate churn probabilities
churn_probabilities = model.predict_proba(X_test)[:, 1]

# Compare two thresholds
thresholds = [0.3, 0.5]

for threshold in thresholds:
    predictions = (churn_probabilities >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n--- Threshold: {threshold} ---")
    print(f"True Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives  (TP): {tp}")

    print(f"Correct predictions: {tn + tp}")
    print(f"Incorrect predictions: {fp + fn}")

from pathlib import Path

# Create a table of error counts for each threshold
error_results = []

for threshold in thresholds:
    predictions = (churn_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    error_results.append({
        "Threshold": threshold,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
        "Correct_Predictions": tn + tp,
        "Incorrect_Predictions": fp + fn
    })

error_df = pd.DataFrame(error_results)

reports_folder = Path("reports")
reports_folder.mkdir(parents=True, exist_ok=True)

results_path = reports_folder / "prediction_error_analysis.csv"
error_df.to_csv(results_path, index=False)

print("\nError analysis saved to:", results_path)
