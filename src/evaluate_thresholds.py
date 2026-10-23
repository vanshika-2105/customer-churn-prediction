
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load the encoded customer churn dataset
data = pd.read_csv("data/processed/encoded_churn.csv")

# Separate input features and target
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

# Create and train the Logistic Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Get the probability that each test customer will churn
churn_probabilities = model.predict_proba(X_test)[:, 1]

# Compare different classification thresholds
thresholds = [0.3, 0.4, 0.5, 0.6]

results = []

for threshold in thresholds:
    # Predict churn when probability meets the threshold
    predictions = (churn_probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "Precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "F1-score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        )
    })

# Display the comparison
results_df = pd.DataFrame(results)

print("\nLogistic Regression: Threshold Comparison")
print(results_df.to_string(index=False))

from pathlib import Path

# Create the reports folder if needed
reports_folder = Path("reports")
reports_folder.mkdir(parents=True, exist_ok=True)

# Save threshold comparison results
results_path = reports_folder / "threshold_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\nThreshold results saved to:", results_path)
