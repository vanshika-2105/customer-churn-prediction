
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load the encoded dataset
data = pd.read_csv("data/processed/encoded_churn.csv")

# Separate input features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Create five stratified folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define the two models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=1,
        random_state=42
    )
}

# Metrics to evaluate
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = []

# Evaluate each model using the same five folds
for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean()
    })

# Display average results as percentages
results_df = pd.DataFrame(results)

metric_columns = ["Accuracy", "Precision", "Recall", "F1-score"]
results_df[metric_columns] = results_df[metric_columns] * 100

print("\n5-Fold Cross-Validation Results")
print(results_df.round(2).to_string(index=False))
from pathlib import Path

# Create the reports folder if it does not exist
reports_folder = Path("reports")
reports_folder.mkdir(parents=True, exist_ok=True)

# Save the cross-validation results
results_path = reports_folder / "cross_validation_results.csv"
results_df.to_csv(results_path, index=False)

print("\nCross-validation results saved to:", results_path)