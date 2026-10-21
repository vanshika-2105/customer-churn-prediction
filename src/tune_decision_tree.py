
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the encoded dataset
data = pd.read_csv("data/processed/encoded_churn.csv")

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Use the same split as previous days
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Ready to tune the Decision Tree!")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Different Decision Tree configurations to test
settings = [
    {"max_depth": 3, "min_samples_leaf": 1},
    {"max_depth": 5, "min_samples_leaf": 1},
    {"max_depth": 7, "min_samples_leaf": 1},
    {"max_depth": 10, "min_samples_leaf": 5},
    {"max_depth": None, "min_samples_leaf": 10},
]

results = []

# Train and evaluate each configuration
for setting in settings:
    model = DecisionTreeClassifier(
        max_depth=setting["max_depth"],
        min_samples_leaf=setting["min_samples_leaf"],
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "max_depth": setting["max_depth"],
        "min_samples_leaf": setting["min_samples_leaf"],
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4)
    })

# Display results in a table
results_df = pd.DataFrame(results)

print("\nDecision Tree Tuning Results:")
print(results_df.to_string(index=False))

from pathlib import Path

# Save tuning results in the reports folder
reports_folder = Path("reports")
reports_folder.mkdir(parents=True, exist_ok=True)

results_path = reports_folder / "decision_tree_tuning.csv"
results_df.to_csv(results_path, index=False)

print("\nTuning results saved to:", results_path)
