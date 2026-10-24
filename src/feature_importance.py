
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the encoded dataset
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

# Create and train the Logistic Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Get the trained Logistic Regression classifier
classifier = model.named_steps["classifier"]

# Pair each feature with its coefficient
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": classifier.coef_[0]
})

# Sort by absolute coefficient magnitude
importance_df["Absolute_Importance"] = (
    importance_df["Coefficient"].abs()
)

importance_df = importance_df.sort_values(
    by="Absolute_Importance",
    ascending=False
)

print("\nTop 10 Features by Absolute Coefficient")
print(
    importance_df[
        ["Feature", "Coefficient"]
    ].head(10).round(4).to_string(index=False)
)

print("\nInterpretation:")
print("Positive coefficient: pushes prediction toward churn.")
print("Negative coefficient: pushes prediction away from churn.")

from pathlib import Path

# Create the reports folder if it does not exist
reports_folder = Path("reports")
reports_folder.mkdir(parents=True, exist_ok=True)

# Save all feature coefficients
results_path = reports_folder / "feature_importance.csv"
importance_df.to_csv(results_path, index=False)

print("\nFeature importance results saved to:", results_path)
