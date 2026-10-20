
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the encoded dataset
data = pd.read_csv("data/processed/encoded_churn.csv")

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Recreate the same training/testing split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model using training data
tree_model.fit(X_train, y_train)

print("\nDecision Tree training completed!")
print("Model trained on", len(X_train), "customers.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Predict churn for test customers
tree_predictions = tree_model.predict(X_test)

# Calculate evaluation metrics
print("\n--- Decision Tree Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, tree_predictions), 4))
print("Precision:", round(precision_score(y_test, tree_predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, tree_predictions, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, tree_predictions, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_predictions))

print("\nClassification Report:")
print(classification_report(y_test, tree_predictions, zero_division=0))
