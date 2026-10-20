
import pandas as pd
from pathlib import Path

# Load the cleaned dataset
data = pd.read_csv("data/processed/cleaned_churn.csv")

# Remove customerID because it is an identifier, not a useful model feature
data = data.drop(columns=["customerID"])

# Convert the target column into numbers: No = 0, Yes = 1
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical input columns into numerical columns
data = pd.get_dummies(data, drop_first=True, dtype=int)

# Create the processed folder if needed
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# Save the encoded dataset
output_path = output_folder / "encoded_churn.csv"
data.to_csv(output_path, index=False)

print("Feature engineering completed!")
print("Encoded dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nChurn values:")
print(data["Churn"].value_counts())
print("\nSaved to:", output_path)

# Separate input features (X) and target (y)
X = data.drop(columns=["Churn"])
y = data["Churn"]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

print("\nFeature column names:")
print(X.columns.tolist())

print("\nFirst five target values:")
print(y.head().tolist())

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the sizes of the datasets
print("\nTraining and testing split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nChurn distribution in training data:")
print(y_train.value_counts())

print("\nChurn distribution in testing data:")
print(y_test.value_counts())

