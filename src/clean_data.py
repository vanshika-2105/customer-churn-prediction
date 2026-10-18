
import pandas as pd

# Load the original dataset
data = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Convert TotalCharges from text to numbers
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Check the missing values
print("Missing TotalCharges values:", data["TotalCharges"].isna().sum())

# Save the cleaned dataset
output_path = "data/processed/cleaned_churn.csv"
data.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved to:", output_path)
print("Dataset shape:", data.shape)
