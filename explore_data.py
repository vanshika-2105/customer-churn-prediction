
import pandas as pd

# Load the dataset
data = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# 1. Check dataset size
print("Dataset shape:")
print(data.shape)

# 2. Check column data types
print("\nColumn data types:")
print(data.dtypes)

# 3. Count missing values in each column
print("\nMissing values:")
print(data.isnull().sum())

# 4. Count duplicate rows
print("\nDuplicate rows:")
print(data.duplicated().sum())

# 5. Check the target column
print("\nChurn counts:")
print(data["Churn"].value_counts())

# 6. Check churn percentages
print("\nChurn percentages:")
print(data["Churn"].value_counts(normalize=True) * 100)

# Check the TotalCharges column

print("\nTotalCharges values that are blank or contain only spaces:")

blank_charges = data[
    data["TotalCharges"].str.strip() == ""
]

print(blank_charges[["customerID", "tenure", "TotalCharges", "Churn"]])

print("\nNumber of blank TotalCharges values:")
print(len(blank_charges))

