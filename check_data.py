
import pandas as pd

# Location of our dataset
file_path = "data/raw/Telco-Customer-Churn.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# Display the first five rows
print("First five rows:")
print(data.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(data.shape)

# Display all column names
print("\nColumn names:")
print(data.columns.tolist())
