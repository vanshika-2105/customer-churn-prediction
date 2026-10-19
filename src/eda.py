import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------------------
# STEP 1: Load the cleaned dataset
# -----------------------------------------

data = pd.read_csv("data/processed/cleaned_churn.csv")

# Create reports folder if it doesn't exist
reports_folder = Path("reports")
reports_folder.mkdir(exist_ok=True)

# Set the style of our graphs
sns.set_theme(style="whitegrid")


# -----------------------------------------
# GRAPH 1: Customer Churn Distribution
# -----------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(data=data, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.tight_layout()

# Save Graph 1
plt.savefig(
    reports_folder / "churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------------------
# GRAPH 2: Customer Churn by Contract Type
# -----------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=data,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()

# Save Graph 2
plt.savefig(
    reports_folder / "churn_by_contract.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------------------
# GRAPH 3: Monthly Charges by Churn Status
# -----------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=data,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")

plt.tight_layout()

# Save Graph 3
plt.savefig(
    reports_folder / "monthly_charges_by_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------------------
# FINAL MESSAGE
# -----------------------------------------

print("\nEDA graphs generated successfully!")
print("All three graphs saved in the reports folder.")

# -----------------------------------------
# STEP 5: Calculate churn by contract type
# -----------------------------------------

print("\nChurn percentage by contract type:")

# Calculate the percentage of customers who churn
churn_percentage = (
    data.groupby("Contract")["Churn"]
    .apply(lambda churn: (churn == "Yes").mean() * 100)
    .round(2)
)

print(churn_percentage)
