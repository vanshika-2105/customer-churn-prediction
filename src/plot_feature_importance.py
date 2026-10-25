
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load feature importance results from Day 11
data = pd.read_csv("reports/feature_importance.csv")

# Select the 10 most important features
top_features = data.head(10).copy()

# Sort them for a clear horizontal bar chart
top_features = top_features.sort_values(
    by="Coefficient",
    ascending=True
)

# Use different bars for positive and negative coefficients
colors = [
    "steelblue" if value > 0 else "orange"
    for value in top_features["Coefficient"]
]

# Create the chart
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"],
    color=colors
)

plt.axvline(x=0, color="black", linewidth=0.8)
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 10 Features Related to Customer Churn")
plt.tight_layout()

# Create reports folder if needed
Path("reports").mkdir(parents=True, exist_ok=True)

# Save the chart
chart_path = "reports/feature_importance.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")

print(f"Chart saved successfully to: {chart_path}")

# Display the chart
plt.show()
