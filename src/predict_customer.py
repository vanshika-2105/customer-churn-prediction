
import joblib
import pandas as pd
from pathlib import Path

# Load the trained model
model_path = Path("models/logistic_regression_model.joblib")
model = joblib.load(model_path)

# Load the dataset to get the exact feature names
data = pd.read_csv("data/processed/encoded_churn.csv")
feature_names = data.drop(columns=["Churn"]).columns


def predict_customer(customer):
    """
    Predict whether a customer is likely to churn.

    customer must be a dictionary containing all 30 encoded features.
    """
    # Convert the customer details into a DataFrame
    customer_df = pd.DataFrame([customer])

    # Ensure features appear in the correct order
    customer_df = customer_df.reindex(columns=feature_names)

    # Check for missing feature values
    if customer_df.isnull().any().any():
        raise ValueError("Customer data is missing one or more features.")

    # Predict churn and calculate its probability
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    return {
        "prediction": "Likely to churn" if prediction == 1
        else "Unlikely to churn",
        "churn_probability": probability
    }


# Test with an example customer from the dataset
example_customer = data.drop(columns=["Churn"]).iloc[100].to_dict()

result = predict_customer(example_customer)

print("Customer Churn Prediction")
print("-------------------------")
print("Prediction:", result["prediction"])
print(f"Estimated churn probability: {result['churn_probability']:.2%}")
