
# Day 13: Customer Churn Prediction Script

## Objective
Create a reusable Python function that loads the trained Logistic Regression model and predicts customer churn.

## Implementation
- Loaded the trained model from `models/logistic_regression_model.joblib`.
- Used `data/processed/encoded_churn.csv` to obtain the 30 expected input features.
- Created a reusable `predict_customer()` function.
- Returned the predicted churn status and estimated churn probability.
- Tested the function using two different customer rows from the dataset.

## Test Results

| Dataset row index | Estimated churn probability | Prediction |
|---|---:|---|
| 0 | 61.25% | Likely to churn |
| 100 | 19.23% | Unlikely to churn |

## Key Learnings
- A saved machine learning model can be loaded and reused without retraining it.
- The model expects 30 encoded input features in the correct order.
- `predict()` returns the predicted class.
- `predict_proba()` provides estimated probabilities for the classes.
- The default classification threshold is 0.50.

## Limitations
The current script tests customers using existing rows from the encoded dataset. It does not yet accept raw customer details directly.

The predictions are estimates, not guarantees of future customer behavior.

## Output Files
- `src/predict_customer.py`
- `reports/day13_prediction_findings.md`
