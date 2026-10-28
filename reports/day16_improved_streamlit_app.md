\# Day 16: Improved Streamlit Customer Churn Prediction App



\## Objective

Improve the Streamlit application to collect a more complete customer profile and use the trained Logistic Regression model for churn prediction.



\## Implementation

\- Updated `app.py` with sections for personal information, account information, and subscribed services.

\- Added inputs for gender, senior-citizen status, partner, dependents, tenure, contract, billing, payment method, charges, phone service, internet service, and additional services.

\- Loaded the saved Logistic Regression model.

\- Retrieved the expected feature names from `data/processed/encoded\_churn.csv`.

\- Encoded the customer inputs into the model's 30 expected features.

\- Displayed the predicted churn class, estimated probability, and progress bar.



\## Testing

\- Verified that the encoded dataset contains 30 model features.

\- Ran `python -m py\_compile app.py` successfully with no syntax errors.

\- Tested the application in the browser and confirmed that a prediction was displayed.



\## Key Learnings

\- Building a more complete customer input form with Streamlit.

\- Mapping categorical inputs to the feature names used during model training.

\- Checking Python syntax before committing code.

\- Understanding why model inputs must match the training features.



\## Limitations

\- Feature-name alignment does not by itself guarantee correct feature encoding or realistic predictions.

\- Input consistency and numerical validation can be improved.

\- Predicted probabilities are estimates, not guarantees of customer behavior.



\## Next Steps

\- Validate customer inputs and service combinations.

\- Test feature encoding against known dataset examples.

\- Improve the application's usability and reliability.



