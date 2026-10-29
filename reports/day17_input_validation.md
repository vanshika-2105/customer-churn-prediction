\# Day 17: Customer Input Validation



\## Objective

Improve the reliability of the Customer Churn Prediction application by validating combinations of customer service selections.



\## Implementation

\- Updated `app.py` to restrict internet-related service choices based on Internet Service.

\- When Internet Service is `No`, the six related service fields automatically use `No internet service`.

\- When Internet Service is `DSL` or `Fiber optic`, those fields offer `No` and `Yes`.

\- Updated Phone Service handling so that when Phone Service is `No`, Multiple Lines offers only `No phone service`.

\- Preserved the model's expected 30-feature order.



\## Testing

\- Ran `python -m py\_compile app.py` successfully.

\- Confirmed the internet-service restrictions in the Streamlit browser.

\- Confirmed the phone-service and multiple-lines restrictions in the browser.

\- Tested the saved Logistic Regression model using encoded customer row 101.



\## Sample Prediction Test

\- Customer row: 101

\- Actual churn: No

\- Predicted churn: No

\- Estimated churn probability: 19.23%

\- Model input features: 30



The sample prediction matched the actual label for this record. This single example does not establish overall model reliability.



\## Key Learnings

\- Validating dependent form inputs.

\- Checking categorical combinations against the original dataset.

\- Testing a saved classification model with encoded records.

\- Understanding the difference between a successful test and overall model performance.



\## Limitations

\- Further automated tests are needed for all feature mappings.

\- Numerical inputs and consistency between charges and tenure could be validated further.

\- Predicted probabilities are estimates, not guarantees of customer behavior.



\## Next Steps

\- Add automated tests for the input encoding.

\- Improve numeric validation.

\- Continue evaluating prediction quality on held-out customer records.



