
# Day 18 - Automated Testing of Customer Feature Encoding

## Objective
Create automated tests to verify that customer information is correctly converted into the 30 features required by the churn prediction model.

## Work Completed

1. Installed pytest in the virtual environment.
2. Created src/feature_encoder.py.
3. Moved customer feature encoding into a reusable function.
4. Connected the encoder to app.py.
5. Created tests/test_feature_encoding.py.
6. Tested the number and order of generated features.
7. Tested categorical-to-numerical feature conversion.
8. Fixed feature mapping and indentation issues.

## Test Results

Command:
python -m pytest -v

Results:
- test_encoder_creates_30_features: PASSED
- test_encoder_converts_customer_values_correctly: PASSED

Total tests: 2
Passed: 2
Failed: 0

## Key Learning

- Automated testing using pytest.
- Python assert statements.
- Reusable Python functions.
- Feature encoding and mapping.
- Importance of correct indentation.
- Testing Pandas DataFrames.

## Limitations

Only two test cases have been implemented so far.
More tests are needed for different customer combinations,
invalid inputs, and edge cases.

## Conclusion

The customer feature encoder successfully generates
the expected 30 features and correctly converts the
tested customer information into numerical values.
