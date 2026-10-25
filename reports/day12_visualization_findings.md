
# Day 12: Feature Importance Visualization

## Objective
Visualize the top 10 features associated with customer churn using the coefficients of the trained Logistic Regression model.

## Visualization
The chart is saved at `reports/feature_importance.png`.

## Key Findings

1. **Tenure:** Has the strongest negative coefficient among the top 10 features. Longer customer tenure is associated with a lower predicted likelihood of churn.
2. **Monthly Charges:** Has a negative coefficient in this model after accounting for the other features.
3. **Fiber Optic Internet:** Has a positive coefficient, indicating an association with a higher predicted likelihood of churn.
4. **Contract Type:** One-year and two-year contracts have negative coefficients. Longer-term contracts are associated with a lower predicted likelihood of churn.
5. **Total Charges:** Has a positive coefficient in this model.
6. **Streaming Services:** The StreamingTV_Yes and StreamingMovies_Yes features have positive coefficients.
7. **Paperless Billing and Multiple Lines:** Both have positive coefficients in this model.

## How to Read the Chart

- Positive coefficients extend to the right and push model predictions toward churn.
- Negative coefficients extend to the left and push model predictions away from churn.
- Larger absolute coefficients indicate stronger model associations on the standardized feature scale.

## Conclusion

The visualization helps identify features that the Logistic Regression model uses when predicting customer churn. Tenure and contract type have notable negative coefficients, while fiber-optic internet has a notable positive coefficient.

These relationships are model associations, not proof of causation. Correlated features can also affect coefficient interpretation.

## Output Files

- `reports/feature_importance.png`
- `reports/feature_importance.csv`
- `reports/day12_visualization_findings.md`
