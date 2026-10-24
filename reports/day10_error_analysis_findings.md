# Day 10 — Prediction Error Analysis

## 1. Objective

Analyze false positives and false negatives in the Logistic Regression churn model at classification thresholds of 0.3 and 0.5.

## 2. Experimental Setup

- Dataset: Telco Customer Churn
- Test records: 1,409
- Model: Logistic Regression with StandardScaler
- Train-test split: 80% / 20%
- Random state: 42
- Thresholds compared: 0.3 and 0.5

## 3. Results

| Measurement | Threshold 0.3 | Threshold 0.5 |
|---|---:|---:|
| True Negatives | 774 | 925 |
| False Positives | 261 | 110 |
| False Negatives | 92 | 162 |
| True Positives | 282 | 212 |
| Correct Predictions | 1,056 | 1,137 |
| Incorrect Predictions | 353 | 272 |

## 4. Interpretation

### Threshold 0.3

The model identified 282 actual churners and missed 92. However, it incorrectly flagged 261 customers who did not churn.

### Threshold 0.5

The model identified 212 actual churners and missed 162. It produced fewer false-positive alerts, with 110 customers incorrectly flagged.

### Comparison

Lowering the threshold from 0.5 to 0.3 detected 70 additional churners and reduced missed churners by 70. It also produced 151 additional false positives.

## 5. Business Implications

A lower threshold may be useful when missing a customer who is about to churn is costly. However, additional false positives may increase the cost of unnecessary retention campaigns.

A higher threshold reduces false alarms but may allow more actual churners to go undetected.

## 6. Conclusion

Threshold selection involves balancing customer retention opportunities against the cost of unnecessary outreach. Threshold 0.3 detected more churners in this experiment, while threshold 0.5 produced more correct predictions overall.

Neither threshold should be selected for deployment based on these counts alone. The business cost of false positives and false negatives, along with validation on separate data, should guide the final decision.

## 7. Next Steps

- Compare model errors using additional metrics.
- Consider the cost of missed churners and unnecessary retention offers.
- Validate the chosen threshold before deployment.