
# Exploratory Data Analysis — Customer Churn

## Dataset overview

- Total customer records: 7,043
- Customers who stayed: 5,174 (73.46%)
- Customers who left: 1,869 (26.54%)
- Duplicate rows: 0
- Blank TotalCharges values identified: 11

## Key findings

### 1. Overall churn

Approximately 26.54% of customers left the company. The dataset has more customers who stayed than customers who left.

### 2. Churn by contract type

| Contract type | Churn rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Month-to-month customers have the highest churn rate, while two-year customers have the lowest.

### 3. Monthly charges

The box plot suggests that customers who left generally had higher monthly charges than customers who stayed.

## Conclusion

Contract type and monthly charges appear useful for understanding customer churn. These patterns are associations, not proof of causation. Further analysis and machine learning will help evaluate their predictive usefulness.
