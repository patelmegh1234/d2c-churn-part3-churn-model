# Model Card: D2C Customer Churn Prediction Model

## Intended Use
Predict whether a customer is at risk of churn (no purchase) within the next 60 days to support targeted retention prioritization.

## Users
Internal CRM, retention, product, and lifecycle marketing teams.

## Data Used
Snapshot-based customer features generated from data available on or before the snapshot date (`2025-09-30`). The data includes static demographics, 180-day RFM aggregates, 90-day support aggregates, and 30-day web activity. Post-snapshot data was strictly excluded to prevent leakage.

## Target
Binary churn label for the next 60 days. `1` = Churn, `0` = Retained.

## Modeling Approach
Compared a baseline Logistic Regression model with a stronger tree-based classifier (Random Forest). Selected Random Forest as the final model due to superior F1 and ROC-AUC scores. Features were preprocessed using median imputation for numerics and one-hot encoding for categoricals.

## Performance
- **Accuracy**: ~84%
- **Precision**: ~82%
- **Recall**: ~70%
- **F1 Score**: ~75%
- **ROC-AUC**: ~88%
*(See metrics.json for exact numbers based on the test set).*

## Threshold Decision
The default threshold is 0.5. However, we tuned the threshold down to **0.45** to increase Recall. For retention, it is generally less costly to send an email or small discount to a false positive than it is to completely lose a high-value false negative.

## Main Drivers (Feature Importance)
1. **recency_days**: The strongest predictor. The longer since the last order, the higher the churn probability.
2. **last_visit_days_ago**: Recent web/app engagement is highly correlated with retention.
3. **frequency_180d**: Habitual buyers are much less likely to churn.
4. **ticket_count_90d**: Elevated support interactions signal friction.

## Limitations
- Prediction is probabilistic, not certain.
- Performance may degrade if customer behavior shifts (e.g., due to macroeconomics or new product lines).
- Sensitive to data-quality issues (e.g., broken web tracking will spike `last_visit_days_ago`).

## Ethical Risks
- Wrongly labeling loyal customers may waste offers.
- Missing true churners may harm customer experience.
- The model should not be used as the only basis for customer treatment (e.g., do not deny service based on a high churn score).

## Monitoring Needs
Track data drift on top features, prediction distribution (percentage of users flagged as high-risk), calibration, intervention outcomes, and retraining triggers.

## When NOT to use
Do not use when input data is incomplete, stale (e.g., pipeline is >3 days delayed), or if post-snapshot logic is somehow violated in production.
