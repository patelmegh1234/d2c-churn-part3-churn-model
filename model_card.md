# Model Card: D2C Customer Churn Prediction Model

## Intended Use
Predict whether a customer is at risk of churn (no purchase) within the **next 60 days**, to support targeted retention prioritization by the CRM, retention, product, and lifecycle marketing teams. This model is **not** intended for credit scoring, fraud detection, or any customer-facing decisions.

---

## Data Used

- **Source**: `rfm_modeling_snapshot.csv` from the D2C dataset (snapshot date: `2025-09-30`)
- **Features**: 25 features grouped into:
  - Static demographics: `city_tier`, `age_group`, `acquisition_channel`, `loyalty_tier`, `preferred_category`, `marketing_consent`
  - RFM (180-day): `recency_days`, `frequency_180d`, `monetary_180d`, `return_rate_180d`, `avg_discount_pct_180d`, `avg_rating_180d`, `category_diversity_180d`
  - Support (90-day): `ticket_count_90d`, `negative_ticket_rate_90d`, `avg_resolution_hours_90d`
  - Account: `days_since_signup`
  - Web/app (30-day): `sessions_30d`, `product_views_30d`, `cart_adds_30d`, `wishlist_adds_30d`, `abandoned_carts_30d`, `email_opens_30d`, `campaign_clicks_30d`, `last_visit_days_ago`
- **Target**: `churn_next_60d` (binary: 1 = churned, 0 = retained)
- **Leakage prevention**: Only data available on or before `2025-09-30` was used. The `churn_next_60d` column was used as target only, never as a feature.

---

## Data Split

Used the provided `split` column from the modeling snapshot:
| Split | Purpose |
|---|---|
| `train` | Model training |
| `validation` | Threshold selection and hyperparameter insight |
| `test` | Final evaluation (reported in metrics.json) |

---

## Model Approach

Two models were trained and compared:

| Model | Notes |
|---|---|
| **Logistic Regression** (baseline) | `class_weight="balanced"`, `max_iter=1000` |
| **Random Forest** (final) | 300 trees, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42` |

Random Forest was selected as the final model based on superior F1 and ROC-AUC on the validation set. Preprocessing used **median imputation** for numerics and **one-hot encoding** for categoricals, all packaged in a Scikit-Learn `Pipeline`.

---

## Performance (Test Set)

All metrics computed at the selected threshold of **0.45**. See `metrics.json` for the exact values.

| Metric | Value |
|---|---|
| Accuracy | **82.4%** |
| Precision | **81.1%** |
| Recall | **84.5%** |
| F1-Score | **82.8%** |
| ROC-AUC | **88.7%** |
| PR-AUC | **87.7%** |
| True Positives | 142 |
| False Positives | 33 |
| True Negatives | 135 |
| False Negatives | 26 |

---

## Threshold Decision

The decision threshold was tuned from the default `0.50` down to **`0.45`** using the validation set.

**Business rationale**: In a retention context, a **False Negative** (missing a true churner) is more costly than a **False Positive** (flagging a loyal customer). Lowering the threshold increases Recall (84.5%), meaning we catch more at-risk customers. The cost of sending an unnecessary discount to a retained customer is far lower than losing a churning customer entirely.

---

## Top Feature Drivers

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `recency_days` | Strongest predictor — the longer since last order, the higher the churn risk |
| 2 | `last_visit_days_ago` | Recent web/app disengagement strongly signals churn |
| 3 | `frequency_180d` | Habitual buyers churn significantly less |
| 4 | `ticket_count_90d` | Elevated support interactions signal friction and frustration |
| 5 | `monetary_180d` | Low recent spend indicates weakening brand attachment |

See `outputs/charts/feature_importance.png` for the full top-15 chart.

---

## Limitations

- Predictions are probabilistic, not certain. Use as one input, not the sole decision-maker.
- Performance may degrade as customer behaviour shifts (new product launches, macroeconomic changes, seasonal patterns).
- Sensitive to data-quality issues (e.g., broken web-tracking inflates `last_visit_days_ago`; missing support-ticket data deflates `ticket_count_90d`).
- The model was trained on data up to `2025-09-30`. Periodic retraining is needed to stay current.

---

## Ethical Risks

- **False Positives**: Loyal customers incorrectly flagged may receive unnecessary discounts, eroding margins.
- **False Negatives**: True churners missed by the model represent lost revenue with no intervention opportunity.
- **Do not use** this model to deny service, downgrade account tiers, or make any negative customer-facing decisions based on a high churn score.
- **Do not share** raw churn probabilities with customers — this could erode trust.
- **Marketing consent**: Never target customers with `marketing_consent = 0` for promotional campaigns, regardless of their score.

---

## Monitoring Needs

After deployment, track the following:
- **Data drift**: Weekly distribution comparison of top features vs. training baseline (KS-test)
- **Prediction drift**: Daily % of customers flagged as high-risk
- **Business outcomes**: Monthly comparison of predicted vs. actual churn rates from 60 days prior
- **Retraining trigger**: If actual churn rate deviates from predicted by >10pp for 2+ consecutive months, or F1 on a fresh holdout drops below 0.70

---

## When NOT to Use This Model

- When input data is stale (pipeline delayed >3 days) or when post-snapshot features are inadvertently included
- For decisions outside its intended scope (fraud, credit, etc.)
- Immediately after a major business event (product recall, pricing change) without revalidating performance
- Without human review for high-stakes interventions (e.g., large discount offers to VIP customers)
