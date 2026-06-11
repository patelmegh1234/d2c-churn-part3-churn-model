# Part 3: Churn Prediction Model & Model Card

This repository contains Part 3 of the **D2C Customer Churn Intelligence & Retention API** Capstone Project.

The goal of this part is to train, evaluate, interpret, and document a machine learning model that predicts whether a customer will churn within the next 60 days.

---

## Repository Structure

```
d2c-churn-part3-churn-model/
├── churn_model.ipynb      # Full modeling notebook (training → evaluation → saving)
├── model.pkl              # Saved Scikit-Learn Pipeline (RandomForestClassifier)
├── metrics.json           # Final test-set evaluation metrics
├── error_analysis.md      # FP/FN analysis with 10 specific customer examples
├── model_card.md          # Structured model card (intended use, limitations, risks)
├── run_model_code.py      # Standalone script to train model and save all outputs
├── outputs/
│   └── charts/
│       └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## Datasets Used

Place the downloaded dataset files in `../dataset/` relative to this repository.

| File | Used For |
|---|---|
| `rfm_modeling_snapshot.csv` | Primary modeling dataset (pre-computed features + labels) |

The modeling snapshot already provides clean, snapshot-safe features. Post-snapshot data is **not** present in this file.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model and Save Outputs
```bash
python run_model_code.py
```

This will produce:
- `model.pkl` — trained pipeline
- `metrics.json` — test-set metrics
- `outputs/charts/feature_importance.png`

### 3. Open the Notebook
```bash
jupyter notebook churn_model.ipynb
```

---

## Load the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load("model.pkl")

# Example: predict on a new row
sample = pd.DataFrame([{
    "city_tier": "Tier 1", "age_group": "25-34", "acquisition_channel": "Organic",
    "loyalty_tier": "Silver", "preferred_category": "Skin Care", "marketing_consent": 1,
    "recency_days": 45.0, "frequency_180d": 3.0, "monetary_180d": 1500.0,
    "return_rate_180d": 0.1, "avg_discount_pct_180d": 5.0, "avg_rating_180d": 3.8,
    "category_diversity_180d": 2.0, "ticket_count_90d": 1.0, "negative_ticket_rate_90d": 0.5,
    "avg_resolution_hours_90d": 24.0, "days_since_signup": 365.0, "sessions_30d": 5.0,
    "product_views_30d": 20.0, "cart_adds_30d": 2.0, "wishlist_adds_30d": 1.0,
    "abandoned_carts_30d": 1.0, "email_opens_30d": 2.0, "campaign_clicks_30d": 0.0,
    "last_visit_days_ago": 15.0
}])

prob = model.predict_proba(sample)[0][1]
print(f"Churn probability: {prob:.2%}")
```

---

## Model Performance (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 82.4% |
| Precision | 81.1% |
| Recall | 84.5% |
| F1-Score | 82.8% |
| ROC-AUC | 88.7% |
| PR-AUC | 87.7% |
| Decision Threshold | 0.45 |
| True Positives | 142 |
| False Positives | 33 |
| True Negatives | 135 |
| False Negatives | 26 |

> Threshold tuned to 0.45 to maximize Recall — missing a true churner (false negative) costs more than incorrectly flagging a retained customer (false positive) in a retention context.

---

## Top Churn Drivers (Feature Importance)

1. `recency_days` — Most predictive: the longer since last order, the higher the churn risk
2. `last_visit_days_ago` — Recent web/app disengagement strongly signals churn
3. `frequency_180d` — Habitual buyers churn significantly less
4. `ticket_count_90d` — Elevated support interactions signal friction
5. `monetary_180d` — Low recent spend correlates with weakening brand attachment

See `outputs/charts/feature_importance.png` for the full top-15 chart.

---

## Data Split

Used the provided `split` column from `rfm_modeling_snapshot.csv`:
- **Train**: `split == "train"`
- **Validation**: `split == "validation"` (used for threshold tuning)
- **Test**: `split == "test"` (used for final metrics in `metrics.json`)

---

## Leakage Prevention

- Used **only** the `rfm_modeling_snapshot.csv` which is pre-computed on data available ≤ `2025-09-30`
- The `churn_next_60d` column was used **only** as the target label, never as a feature
- The `customer_id`, `snapshot_date`, and `split` columns were dropped before training
