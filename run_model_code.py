import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

BASE_DIR = "c:/Users/Megh/Desktop/project"
RAW_DIR = os.path.join(BASE_DIR, "dataset")
PART3_DIR = os.path.join(BASE_DIR, "part3_modeling")
CHART_DIR = os.path.join(PART3_DIR, "outputs", "charts")

os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RAW_DIR, "rfm_modeling_snapshot.csv"))

target_col = "churn_next_60d"

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validation']
test_df = df[df['split'] == 'test']

drop_cols = [target_col, "customer_id", "snapshot_date", "split"]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target_col]

X_val = val_df.drop(columns=drop_cols)
y_val = val_df[target_col]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target_col]

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

strong_model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ))
])

strong_model.fit(X_train, y_train)

def evaluate_model(model, X_data, y_true, threshold=0.5):
    probs = model.predict_proba(X_data)[:, 1]
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold
    }

selected_threshold = 0.45
final_metrics = evaluate_model(strong_model, X_test, y_test, threshold=selected_threshold)

with open(os.path.join(PART3_DIR, "metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

joblib.dump(strong_model, os.path.join(PART3_DIR, "model.pkl"))

feature_names = num_cols + list(strong_model.named_steps['prep'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(cat_cols))
importances = strong_model.named_steps['clf'].feature_importances_

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'feature_importance.png'))
plt.close()

print("Part 3 ML training completed and outputs saved.")
