# Part 3: Churn Prediction Model & Model Card

This repository contains the third part of the D2C Customer Churn Intelligence & Retention API Capstone Project.

## Contents
- `churn_model.ipynb`: Jupyter notebook containing model training, evaluation, threshold selection, and feature importance.
- `model.pkl`: The trained, final Scikit-Learn Pipeline (RandomForestClassifier).
- `metrics.json`: Final model evaluation metrics based on the chosen threshold.
- `error_analysis.md`: Detailed analysis of false positives and false negatives, including 10 specific customer examples.
- `model_card.md`: A structured model card detailing intended use, limitations, ethical risks, and performance.
- `requirements.txt`: Python dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset files in the `../dataset` relative path. (Uses `rfm_modeling_snapshot.csv`).
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook churn_model.ipynb
   ```
