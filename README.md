# ML-Recidivism-Model
Recidivism Prediction using COMPAS dataset - ML model with 74% ROC-AUC for predicting 2-year recidivism risk.
# ML-Recidivism-Model

## Overview

This project develops a machine learning model to predict the risk of recidivism within two years using the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset. The final model achieves a ROC-AUC of 0.7437 and includes a survival analysis to identify critical risk windows.

## Model Performance

The final model is a **Logistic Regression** classifier, trained with **SMOTE** to handle class imbalance. This model was selected for its superior performance compared to other tested algorithms.

| Model | ROC-AUC | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.7437** | **0.6909** | **0.6201** |
| Gradient Boosting | 0.7401 | 0.6833 | 0.6182 |
| XGBoost | 0.7148 | 0.6632 | 0.5936 |
| Random Forest | 0.6909 | 0.6480 | 0.6006 |

## Survival Analysis

In addition to the classification model, a survival analysis was performed to understand the dynamics of recidivism over time.

- **Critical Risk Window:** `6-12 months` (68.5% of recidivism cases occur during this period).
- **Mean Time to Recidivism:** `245 days`.
- **Median Time to Recidivism:** `172 days`.

## Feature Engineering

The following features were engineered to improve model performance:

- `priors_per_age`: Ratio of prior offenses to age.
- `total_risk_score`: Sum of COMPAS risk scores.
- `high_risk_flag`: Binary indicator for a risk score >= 7.
- `total_juv_count`: Total number of juvenile offenses.
- `has_juv_record`: Binary indicator for having a juvenile record.
- `priors_high_risk`: Interaction term between priors count and high risk flag.

## Project Structure
