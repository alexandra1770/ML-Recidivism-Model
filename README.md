# ML-Recidivism-Model

## Overview

This project develops a machine learning model to predict the risk of recidivism within two years using the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset. The final model achieves a ROC-AUC of 0.7437 and includes a survival analysis to identify critical risk windows.

## Model Performance

The final model is a **Logistic Regression** classifier, trained with **SMOTE** to handle class imbalance. This model was selected for its superior performance compared to other tested algorithms.

| Model | ROC-AUC | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | **0.7437** | **0.6909** | **0.6201** |
| Gradient Boosting | 0.7401 | 0.6833 | 0.6182 |
| XGBoost | 0.7148 | 0.6632 | 0.5936 |
| Random Forest | 0.6909 | 0.6480 | 0.6006 |

## Survival Analysis

In addition to the classification model, a survival analysis was performed to understand the dynamics of recidivism over time.

- **Critical Risk Window:** 6-12 months (68.5% of recidivism cases occur during this period)
- **Mean Time to Recidivism:** 245 days
- **Median Time to Recidivism:** 172 days

## Feature Engineering

The following features were engineered to improve model performance:

- priors_per_age: Ratio of prior offenses to age
- total_risk_score: Sum of COMPAS risk scores
- high_risk_flag: Binary indicator for a risk score >= 7
- total_juv_count: Total number of juvenile offenses
- has_juv_record: Binary indicator for having a juvenile record
- priors_high_risk: Interaction term between priors count and high risk flag

## Project Structure

ML-Recidivism-Model/
│
├── ML RECIDIVISM MODEL (1).ipynb   # Main analysis and training notebook
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
│
├── models/                          # Trained models directory
│   ├── best_model_optimized.pkl    # Final optimized pipeline (SMOTE + Logistic Regression)
│   ├── predictor.pkl               # Predictor object for inference
│   ├── config.json                 # Model and feature configuration
│   └── model_results.csv           # Comparison results of all models
│
└── data/                            # Data directory
    ├── compas-scores-raw.csv
    └── compas-scores-two-years.csv

## Installation

Option 1: Using pip

pip install -r requirements.txt

Option 2: Using Anaconda (recommended)

This project was developed using a Conda virtual environment with Python 3.11.

# Create and activate conda environment
conda create -n recidivism python=3.11
conda activate recidivism

# Install dependencies
pip install -r requirements.txt

## How to Use

To make predictions using the trained model:

import joblib
import pandas as pd

# Load the predictor
predictor = joblib.load('models/predictor.pkl')

# Create a new case
new_case = pd.DataFrame({
    'age': [25],
    'priors_count': [3],
    'priors_per_age': [0.12],
    'total_risk_score': [8],
    'total_juv_count': [1],
    'priors_high_risk': [3],
    'high_risk_flag': [1],
    'has_juv_record': [1]
})

# Make a prediction
result = predictor.predict(new_case)
print(f"Recidivism Probability: {result['probabilities'][0]:.2%}")
print(f"Risk Level: {result['risk_levels'][0]}")

## SHAP Analysis

SHAP analysis indicates the most important features for the model are:

1. priors_per_age: Number of prior offenses relative to age
2. total_risk_score: Total COMPAS risk score
3. age: Individual's age
4. priors_high_risk: Interaction between prior offenses and high risk
5. priors_count: Total number of prior offenses

## Technologies Used

- Python 3.11 - Primary programming language
- Pandas, NumPy - Data manipulation and analysis
- Scikit-learn - Preprocessing, model training, and evaluation
- XGBoost - Gradient boosting model
- Imbalanced-learn (SMOTE) - Class imbalance handling
- Lifelines - Survival analysis (Kaplan-Meier)
- SHAP - Model interpretability
- Matplotlib, Seaborn - Data visualization
- Joblib - Model serialization

## Data Source

The COMPAS dataset used in this project is available from ProPublica's GitHub repository: https://github.com/propublica/compas-analysis

## Conclusion

This project demonstrates the development of a functional recidivism risk assessment tool. The logistic regression model, enhanced through feature engineering and data balancing techniques, provides robust performance with a ROC-AUC of 0.7437. The survival analysis identifies the critical 6-12 month post-release window, suggesting a key period for targeted interventions to reduce recidivism.
