# Customer Churn Prediction

A complete machine learning pipeline and web application for predicting customer churn using the **Telco Customer Churn dataset**. The project covers data preprocessing, hyperparameter tuning, model selection, class imbalance handling, overfitting mitigation, and stratified cross-validation — all wrapped in a Flask web UI.

---

## Table of Contents

1. [Overview](#overview)
2. [ML Techniques](#ml-techniques)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)

---

## Overview

Customer churn prediction identifies subscribers likely to cancel their service contracts. This project builds a binary classification model on the Telco Customer Churn dataset, handling:

- **Imbalanced classes** (≈27% churn rate) via SMOTE oversampling and random downsampling
- **Categorical encoding** for all 19 features
- **Feature scaling** with StandardScaler
- **5 key ML techniques**: hyperparameter tuning, model selection, downsampling, overfitting mitigation, and stratified k-fold cross-validation

---

## ML Techniques

### 1. Hyperparameter Tuning

`src/hyperparameter_tuning.py` provides `tune_hyperparameters()` supporting two search strategies:

| Strategy | Method | Use Case |
|----------|--------|----------|
| Grid Search | `GridSearchCV` | Exhaustive search over a defined parameter grid |
| Random Search | `RandomizedSearchCV` | Sampling a fixed number of parameter combinations |

Both use **StratifiedKFold** cross-validation (default 5-fold) and optimize for the **F1 macro score** — the most appropriate metric for imbalanced churn prediction.

Per-model parameter grids are defined in `src/model_training.py`:

```python
# Example: Random Forest grid
{
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2", None],
}
```

### 2. Model Selection

`src/model_selection.py` compares **10 models** (5 base + 5 regularized) using stratified k-fold cross-validation:

| Model | Base | Regularized Variant |
|-------|------|--------------------|
| Logistic Regression | ✓ | L1 & L2 regularization |
| Decision Tree | ✓ | Pruned (max_depth, min_samples) |
| Random Forest | ✓ | max_depth, min_samples, max_features |
| Gradient Boosting | ✓ | — |
| XGBoost | ✓ | reg_alpha, reg_lambda, subsampling |

Models are ranked by F1, Accuracy, Precision, Recall, and ROC AUC.

### 3. Downsampling

`src/data_preprocessing.py` implements `apply_downsampling()` using `RandomUnderSampler` from imbalanced-learn. The pipeline compares three sampling strategies:

| Strategy | Class 0 | Class 1 | Best XGBoost F1 |
|----------|---------|---------|-----------------|
| No Sampling | 4,139 | 1,495 | 0.553 |
| SMOTE Oversampling | 4,139 | 4,139 | **0.850** |
| Random Downsampling | 1,495 | 1,495 | 0.735 |

**Result**: SMOTE oversampling significantly outperforms downsampling, which loses too much information from the majority class.

### 4. Overfitting Mitigation

`src/model_training.py` provides `get_regularized_models()` with techniques tailored to each algorithm:

- **Logistic Regression**: L1 (Lasso) and L2 (Ridge) penalties with `liblinear` solver
- **Decision Tree**: Depth limiting (`max_depth=5`), `min_samples_split=10`, `min_samples_leaf=5`
- **Random Forest**: `max_features="sqrt"`, depth constraints, `min_samples_leaf`
- **XGBoost**: `reg_alpha` (L1), `reg_lambda` (L2), `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight`

The pipeline also includes an **overfitting diagnosis** step in `train_pipeline.py` that computes the train-test accuracy gap:

```python
train_acc = 0.9770  # RF on SMOTE data
test_acc  = 0.7743
gap       = 0.2027  # > 0.05 → overfitting warning
```

### 5. Stratified K-Fold Cross-Validation

`src/hyperparameter_tuning.py` provides `get_stratified_kfold()`:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

This is used in:
- `tune_hyperparameters()` — for grid/random search CV
- `compare_models()` — for model selection CV
- `train_pipeline.py` — for all evaluation steps

Stratification preserves the class ratio in each fold, critical for the imbalanced churn dataset.

---

## Project Structure

```
Customer_Churn_Prediction/
├── Dataset/                          # Raw dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/                              # Source modules
│   ├── __init__.py                   # Package exports
│   ├── data_preprocessing.py         # Loading, cleaning, encoding, sampling
│   ├── model_training.py             # Base & regularized model definitions + param grids
│   ├── hyperparameter_tuning.py      # GridSearchCV / RandomizedSearchCV
│   ├── model_selection.py            # Model comparison with stratified CV
│   ├── model_evaluation.py           # Metrics, confusion matrix, feature importance plots
│   └── utils.py                      # Save/load artifacts, single-row prediction
├── models/                           # Trained model artifacts
│   ├── customer_churn_model.pkl      # Trained model
│   ├── encoders.pkl                  # Label encoders for categorical features
│   ├── scaler.pkl                    # StandardScaler
│   ├── evaluation_results.json       # Metrics summary
│   └── feature_importance.png        # Feature importance chart
├── app/                              # Flask web application
│   ├── __init__.py                   # App factory
│   ├── routes.py                     # Routes: /, /predict, /evaluation, /api/predict, /health
│   ├── templates/
│   │   ├── base.html                 # Base template with navbar
│   │   ├── index.html                # Home page
│   │   ├── predict.html              # Prediction form
│   │   ├── result.html               # Prediction result display
│   │   └── evaluation.html           # Model evaluation dashboard
│   └── static/
│       ├── css/style.css             # Custom styles
│       └── js/main.js                # Form interaction
├── notebooks/                        # Jupyter notebooks
│   └── model.ipynb                   # Original exploratory notebook
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_data_preprocessing.py    # 8 tests
│   ├── test_models.py                # 14 tests
│   └── test_app.py                   # 4 tests
├── train_pipeline.py                 # End-to-end training script
├── run.py                            # Flask app entry point
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest configuration
├── AGENTS.md                         # Agent command shortcuts
└── README.md
```

---

## Installation

```bash
# Clone and enter directory
git clone <repo-url> Customer_Churn_Prediction
cd Customer_Churn_Prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the Model

Runs the full pipeline: data preprocessing → sampling comparison → model selection → hyperparameter tuning → evaluation → artifact saving.

```bash
python train_pipeline.py
```

### Run the Web App

Starts a local Flask server on port 5000.

```bash
python run.py
```

| Route | Description |
|-------|-------------|
| `GET /` | Home page |
| `GET /predict` | Prediction form |
| `POST /predict` | Form-based prediction |
| `POST /api/predict` | JSON-based prediction API |
| `GET /evaluation` | Model evaluation dashboard |
| `GET /health` | Health check |

### Run Tests

```bash
python -m pytest tests/ -v
```

Expected output: **27 passed**

---

## Results

The final trained model is a **Random Forest** (tuned via RandomizedSearchCV with SMOTE oversampling):

| Metric | Value |
|--------|-------|
| Accuracy | 77.4% |
| Precision | 56.7% |
| Recall | 63.4% |
| F1 Score | 59.9% |
| ROC AUC | 82.7% |

**Top features**: Contract, tenure, MonthlyCharges, TotalCharges, PaymentMethod.

The pipeline also reports overfitting diagnostics to guide selection of regularized model variants when train-test gaps exceed 5%.
