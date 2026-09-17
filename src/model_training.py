import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42


def get_base_models(random_state=RANDOM_STATE):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            random_state=random_state, eval_metric="logloss", use_label_encoder=False
        ),
    }
    return models


def get_regularized_models(random_state=RANDOM_STATE):
    models = {
        "Logistic Regression (L1)": LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=1000, random_state=random_state
        ),
        "Logistic Regression (L2)": LogisticRegression(
            penalty="l2", max_iter=1000, random_state=random_state
        ),
        "Decision Tree (pruned)": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, min_samples_leaf=5,
            random_state=random_state,
        ),
        "Random Forest (regularized)": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt",
            random_state=random_state,
        ),
        "XGBoost (regularized)": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
            reg_lambda=1.0, random_state=random_state,
            eval_metric="logloss", use_label_encoder=False,
        ),
    }
    return models


def _get_param_grids():
    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 5, 10, 20],
            "max_features": ["sqrt", "gini", None],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
    }
    return param_grids


def get_param_grids_for_random_search():
    param_distributions = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100, 1000],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear", "saga"],
            "max_iter": [1000, 2000],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20, 50],
            "min_samples_leaf": [1, 5, 10, 20, 50],
            "max_features": ["sqrt", "gini", "entropy", None],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 5, 10, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, None],
            "bootstrap": [True, False],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [2, 3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 4, 5, 6, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.3, 0.5, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
            "min_child_weight": [1, 3, 5, 10],
        },
    }
    return param_distributions
