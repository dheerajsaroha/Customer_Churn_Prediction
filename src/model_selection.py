import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from src.model_training import get_base_models, get_regularized_models, RANDOM_STATE
from src.hyperparameter_tuning import get_stratified_kfold


def get_stratified_cv_scores(models, X, y, n_splits=5, scoring="f1",
                             random_state=RANDOM_STATE):
    cv = get_stratified_kfold(n_splits=n_splits, random_state=random_state)
    scoring_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv, scoring=scoring_list, n_jobs=-1, return_train_score=False
        )
        results[name] = {
            "fit_time": scores["fit_time"],
            "test_accuracy": scores["test_accuracy"],
            "test_precision": scores["test_precision"],
            "test_recall": scores["test_recall"],
            "test_f1": scores["test_f1"],
            "test_roc_auc": scores["test_roc_auc"],
        }
    return results


def compare_models(X, y, n_splits=5, random_state=RANDOM_STATE):
    base_models = get_base_models(random_state=random_state)
    regularized_models = get_regularized_models(random_state=random_state)

    all_models = {}
    all_models.update(
        {f"{k} (Base)": v for k, v in base_models.items()}
    )
    all_models.update(
        {f"{k} (Regularized)": v for k, v in regularized_models.items()}
    )

    cv_results = get_stratified_cv_scores(all_models, X, y, n_splits=n_splits,
                                          random_state=random_state)
    return cv_results


def summarize_cv_results(cv_results):
    summary = []
    for name, scores in cv_results.items():
        summary.append({
            "Model": name,
            "Accuracy": np.mean(scores["test_accuracy"]),
            "Std Accuracy": np.std(scores["test_accuracy"]),
            "Precision": np.mean(scores["test_precision"]),
            "Recall": np.mean(scores["test_recall"]),
            "F1": np.mean(scores["test_f1"]),
            "ROC AUC": np.mean(scores["test_roc_auc"]),
        })
    df_summary = pd.DataFrame(summary).sort_values("F1", ascending=False)
    return df_summary


def select_best_model(cv_results, metric="F1"):
    summary = summarize_cv_results(cv_results)
    best_model_name = summary.iloc[0]["Model"]
    return best_model_name, summary
