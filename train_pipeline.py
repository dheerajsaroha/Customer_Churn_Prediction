import warnings
import json
import os
import sys

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data_preprocessing import (
    load_raw_data,
    preprocess_data,
    apply_oversampling,
    apply_downsampling,
    get_class_distribution,
)
from src.hyperparameter_tuning import tune_hyperparameters, get_stratified_kfold
from src.model_selection import compare_models, summarize_cv_results
from src.model_training import get_base_models, RANDOM_STATE
from src.model_evaluation import evaluate_model, plot_feature_importance
from src.utils import save_artifacts


def run_model_comparison(X, y, n_splits=5):
    print("=" * 60)
    print("MODEL SELECTION: Stratified K-Fold CV (Base Models)")
    print("=" * 60)

    cv_results = compare_models(X, y, n_splits=n_splits)
    summary = summarize_cv_results(cv_results)
    print("\nCross-Validation Results (mean):")
    print(summary.to_string(index=False))

    best_model_name = summary.iloc[0]["Model"]
    print(f"\nBest model from comparison: {best_model_name}")
    return cv_results, summary


def run_sampling_comparison(X_train, y_train, n_splits=5):
    print("\n" + "=" * 60)
    print("SAMPLING TECHNIQUE COMPARISON (SMOTE vs DOWNSAMPLING)")
    print("=" * 60)

    X_oversampled, y_oversampled = apply_oversampling(X_train, y_train)
    X_downsampled, y_downsampled = apply_downsampling(X_train, y_train)

    print(f"Original train class distribution:     {get_class_distribution(y_train)}")
    print(f"SMOTE oversampled class distribution:  {get_class_distribution(y_oversampled)}")
    print(f"Downsampled class distribution:        {get_class_distribution(y_downsampled)}")

    strategies = {
        "No Sampling": (X_train, y_train),
        "SMOTE Oversampling": (X_oversampled, y_oversampled),
        "Random Downsampling": (X_downsampled, y_downsampled),
    }

    cv = get_stratified_kfold(n_splits=n_splits)
    base_models = get_base_models(random_state=RANDOM_STATE)
    key_models = ["XGBoost", "Random Forest", "Logistic Regression"]

    results = {}
    for strategy_name, (X_strategy, y_strategy) in strategies.items():
        print(f"\n--- Strategy: {strategy_name} ---")
        strategy_scores = {}
        for model_name in key_models:
            model = base_models[model_name]
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_strategy, y_strategy, cv=cv,
                                     scoring="f1", n_jobs=-1)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            strategy_scores[model_name] = {"mean_f1": mean_score, "std_f1": std_score}
            print(f"  {model_name}: F1 = {mean_score:.4f} +/- {std_score:.4f}")
        results[strategy_name] = strategy_scores

    print("\n--- Summary: XGBoost F1 per strategy ---")
    for strategy_name, scores in results.items():
        xgb_f1 = scores["XGBoost"]["mean_f1"]
        print(f"  {strategy_name}: XGBoost F1 = {xgb_f1:.4f}")

    return results, X_oversampled, y_oversampled, X_downsampled, y_downsampled


def run_hyperparameter_tuning(X_train, y_train, model_name, n_splits=5,
                              method="random", scoring="f1", n_iter=30):
    print(f"\n--- Hyperparameter Tuning: {model_name} ({method}, n_iter={n_iter}) ---")

    tuning_result = tune_hyperparameters(
        X_train, y_train, model_name=model_name, method=method,
        n_splits=n_splits, scoring=scoring, n_iter=n_iter
    )

    result = tuning_result[model_name]
    print(f"  Best F1 Score: {result['best_score']:.4f} +/- {result['std_test_score']:.4f}")
    print(f"  Best Params: {result['best_params']}")

    return result["best_model"], result["best_params"], result


def check_overfitting(model, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSIS")
    print("=" * 60)

    from sklearn.metrics import f1_score
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, test_pred)

    gap = train_acc - test_acc

    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Train F1:       {train_f1:.4f}")
    print(f"  Test F1:        {test_f1:.4f}")
    print(f"  Gap (train - test accuracy): {gap:.4f}")

    if gap > 0.05:
        print("  [WARNING] Potential overfitting detected (gap > 0.05)")
    else:
        print("  [OK] Overfitting unlikely (gap <= 0.05)")

    return {"train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
            "train_f1": float(train_f1), "test_f1": float(test_f1),
            "gap": float(gap)}


def train_final_model(X_train, y_train, X_test, y_test, feature_names,
                      use_downsampling=True, n_splits=5, scoring="f1"):
    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING (Hyperparameter Tuned)")
    print("=" * 60)

    if use_downsampling:
        X_resampled, y_resampled = apply_downsampling(X_train, y_train)
        strategy = "Random Undersampling"
    else:
        X_resampled, y_resampled = apply_oversampling(X_train, y_train)
        strategy = "SMOTE Oversampling"

    print(f"Sampling strategy: {strategy}")
    print(f"Resampled class distribution: {get_class_distribution(y_resampled)}")

    from sklearn.model_selection import cross_val_score
    cv = get_stratified_kfold(n_splits=n_splits)
    base_models = get_base_models(random_state=RANDOM_STATE)

    best_f1 = -1
    best_model_name = None
    for name, model in base_models.items():
        scores = cross_val_score(model, X_resampled, y_resampled, cv=cv,
                                 scoring="f1", n_jobs=-1)
        mean_f1 = np.mean(scores)
        print(f"  {name}: F1 = {mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name

    print(f"\nBest base model: {best_model_name} (F1 = {best_f1:.4f})")
    print(f"\nRunning GridSearchCV hyperparameter tuning on {best_model_name}...")

    best_model, best_params, tuning_result = run_hyperparameter_tuning(
        X_resampled, y_resampled, best_model_name,
        n_splits=n_splits, method="random", scoring=scoring, n_iter=30
    )

    best_model.fit(X_resampled, y_resampled)

    print(f"\nFinal model: {best_model_name}")
    print(f"Best parameters: {best_params}")

    overfitting_check = check_overfitting(best_model, X_resampled, y_resampled,
                                            X_test, y_test)

    eval_results = evaluate_model(best_model, X_test, y_test)
    print("\nFinal Evaluation on Test Set:")
    print(f"  Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall:    {eval_results['recall']:.4f}")
    print(f"  F1 Score:  {eval_results['f1']:.4f}")
    print(f"  ROC AUC:   {eval_results['roc_auc']:.4f}")
    print(f"  Confusion Matrix:\n{eval_results['confusion_matrix']}")
    print(f"\n  Classification Report:\n{eval_results['classification_report']}")

    feature_imp = plot_feature_importance(best_model, feature_names,
                                          save_path="models/feature_importance.png")
    if feature_imp is not None:
        print(f"\nTop 10 features:\n{feature_imp.head(10).to_string(index=False)}")

    return best_model, best_model_name, best_params, eval_results, \
        overfitting_check, strategy


def main():
    print("Loading and preprocessing data...")
    processed = preprocess_data()

    X_train = processed["X_train"]
    X_test = processed["X_test"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]
    X_train_scaled = processed["X_train_scaled"]
    X_test_scaled = processed["X_test_scaled"]
    feature_names = processed["feature_names"]
    encoders = processed["encoders"]
    scaler = processed["scaler"]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {len(feature_names)}")

    run_sampling_comparison(
        X_train_scaled.values, y_train.values, n_splits=5
    )

    run_model_comparison(X_train_scaled.values, y_train.values, n_splits=5)

    best_model, best_model_name, best_params, eval_results, \
        overfitting_check, strategy = train_final_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        feature_names, use_downsampling=False, n_splits=5
    )

    artifacts = {
        "model": best_model,
        "encoders": encoders,
        "scaler": scaler,
        "evaluation_results": {
            "test_accuracy": float(eval_results["accuracy"]),
            "test_precision": float(eval_results["precision"]),
            "test_recall": float(eval_results["recall"]),
            "test_f1": float(eval_results["f1"]),
            "test_roc_auc": float(eval_results["roc_auc"]) if eval_results["roc_auc"] else None,
            "confusion_matrix": eval_results["confusion_matrix"].tolist(),
            "classification_report": eval_results["classification_report"],
            "overfitting_check": overfitting_check,
            "best_model_name": best_model_name,
            "best_params": best_params,
            "sampling_strategy": strategy,
            "feature_names": feature_names,
        },
    }

    saved_paths = save_artifacts(artifacts)
    print(f"\nArtifacts saved:")
    for k, v in saved_paths.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
