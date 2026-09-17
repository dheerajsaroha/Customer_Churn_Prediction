import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

from src.hyperparameter_tuning import tune_hyperparameters, get_stratified_kfold
from src.model_training import get_base_models, get_regularized_models, _get_param_grids, get_param_grids_for_random_search, RANDOM_STATE
from src.model_selection import compare_models, summarize_cv_results, select_best_model
from src.model_evaluation import evaluate_model
from src.utils import save_model, load_model, save_artifacts, load_artifacts, predict_single


class TestHyperparameterTuning:

    def test_get_stratified_kfold(self):
        cv = get_stratified_kfold(n_splits=5)
        assert isinstance(cv, StratifiedKFold)
        assert cv.n_splits == 5

    def test_tune_single_model_grid(self):
        from src.data_preprocessing import preprocess_data
        result = preprocess_data()
        X = result["X_train_scaled"].values
        y = result["y_train"].values

        tuning = tune_hyperparameters(X, y, model_name="Logistic Regression",
                                      method="grid", n_splits=3)
        assert "Logistic Regression" in tuning
        assert "best_model" in tuning["Logistic Regression"]
        assert "best_params" in tuning["Logistic Regression"]
        assert tuning["Logistic Regression"]["best_score"] > 0

    def test_tune_single_model_random(self):
        from src.data_preprocessing import preprocess_data
        result = preprocess_data()
        X = result["X_train_scaled"].values
        y = result["y_train"].values

        tuning = tune_hyperparameters(X, y, model_name="Logistic Regression",
                                      method="random", n_splits=3, n_iter=5)
        assert "Logistic Regression" in tuning
        assert tuning["Logistic Regression"]["best_score"] > 0


class TestModelTraining:

    def test_get_base_models(self):
        models = get_base_models()
        assert "Logistic Regression" in models
        assert "Decision Tree" in models
        assert "Random Forest" in models
        assert "Gradient Boosting" in models
        assert "XGBoost" in models

    def test_get_regularized_models(self):
        models = get_regularized_models()
        assert len(models) == 5
        for name in models:
            assert any(kw in name.lower() for kw in ["regularized", "pruned", "l1", "l2"])

    def test_param_grids_exist(self):
        grids = _get_param_grids()
        assert "Logistic Regression" in grids
        assert "Decision Tree" in grids
        assert "Random Forest" in grids
        assert "Gradient Boosting" in grids
        assert "XGBoost" in grids

    def test_random_search_grids_exist(self):
        grids = get_param_grids_for_random_search()
        assert "Logistic Regression" in grids
        assert "Random Forest" in grids


class TestModelSelection:

    def test_compare_models(self):
        from src.data_preprocessing import preprocess_data
        result = preprocess_data()
        X = result["X_train_scaled"].values
        y = result["y_train"].values

        cv_results = compare_models(X, y, n_splits=3)
        assert len(cv_results) > 0
        for model_name, scores in cv_results.items():
            assert "test_accuracy" in scores
            assert "test_f1" in scores

    def test_summarize_cv_results(self):
        from src.data_preprocessing import preprocess_data
        result = preprocess_data()
        X = result["X_train_scaled"].values
        y = result["y_train"].values

        cv_results = compare_models(X, y, n_splits=3)
        summary = summarize_cv_results(cv_results)
        assert "Model" in summary.columns
        assert "F1" in summary.columns
        assert "Accuracy" in summary.columns

    def test_select_best_model(self):
        from src.data_preprocessing import preprocess_data
        result = preprocess_data()
        X = result["X_train_scaled"].values
        y = result["y_train"].values

        cv_results = compare_models(X, y, n_splits=3)
        best_name, summary = select_best_model(cv_results)
        assert best_name is not None


class TestModelEvaluation:

    def test_evaluate_model(self):
        from src.data_preprocessing import preprocess_data
        from src.model_training import get_base_models
        from src.data_preprocessing import apply_oversampling

        result = preprocess_data()
        X_train = result["X_train_scaled"]
        y_train = result["y_train"]
        X_test = result["X_test_scaled"]
        y_test = result["y_test"]

        X_res, y_res = apply_oversampling(X_train.values, y_train.values)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_res, y_res)

        eval_results = evaluate_model(model, X_test.values, y_test.values)
        assert "accuracy" in eval_results
        assert "precision" in eval_results
        assert "recall" in eval_results
        assert "f1" in eval_results
        assert "confusion_matrix" in eval_results
        assert eval_results["accuracy"] >= 0


class TestUtils:

    def test_save_and_load_model(self, tmp_path):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        filepath = str(tmp_path / "test_model.pkl")
        save_model(model, filepath)
        loaded = load_model(filepath)
        assert loaded is not None
        assert type(loaded) == type(model)

    def test_predict_single(self):
        from src.data_preprocessing import preprocess_data, apply_oversampling
        from src.model_training import get_base_models
        from src.utils import load_model

        result = preprocess_data()
        X_train = result["X_train_scaled"]
        y_train = result["y_train"]

        X_res, y_res = apply_oversampling(X_train.values, y_train.values)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_res, y_res)
        scaler = result["scaler"]
        encoders = result["encoders"]
        feature_names = result["feature_names"]

        input_sample = {name: 0 for name in feature_names}
        pred = predict_single(input_sample, model, encoders, scaler, feature_names)
        assert "churn" in pred
        assert "probability" in pred
        assert pred["churn"] in [0, 1]
