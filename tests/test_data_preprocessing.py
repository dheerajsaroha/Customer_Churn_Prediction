import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    load_raw_data,
    preprocess_data,
    clean_data,
    encode_target,
    encode_features,
    apply_oversampling,
    apply_downsampling,
    get_class_distribution,
)


class TestDataPreprocessing:

    def test_load_raw_data(self):
        df = load_raw_data()
        assert df is not None
        assert "Churn" in df.columns
        assert "customerID" in df.columns

    def test_clean_data_drops_customer_id(self):
        df = load_raw_data()
        df = clean_data(df)
        assert "customerID" not in df.columns

    def test_clean_data_total_charges_numeric(self):
        df = load_raw_data()
        df = clean_data(df)
        assert df["TotalCharges"].dtype in [np.float64, np.float32, float]

    def test_encode_target(self):
        df = clean_data(load_raw_data())
        df = encode_target(df)
        assert set(df["Churn"].unique()) == {0, 1}

    def test_encode_features(self):
        df = clean_data(load_raw_data())
        df = encode_target(df)
        df, encoders = encode_features(df)
        assert len(encoders) > 0
        object_cols = df.select_dtypes(include=["object"]).columns
        assert len(object_cols) == 0
        assert "Churn" not in encoders

    def test_preprocess_data(self):
        result = preprocess_data()
        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert "feature_names" in result
        assert "encoders" in result
        assert "scaler" in result
        assert len(result["feature_names"]) == 19
        assert len(result["X_train"]) + len(result["X_test"]) == 7043 or \
               len(result["X_train"]) + len(result["X_test"]) == 5634 + 1409

    def test_preprocess_data_shapes(self):
        result = preprocess_data()
        assert result["X_train"].shape[1] == 19
        assert result["X_test"].shape[1] == 19

    def test_stratify_in_split(self):
        result = preprocess_data()
        y_train = result["y_train"]
        y_test = result["y_test"]
        train_pos = (y_train == 1).sum() / len(y_train)
        test_pos = (y_test == 1).sum() / len(y_test)
        assert abs(train_pos - test_pos) < 0.05

    def test_apply_oversampling(self):
        result = preprocess_data()
        X_train = result["X_train_scaled"].values
        y_train = result["y_train"].values
        X_res, y_res = apply_oversampling(X_train, y_train)
        counts = get_class_distribution(y_res)
        assert len(set(counts.values())) == 1

    def test_apply_downsampling(self):
        result = preprocess_data()
        X_train = result["X_train_scaled"].values
        y_train = result["y_train"].values
        X_res, y_res = apply_downsampling(X_train, y_train)
        counts = get_class_distribution(y_res)
        assert len(set(counts.values())) == 1
