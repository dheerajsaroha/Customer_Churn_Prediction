import pickle
import os
import json
import warnings
import pandas as pd
import numpy as np

from src.data_preprocessing import load_raw_data, preprocess_data, MODEL_DIR


def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model, filepath):
    ensure_model_dir()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_artifacts(artifacts_dict, base_dir=None):
    if base_dir is None:
        base_dir = MODEL_DIR
    ensure_model_dir()

    model_path = None
    encoders_path = os.path.join(base_dir, "encoders.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    results_path = os.path.join(base_dir, "evaluation_results.json")

    if "model" in artifacts_dict:
        model_path = os.path.join(base_dir, "customer_churn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(artifacts_dict["model"], f)

    if "encoders" in artifacts_dict:
        with open(encoders_path, "wb") as f:
            pickle.dump(artifacts_dict["encoders"], f)

    if "scaler" in artifacts_dict:
        with open(scaler_path, "wb") as f:
            pickle.dump(artifacts_dict["scaler"], f)

    if "evaluation_results" in artifacts_dict:
        results = artifacts_dict["evaluation_results"]
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (np.integer,)):
                serializable[k] = int(v)
            elif isinstance(v, (np.floating,)):
                serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, pd.DataFrame):
                serializable[k] = v.to_dict()
            else:
                serializable[k] = v
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    return {
        "model_path": model_path,
        "encoders_path": encoders_path,
        "scaler_path": scaler_path,
        "results_path": results_path,
    }


def load_artifacts(base_dir=None):
    if base_dir is None:
        base_dir = MODEL_DIR

    artifacts = {}

    model_path = os.path.join(base_dir, "customer_churn_model.pkl")
    if os.path.exists(model_path):
        artifacts["model"] = load_model(model_path)

    encoders_path = os.path.join(base_dir, "encoders.pkl")
    if os.path.exists(encoders_path):
        artifacts["encoders"] = load_model(encoders_path)

    scaler_path = os.path.join(base_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        artifacts["scaler"] = load_model(scaler_path)

    results_path = os.path.join(base_dir, "evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            artifacts["evaluation_results"] = json.load(f)

    return artifacts


def get_feature_names():
    processed = preprocess_data()
    return processed["feature_names"]


def predict_single(input_dict, model, encoders, scaler, feature_names):
    input_df = pd.DataFrame([input_dict])

    for col in encoders:
        if col in input_df.columns:
            le = encoders[col]
            input_df[col] = input_df[col].map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            input_df[col] = le.transform(input_df[col])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        prediction = model.predict(input_scaled)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn": int(prediction),
        "probability": float(probability) if probability is not None else None,
    }
