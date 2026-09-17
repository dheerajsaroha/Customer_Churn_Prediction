import os
import json
from flask import Blueprint, render_template, request, jsonify
from src.data_preprocessing import (
    preprocess_data, get_class_distribution,
)
from src.utils import load_model, load_artifacts, get_feature_names, predict_single

main_bp = Blueprint("main", __name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

FEATURE_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "tenure": "number",
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
    "MonthlyCharges": "number",
    "TotalCharges": "number",
}


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/predict")
def predict_page():
    feature_names = get_feature_names()
    return render_template("predict.html", feature_options=FEATURE_OPTIONS,
                           feature_names=feature_names)


@main_bp.route("/predict", methods=["POST"])
def predict():
    try:
        artifacts = load_artifacts(MODEL_DIR)
        if "model" not in artifacts:
            return jsonify({"error": "Model not found. Please train the model first."}), 404

        model = artifacts["model"]
        encoders = artifacts.get("encoders", {})
        scaler = artifacts.get("scaler")
        feature_names = artifacts.get("feature_names", get_feature_names())

        input_data = {}
        for key, value in request.form.items():
            if key == "submit":
                continue
            if key in FEATURE_OPTIONS and FEATURE_OPTIONS[key] == "number":
                input_data[key] = float(value) if value else 0.0
            else:
                input_data[key] = value

        for col, le in encoders.items():
            if col in input_data and isinstance(input_data[col], str):
                input_data[col] = input_data[col]

        result = predict_single(input_data, model, encoders, scaler, feature_names)

        prediction_text = "Churn" if result["churn"] == 1 else "No Churn"
        confidence = result["probability"]
        if confidence is not None:
            if result["churn"] == 1:
                confidence = round(confidence * 100, 2)
            else:
                confidence = round((1 - confidence) * 100, 2)

        return render_template(
            "result.html",
            prediction=prediction_text,
            probability=confidence,
            input_data=input_data,
            feature_names=feature_names,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        artifacts = load_artifacts(MODEL_DIR)
        if "model" not in artifacts:
            return jsonify({"error": "Model not found"}), 404

        model = artifacts["model"]
        encoders = artifacts.get("encoders", {})
        scaler = artifacts.get("scaler")
        feature_names = artifacts.get("feature_names", get_feature_names())

        input_data = request.get_json(silent=True) or dict(request.form)
        input_data = {k: v for k, v in input_data.items() if k != "submit"}

        for key in list(input_data.keys()):
            if FEATURE_OPTIONS.get(key) == "number":
                try:
                    input_data[key] = float(input_data[key])
                except (ValueError, TypeError):
                    input_data[key] = 0.0

        result = predict_single(input_data, model, encoders, scaler, feature_names)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/evaluation")
def evaluation():
    results_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = None

    chart_path = os.path.join(MODEL_DIR, "feature_importance.png")
    has_chart = os.path.exists(chart_path)

    processed = preprocess_data()
    class_dist = get_class_distribution(processed["y_train"])
    test_class_dist = get_class_distribution(processed["y_test"])

    return render_template(
        "evaluation.html",
        results=results,
        has_chart=has_chart,
        class_distribution=class_dist,
        test_class_distribution=test_class_dist,
        feature_names=processed["feature_names"],
    )


@main_bp.route("/health")
def health():
    return jsonify({"status": "ok"})


@main_bp.route("/static/feature_importance.png")
def feature_importance_image():
    img_path = os.path.join(MODEL_DIR, "feature_importance.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "static", "img", "feature_importance.png"
        )
    if os.path.exists(img_path):
        from flask import send_file
        return send_file(img_path, mimetype="image/png")
    return "", 404
