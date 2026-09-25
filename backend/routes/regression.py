"""
AI Insight Hub — Linear Regression API Routes (Phase 3)
Exposes the trained Linear Regression model: prediction, model metrics,
held-out evaluation data, and the input schema used by the frontend form.
"""
from flask import request, jsonify

from services.regression_service import (
    MODEL_NAME,
    TRAIN_SCRIPT,
    get_evaluation_data,
    get_feature_schema,
    get_training_metrics,
    is_model_trained,
    predict_price,
    validate_property_input,
)
from . import api_bp


@api_bp.route('/regression/predict', methods=['POST'])
def regression_predict():
    """
    POST /api/regression/predict

    Accepts a property feature payload, validates it, executes the R prediction
    script through the RRunner bridge, and returns the model's prediction.

    Request body:
        {"area": 1500, "bedrooms": 3, "bathrooms": 2,
         "location_score": 8, "property_age": 5}

    Status codes:
        200 — prediction produced
        400 — invalid or missing input
        415 — request was not JSON
        502 — the R engine failed to execute
        503 — the model has not been trained yet
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "model": MODEL_NAME,
            "error": "Request must use Content-Type: application/json."
        }), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "success": False,
            "model": MODEL_NAME,
            "error": "Request body could not be parsed as JSON."
        }), 400

    validated_input, errors = validate_property_input(payload)
    if errors:
        return jsonify({
            "success": False,
            "model": MODEL_NAME,
            "error": " ".join(errors),
            "errors": errors
        }), 400

    result = predict_price(validated_input)
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/regression/metrics', methods=['GET'])
def regression_metrics():
    """
    GET /api/regression/metrics

    Returns the real R², RMSE and MAE produced by the R training script, read
    from the metrics artifact that train.R writes. Nothing is hard-coded.
    """
    result = get_training_metrics()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/regression/evaluation', methods=['GET'])
def regression_evaluation():
    """
    GET /api/regression/evaluation

    Returns the held-out test-set actual vs predicted price pairs, used by the
    frontend scatter chart to visualise model fit.
    """
    result = get_evaluation_data()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/regression/schema', methods=['GET'])
def regression_schema():
    """
    GET /api/regression/schema

    Returns each input feature with its label and the valid range recorded at
    training time, so the frontend form can constrain inputs to real data.
    """
    schema = get_feature_schema()
    return jsonify({
        "success": True,
        "model": MODEL_NAME,
        "target": "price",
        "unit": "INR",
        "model_trained": is_model_trained(),
        "train_script": TRAIN_SCRIPT,
        **schema
    }), 200
