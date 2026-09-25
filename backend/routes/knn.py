"""
AI Insight Hub — KNN API Routes (Phase 5)
Exposes the trained K-Nearest Neighbors classifier: student performance
prediction, model metrics, and the configuration the frontend form needs.
"""
from flask import request, jsonify

from services.knn_service import (
    MODEL_NAME,
    TRAIN_SCRIPT,
    get_feature_schema,
    get_model_config,
    get_training_metrics,
    is_model_trained,
    predict_performance,
    validate_student_input,
)
from . import api_bp


@api_bp.route('/knn/predict', methods=['POST'])
def knn_predict():
    """
    POST /api/knn/predict

    Accepts a student feature payload, validates it, executes the R prediction
    script through the RRunner bridge, and returns the predicted performance
    tier along with the K nearest training observations behind the answer.

    Request body:
        {"study_hours": 6.5, "attendance": 88, "previous_score": 74,
         "assignments_completed": 9, "practical_score": 81}

    Status codes:
        200 — prediction produced
        400 — invalid or missing input
        415 — request was not JSON
        502 — the R engine failed to execute or returned an unusable result
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

    validated_input, errors = validate_student_input(payload)
    if errors:
        return jsonify({
            "success": False,
            "model": MODEL_NAME,
            "error": " ".join(errors),
            "errors": errors
        }), 400

    result = predict_performance(validated_input)
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/knn/metrics', methods=['GET'])
def knn_metrics():
    """
    GET /api/knn/metrics

    Returns the real accuracy, precision, recall, F1 score, confusion matrix,
    per-class metrics and K-comparison produced by the R training script, read
    from the metrics artifact that train.R writes. Nothing is hard-coded.
    """
    result = get_training_metrics()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/knn/config', methods=['GET'])
def knn_config():
    """
    GET /api/knn/config

    Returns the selected K, the evaluated K grid, feature names with their
    training ranges, the class set and the scaling method — everything the
    frontend needs to build the form and explain the result. Internal details
    such as the stored training matrix are deliberately not exposed.
    """
    result = get_model_config()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/knn/schema', methods=['GET'])
def knn_schema():
    """
    GET /api/knn/schema

    Returns each input feature with its label and the valid range recorded at
    training time, plus the model's class set, so the frontend form can
    constrain inputs to real data.
    """
    schema = get_feature_schema()
    return jsonify({
        "success": True,
        "model": MODEL_NAME,
        "target": "performance",
        "model_trained": is_model_trained(),
        "train_script": TRAIN_SCRIPT,
        **schema
    }), 200
