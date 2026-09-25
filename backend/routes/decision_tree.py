"""
AI Insight Hub — Decision Tree API Routes (Phase 4)
Exposes the trained Decision Tree classifier: financial-risk prediction, model
metrics, the fitted tree structure, and the input schema used by the form.
"""
from flask import request, jsonify

from services.decision_tree_service import (
    MODEL_NAME,
    TRAIN_SCRIPT,
    get_feature_schema,
    get_training_metrics,
    get_tree_structure,
    is_model_trained,
    predict_risk,
    validate_applicant_input,
)
from . import api_bp


@api_bp.route('/decision-tree/predict', methods=['POST'])
def decision_tree_predict():
    """
    POST /api/decision-tree/predict

    Accepts an applicant feature payload, validates it, executes the R
    prediction script through the RRunner bridge, and returns the model's
    risk class, confidence, class probabilities and decision path.

    Request body:
        {"age": 32, "income": 95000, "credit_score": 735,
         "existing_loans": 1, "employment_years": 5}

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

    validated_input, errors = validate_applicant_input(payload)
    if errors:
        return jsonify({
            "success": False,
            "model": MODEL_NAME,
            "error": " ".join(errors),
            "errors": errors
        }), 400

    result = predict_risk(validated_input)
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/decision-tree/metrics', methods=['GET'])
def decision_tree_metrics():
    """
    GET /api/decision-tree/metrics

    Returns the real accuracy, precision, recall, F1 score, confusion matrix
    and per-class metrics produced by the R training script, read from the
    metrics artifact that train.R writes. Nothing is hard-coded.
    """
    result = get_training_metrics()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/decision-tree/tree', methods=['GET'])
def decision_tree_structure():
    """
    GET /api/decision-tree/tree

    Returns the fitted rpart tree — every node, its split rule, the branch
    conditions and its class distribution — so the frontend can visualise the
    real model rather than a decorative diagram.
    """
    result = get_tree_structure()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/decision-tree/schema', methods=['GET'])
def decision_tree_schema():
    """
    GET /api/decision-tree/schema

    Returns each input feature with its label and the valid range recorded at
    training time, plus the model's class set, so the frontend form can
    constrain inputs to real data.
    """
    schema = get_feature_schema()
    return jsonify({
        "success": True,
        "model": MODEL_NAME,
        "target": "risk",
        "model_trained": is_model_trained(),
        "train_script": TRAIN_SCRIPT,
        **schema
    }), 200
