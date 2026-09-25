"""
AI Insight Hub — K-Means API Routes (Phase 6)
Exposes the trained K-Means clustering model: customer segment assignment, model
metrics with the K-selection evidence, the per-customer cluster assignments and
the per-cluster profiles.
"""
from flask import request, jsonify

from services.kmeans_service import (
    MODEL_NAME,
    TRAIN_SCRIPT,
    assign_cluster,
    get_cluster_assignments,
    get_cluster_profiles,
    get_feature_schema,
    get_model_config,
    get_training_metrics,
    is_model_trained,
    validate_customer_input,
)
from . import api_bp


@api_bp.route('/kmeans/predict', methods=['POST'])
def kmeans_predict():
    """
    POST /api/kmeans/predict

    Accepts a customer feature payload, validates it, executes the R assignment
    script through the RRunner bridge, and returns the nearest of the saved
    cluster centres along with the distance that decided it.

    K-Means is unsupervised, so this is an ASSIGNMENT rather than a class
    prediction: the answer is the centroid the customer sits closest to, which is
    the same nearest-centre rule kmeans() used on its own training rows.

    Request body:
        {"age": 42, "annual_income": 85, "spending_score": 55,
         "purchase_frequency": 28}

    Status codes:
        200 — segment assigned
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

    validated_input, errors = validate_customer_input(payload)
    if errors:
        return jsonify({
            "success": False,
            "model": MODEL_NAME,
            "error": " ".join(errors),
            "errors": errors
        }), 400

    result = assign_cluster(validated_input)
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/kmeans/metrics', methods=['GET'])
def kmeans_metrics():
    """
    GET /api/kmeans/metrics

    Returns the real clustering diagnostics produced by the R training script:
    the within-cluster sum of squares, the between-cluster ratio, the average
    silhouette width, the per-cluster sizes and the full K-comparison that
    explains which K was chosen and why. Nothing is hard-coded.

    Note the metric set is internal, not predictive: an unsupervised model has no
    held-out accuracy to report.
    """
    result = get_training_metrics()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/kmeans/clusters', methods=['GET'])
def kmeans_clusters():
    """
    GET /api/kmeans/clusters

    Returns the actual cluster the model assigned to every customer in
    customers.csv, read from the clusters artifact train.R wrote out of the real
    clustering. Each record carries the customer's identifier, the four
    clustering features, the cluster id, the distance to that cluster's centre
    and the 2-D position used to draw the dashboard scatter plot.
    """
    result = get_cluster_assignments()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/kmeans/profiles', methods=['GET'])
def kmeans_profiles():
    """
    GET /api/kmeans/profiles

    Returns the real per-cluster profile statistics: how many customers each
    cluster holds, its mean feature values, its centre in both scaled and
    original units, and the descriptive label train.R generated from that
    centre's actual position relative to the population.
    """
    result = get_cluster_profiles()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code


@api_bp.route('/kmeans/schema', methods=['GET'])
def kmeans_schema():
    """
    GET /api/kmeans/schema

    Returns each clustering feature with its label and the valid range recorded
    at training time, plus the identifier column, so the frontend form can
    constrain its inputs to the real data the model was fitted on.
    """
    schema = get_feature_schema()
    return jsonify({
        "success": True,
        "model": MODEL_NAME,
        "task": "Customer Segmentation",
        "model_trained": is_model_trained(),
        "train_script": TRAIN_SCRIPT,
        **schema
    }), 200


@api_bp.route('/kmeans/config', methods=['GET'])
def kmeans_config():
    """
    GET /api/kmeans/config

    Returns the trained model's configuration in one place: the model name, the
    K that was selected and the grid it was chosen from, the clustering feature
    names in the order the model uses them, and each feature's range across the
    training data.

    Every value is read from the artifact train.R produced, so this is a
    description of the model that is actually deployed rather than a restatement
    of the training configuration.
    """
    result = get_model_config()
    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code
