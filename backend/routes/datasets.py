"""
AI Insight Hub — Datasets API Routes
Provides endpoints to inspect, validate, and retrieve dataset summaries.
"""
from flask import jsonify
from services.dataset_validator import validate_all_datasets, validate_dataset, DATASET_SCHEMAS
from . import api_bp


@api_bp.route('/datasets', methods=['GET'])
def get_datasets_overview():
    """Returns overview of all available benchmark datasets and schemas."""
    return jsonify({
        "status": "success",
        "total_datasets": len(DATASET_SCHEMAS),
        "schemas": DATASET_SCHEMAS
    }), 200


@api_bp.route('/datasets/validate', methods=['GET'])
def validate_datasets_endpoint():
    """Runs data integrity checks on all project datasets."""
    report = validate_all_datasets()
    status_code = 200 if report["all_valid"] else 422
    return jsonify(report), status_code


@api_bp.route('/datasets/<filename>/validate', methods=['GET'])
def validate_single_dataset_endpoint(filename: str):
    """Validates a specific dataset file."""
    report = validate_dataset(filename)
    status_code = 200 if report["valid"] else 404 if "File does not exist" in str(report.get("errors", [])) else 422
    return jsonify(report), status_code
