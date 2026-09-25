"""
AI Insight Hub — R Engine REST API Routes
Provides endpoints to test R execution environment, verify packages, and retrieve status.
"""
from flask import request, jsonify
from services.r_runner import r_runner
from . import api_bp


@api_bp.route('/r-engine/test', methods=['GET', 'POST'])
def test_r_engine():
    """
    Executes r_models/test_engine.R and returns structured JSON diagnostics.
    Handles missing R, timeouts, and execution errors gracefully.
    """
    # Accept optional request JSON payload if provided in POST
    input_data = None
    if request.method == 'POST' and request.is_json:
        try:
            input_data = request.get_json()
        except Exception:
            input_data = None

    result = r_runner.execute_script("r_models/test_engine.R", input_data=input_data)

    status_code = 200 if result.get("success") else 503
    return jsonify(result), status_code


@api_bp.route('/r-engine/status', methods=['GET'])
def get_r_engine_status():
    """
    Quick status probe for R execution engine without running external test scripts.
    """
    version_info = r_runner.get_version_info()
    is_avail = version_info.get("available", False)

    response = {
        "engine": "R",
        "available": is_avail,
        "status": "online" if is_avail else "not_connected",
        "version_info": version_info
    }
    return jsonify(response), 200 if is_avail else 503
