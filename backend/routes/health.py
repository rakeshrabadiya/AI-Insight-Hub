"""
Health Check and Diagnostics Routes.
Provides system status for the frontend dashboard.
"""
from datetime import datetime, timezone
from flask import jsonify
from services.r_runner import r_runner
from . import api_bp


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint returning the status of the Flask backend,
    the connection status of the R machine learning engine, and timestamp.
    """
    is_r_connected = r_runner.is_available()
    response = {
        "status": "online",
        "message": "AI Insight Hub API is running",
        "r_engine": "online" if is_r_connected else "not_connected",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return jsonify(response), 200
