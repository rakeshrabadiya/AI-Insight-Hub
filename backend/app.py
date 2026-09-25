"""
AI Insight Hub - Flask Application Entry Point
Multi-Model Machine Learning Analytics Platform
"""
import os
import sys
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes import api_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] in %(module)s: %(message)s"
)
logger = logging.getLogger("ai_insight_hub")


def create_app():
    """Application factory for AI Insight Hub Flask backend."""
    app = Flask(__name__)

    # Enable Cross-Origin Resource Sharing (CORS) for all frontend origins
    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        supports_credentials=True
    )

    # Register Blueprints
    app.register_blueprint(api_bp)

    # Root route for API documentation/welcome
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "platform": "AI Insight Hub",
            "description": "Multi-Model Machine Learning Analytics Platform",
            "version": "1.0.0",
            "phase": "Phase 3 - Linear Regression Property Price Prediction",
            "status": "online",
            "endpoints": {
                "health": "/api/health",
                "r_engine": "/api/r-engine/status",
                "datasets": "/api/datasets",
                "regression_predict": "/api/regression/predict",
                "regression_metrics": "/api/regression/metrics",
                "regression_evaluation": "/api/regression/evaluation",
                "regression_schema": "/api/regression/schema"
            }
        }), 200

    # Custom 404 handler
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found on this server.",
            "status_code": 404
        }), 404

    # Custom 405 handler
    @app.errorhandler(405)
    def method_not_allowed_error(error):
        return jsonify({
            "error": "Method Not Allowed",
            "message": "The method is not allowed for the requested URL.",
            "status_code": 405
        }), 405

    # Custom 500 handler
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred on the server.",
            "status_code": 500
        }), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    logger.info(f"Starting AI Insight Hub Backend on http://127.0.0.1:{port} (Debug: {debug})")
    app.run(host="127.0.0.1", port=port, debug=debug)
