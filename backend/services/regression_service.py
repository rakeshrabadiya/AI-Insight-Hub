"""
AI Insight Hub — Phase 3 Regression Service
Handles Linear Regression model artifact access, input validation, and
orchestration of the R prediction script through the existing RRunner bridge.

Responsibilities:
  - Resolve on-disk locations of the trained model artifacts
  - Validate incoming property input against the ranges recorded at training time
  - Execute r_models/regression/predict.R via RRunner (never retrains)
  - Read the real metrics/evaluation artifacts produced by train.R
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from services.r_runner import r_runner

logger = logging.getLogger("ai_insight_hub.regression")

# Model identity — the single source of truth for the Phase 3 model
MODEL_NAME = "Linear Regression"
PREDICT_SCRIPT = "r_models/regression/predict.R"
TRAIN_SCRIPT = "r_models/regression/train.R"

# Canonical feature order, matching the formula fitted in train.R
FEATURES: Tuple[str, ...] = ("area", "bedrooms", "bathrooms", "location_score", "property_age")

# Human-readable labels for validation errors, so the frontend can show
# user-friendly messages without duplicating the field list.
FIELD_LABELS: Dict[str, str] = {
    "area": "Area (sq.ft)",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "location_score": "Location Score",
    "property_age": "Property Age",
}

# Rule constraints requested for each field. The `min`/`max` from the trained
# data are applied on top of these (see `_get_feature_ranges`).
FIELD_RULES: Dict[str, Dict[str, Any]] = {
    "area": {"min": 0, "exclusive_min": True, "label": "Area (sq.ft)"},
    "bedrooms": {"min": 0, "exclusive_min": True, "label": "Bedrooms"},
    "bathrooms": {"min": 0, "exclusive_min": True, "label": "Bathrooms"},
    "property_age": {"min": 0, "exclusive_min": False, "label": "Property Age"},
}


def get_regression_dir() -> str:
    """Returns the absolute path to the r_models/regression directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "r_models", "regression")


def get_model_path() -> str:
    """Absolute path to the trained model bundle (model.rds)."""
    return os.path.join(get_regression_dir(), "model.rds")


def get_metrics_path() -> str:
    """Absolute path to the training metrics artifact (metrics.json)."""
    return os.path.join(get_regression_dir(), "metrics.json")


def get_evaluation_path() -> str:
    """Absolute path to the test-set evaluation artifact (evaluation.json)."""
    return os.path.join(get_regression_dir(), "evaluation.json")


def is_model_trained() -> bool:
    """True when model.rds exists and is non-empty."""
    path = get_model_path()
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """Reads a JSON artifact, returning None if it is missing or unparseable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(f"Failed to read artifact {path}: {exc}")
        return None


def _get_feature_ranges() -> Dict[str, Dict[str, float]]:
    """
    Returns the per-feature min/max the model was trained on.

    These come from metrics.json, which train.R wrote from the cleaned dataset,
    so the API validates against real data rather than a hand-written constant.
    Falls back to an empty mapping when the artifact is unavailable, in which
    case only the static field rules apply.
    """
    metrics = _read_json(get_metrics_path())
    if not metrics:
        return {}
    dataset = metrics.get("dataset", {})
    ranges = dataset.get("feature_ranges", {})
    cleaned: Dict[str, Dict[str, float]] = {}
    for name, bounds in ranges.items():
        if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
            try:
                cleaned[name] = {"min": float(bounds["min"]), "max": float(bounds["max"])}
            except (TypeError, ValueError):
                continue
    return cleaned


def get_feature_schema() -> Dict[str, Any]:
    """
    Returns the input schema for the frontend: each feature with its label,
    unit, and the valid range recorded at training time.
    """
    ranges = _get_feature_ranges()
    fields = []
    for name in FEATURES:
        rules = FIELD_RULES.get(name, {})
        bounds = ranges.get(name, {})
        fields.append({
            "name": name,
            "label": FIELD_LABELS.get(name, name),
            "min": bounds.get("min"),
            "max": bounds.get("max"),
            "rule": rules.get("min"),
        })
    return {"features": fields}


def _coerce_number(value: Any) -> Optional[float]:
    """
    Converts a JSON value to a finite float.

    Returns None when the value cannot be a number. Booleans are rejected
    explicitly because Python treats bool as a subclass of int, and `True`
    would otherwise silently become 1.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
    else:
        return None

    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _trim(value: float) -> str:
    """Formats a float without trailing '.0' for whole numbers."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def validate_property_input(payload: Any) -> Tuple[Dict[str, float], List[str]]:
    """
    Validates a property feature payload.

    Checks, in order of usefulness to the caller:
      - payload is a JSON object
      - every required feature is present
      - every value is a finite number
      - area / bedrooms / bathrooms > 0
      - property_age >= 0
      - every value lies inside the dataset's observed range

    :returns: (cleaned_values, errors) where errors is a list of
              user-facing messages (empty when the input is valid).
    """
    errors: List[str] = []
    cleaned: Dict[str, float] = {}

    if not isinstance(payload, dict):
        return cleaned, ["Request body must be a JSON object containing the property features."]

    ranges = _get_feature_ranges()

    for name in FEATURES:
        label = FIELD_LABELS.get(name, name)

        if name not in payload or payload[name] is None:
            errors.append(f"{label} is required.")
            continue

        number = _coerce_number(payload[name])
        if number is None:
            errors.append(f"{label} must be a number.")
            continue

        # Static domain rules from the model specification
        rules = FIELD_RULES.get(name)
        if rules:
            threshold = float(rules["min"])
            if rules.get("exclusive_min") and number <= threshold:
                errors.append(f"{label} must be greater than {int(threshold)}.")
                continue
            if not rules.get("exclusive_min") and number < threshold:
                errors.append(f"{label} must be {int(threshold)} or greater.")
                continue

        # Range observed in the training data (e.g. location_score 3.1 - 9.7)
        bounds = ranges.get(name)
        if bounds:
            if number < bounds["min"] or number > bounds["max"]:
                errors.append(
                    f"{label} must be between {_trim(bounds['min'])} and "
                    f"{_trim(bounds['max'])} (the range the model was trained on)."
                )
                continue

        cleaned[name] = number

    # Reject unexpected keys so a typo like "bedroom" is not silently ignored.
    unknown = [key for key in payload if key not in FEATURES]
    if unknown:
        errors.append(
            "Unrecognised field(s): " + ", ".join(sorted(str(k) for k in unknown)) +
            ". Expected: " + ", ".join(FEATURES) + "."
        )

    return cleaned, errors


def predict_price(validated_input: Dict[str, float]) -> Dict[str, Any]:
    """
    Executes the R prediction script against the saved model via RRunner.

    The model is loaded from model.rds by predict.R — this function never
    retrains it.

    :returns: A Flask-ready response dict including a 'status_code' key.
    """
    if not is_model_trained():
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "The Linear Regression model has not been trained yet. "
                "Run 'Rscript r_models/regression/train.R' to generate model.rds."
            ),
            "status_code": 503,
        }

    result = r_runner.execute_script(PREDICT_SCRIPT, input_data=validated_input)

    if not result.get("success"):
        logger.error("R prediction script failed: %s", result.get("error"))
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": result.get("error") or "The R prediction engine failed to run.",
            "message": result.get("message") or "R execution failure.",
            "status_code": 502,
        }

    data = result.get("data")
    if not isinstance(data, dict):
        logger.error("R prediction script returned unparseable output: %s", result.get("stdout"))
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": "The R engine returned an unreadable response.",
            "status_code": 502,
        }

    # The R script can report a validation failure with a zero exit code, so the
    # success flag inside the payload must be checked explicitly.
    if not data.get("success"):
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": data.get("error") or "The model rejected the supplied input.",
            "errors": data.get("errors", []),
            "status_code": 400,
        }

    prediction = data.get("prediction")
    if not isinstance(prediction, (int, float)) or isinstance(prediction, bool):
        logger.error(f"R returned a non-numeric prediction: {prediction!r}")
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": "The R engine returned a non-numeric prediction.",
            "status_code": 502,
        }

    response: Dict[str, Any] = {
        "success": True,
        "model": data.get("model", MODEL_NAME),
        "algorithm": data.get("algorithm"),
        "prediction": round(float(prediction), 2),
        "unit": data.get("unit", "INR"),
        "inputs": data.get("inputs", validated_input),
        "target": data.get("target", "price"),
        "trained_at": data.get("trained_at"),
        "r_version": data.get("r_version"),
        "execution_time_ms": result.get("execution_time_ms"),
        "status_code": 200,
    }

    # Surface the model's real training metrics alongside each prediction so the
    # UI never has to hard-code a performance number.
    for metric_key in ("r2", "rmse", "mae"):
        value = data.get(f"model_{metric_key}")
        if value is not None:
            response[metric_key] = value

    return response


def get_training_metrics() -> Dict[str, Any]:
    """
    Returns the real metrics produced by train.R, read from metrics.json.
    """
    metrics = _read_json(get_metrics_path())
    if not metrics:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Model metrics are not available. "
                "Run 'Rscript r_models/regression/train.R' to train the model."
            ),
            "status_code": 503,
        }

    values = metrics.get("metrics", {})
    required = ("r2", "rmse", "mae")
    if any(values.get(key) is None for key in required):
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": "The metrics artifact is incomplete — expected r2, rmse and mae.",
            "status_code": 500,
        }

    return {
        "success": True,
        "model": metrics.get("model", MODEL_NAME),
        "algorithm": metrics.get("algorithm"),
        "trained_at": metrics.get("trained_at"),
        "metrics": {
            "r2": values.get("r2"),
            "rmse": values.get("rmse"),
            "mae": values.get("mae"),
            "adjusted_r2": values.get("adjusted_r2"),
            "train_r2": values.get("train_r2"),
            "train_rmse": values.get("train_rmse"),
            "train_mae": values.get("train_mae"),
        },
        "dataset": metrics.get("dataset", {}),
        "coefficients": metrics.get("coefficients", []),
        "model_file": "r_models/regression/model.rds",
        "status_code": 200,
    }


def get_evaluation_data() -> Dict[str, Any]:
    """
    Returns the held-out test-set actual vs predicted pairs written by train.R,
    used by the frontend scatter chart.
    """
    evaluation = _read_json(get_evaluation_path())
    if not evaluation:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Evaluation data is not available. "
                "Run 'Rscript r_models/regression/train.R' to train the model."
            ),
            "status_code": 503,
        }

    points = evaluation.get("points", [])
    if not points:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": "The evaluation artifact contains no test-set points.",
            "status_code": 500,
        }

    return {
        "success": True,
        "model": evaluation.get("model", MODEL_NAME),
        "unit": evaluation.get("unit", "INR"),
        "trained_at": evaluation.get("trained_at"),
        "count": evaluation.get("count", len(points)),
        "points": points,
        "status_code": 200,
    }
