"""
AI Insight Hub — Phase 5 KNN Service
Handles K-Nearest Neighbors model artifact access, student input validation, and
orchestration of the R prediction script through the existing RRunner bridge.

Responsibilities:
  - Resolve on-disk locations of the trained model artifacts
  - Validate incoming student input against the ranges recorded at training time
  - Execute r_models/knn/predict.R via RRunner (never retrains)
  - Read the real metrics / K-comparison artifacts produced by train.R
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from services.r_runner import r_runner

logger = logging.getLogger("ai_insight_hub.knn")

# Model identity — the single source of truth for the Phase 5 model
MODEL_NAME = "K-Nearest Neighbors"
PREDICT_SCRIPT = "r_models/knn/predict.R"
TRAIN_SCRIPT = "r_models/knn/train.R"

# Canonical feature order, matching the scaled training matrix in train.R
FEATURES: Tuple[str, ...] = (
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "practical_score",
)

# The three performance tiers declared by the dataset. A class outside this set
# in an R response is rejected rather than passed through to the user.
CLASSES: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")

# Human-readable labels for validation errors, so the frontend can show
# user-friendly messages without duplicating the field list.
FIELD_LABELS: Dict[str, str] = {
    "study_hours": "Study Hours",
    "attendance": "Attendance",
    "previous_score": "Previous Score",
    "assignments_completed": "Assignments Completed",
    "practical_score": "Practical Score",
}

# Domain constraints the dataset schema itself defines. The min/max observed in
# the trained data are applied on top of these (see `_get_feature_ranges`).
FIELD_RULES: Dict[str, Dict[str, Any]] = {
    "study_hours": {"min": 0, "label": "Study Hours"},
    "attendance": {"min": 0, "max": 100, "label": "Attendance"},
    "previous_score": {"min": 0, "max": 100, "label": "Previous Score"},
    "assignments_completed": {"min": 0, "label": "Assignments Completed"},
    "practical_score": {"min": 0, "max": 100, "label": "Practical Score"},
}


def get_knn_dir() -> str:
    """Returns the absolute path to the r_models/knn directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "r_models", "knn")


def get_model_path() -> str:
    """Absolute path to the trained model bundle (model.rds)."""
    return os.path.join(get_knn_dir(), "model.rds")


def get_metrics_path() -> str:
    """Absolute path to the training metrics artifact (metrics.json)."""
    return os.path.join(get_knn_dir(), "metrics.json")


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
    Returns the input schema for the frontend: each feature with its label and
    the valid range recorded at training time.
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
    return {"features": fields, "classes": list(CLASSES)}


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


def validate_student_input(payload: Any) -> Tuple[Dict[str, float], List[str]]:
    """
    Validates a student feature payload.

    Checks, in order of usefulness to the caller:
      - payload is a JSON object
      - every required feature is present
      - every value is a finite number
      - study_hours / assignments_completed >= 0
      - attendance / previous_score / practical_score within 0-100
      - no unrecognised fields

    Values outside the dataset's observed range are NOT rejected — a KNN model
    can still classify them, the nearest training rows are simply further away
    than usual — and the caller reports them as a warning via
    `find_extrapolation_warnings`.

    :returns: (cleaned_values, errors) where errors is a list of
              user-facing messages (empty when the input is valid).
    """
    errors: List[str] = []
    cleaned: Dict[str, float] = {}

    if not isinstance(payload, dict):
        return cleaned, ["Request body must be a JSON object containing the student features."]

    for name in FEATURES:
        label = FIELD_LABELS.get(name, name)

        if name not in payload or payload[name] is None:
            errors.append(f"{label} is required.")
            continue

        number = _coerce_number(payload[name])
        if number is None:
            errors.append(f"{label} must be a number.")
            continue

        # Static domain rules from the dataset specification
        rules = FIELD_RULES.get(name)
        if rules:
            threshold = float(rules["min"])
            if number < threshold:
                errors.append(f"{label} must be {int(threshold)} or greater.")
                continue
            if "max" in rules and number > float(rules["max"]):
                errors.append(
                    f"{label} must be at most {int(float(rules['max']))}."
                )
                continue

        # A value outside the dataset's observed range is still classifiable, so
        # it is not rejected here — `find_extrapolation_warnings` reports it
        # alongside the prediction instead.
        cleaned[name] = number

    # Reject unexpected keys so a typo like "studhours" is not silently ignored.
    unknown = [key for key in payload if key not in FEATURES]
    if unknown:
        errors.append(
            "Unrecognised field(s): " + ", ".join(sorted(str(k) for k in unknown)) +
            ". Expected: " + ", ".join(FEATURES) + "."
        )

    return cleaned, errors


def find_extrapolation_warnings(cleaned: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Flags any feature whose value falls outside the range the model was trained
    on.

    KNN always has a nearest neighbour, so such a value is still answered — but
    the answer is an extrapolation rather than an interpolation, and the caller
    is told which inputs to be careful about.

    :returns: one entry per out-of-range feature, each with the supplied value
              and the trained min/max. Empty when everything is in range.
    """
    ranges = _get_feature_ranges()
    warnings: List[Dict[str, Any]] = []

    for name in FEATURES:
        bounds = ranges.get(name)
        if not bounds or name not in cleaned:
            continue
        value = cleaned[name]
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append({
                "field": name,
                "label": FIELD_LABELS.get(name, name),
                "value": value,
                "trained_min": bounds["min"],
                "trained_max": bounds["max"],
                "message": (
                    f"{FIELD_LABELS.get(name, name)} ({_trim(value)}) is outside the "
                    f"{_trim(bounds['min'])} – {_trim(bounds['max'])} range the model "
                    f"was trained on, so this prediction is extrapolated."
                ),
            })

    return warnings


def predict_performance(validated_input: Dict[str, float]) -> Dict[str, Any]:
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
                "The KNN model has not been trained yet. "
                "Run 'Rscript r_models/knn/train.R' to generate model.rds."
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
    if not isinstance(prediction, str) or prediction not in CLASSES:
        logger.error(f"R returned an invalid performance class: {prediction!r}")
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": f"The R engine returned an unexpected performance class: {prediction!r}",
            "status_code": 502,
        }

    response: Dict[str, Any] = {
        "success": True,
        "model": data.get("model", MODEL_NAME),
        "algorithm": data.get("algorithm"),
        "prediction": prediction,
        "k": data.get("k"),
        "inputs": data.get("inputs", validated_input),
        "scaled_inputs": data.get("scaled_inputs", {}),
        "target": data.get("target", "performance"),
        "classes": data.get("classes", list(CLASSES)),
        "features": data.get("features", list(FEATURES)),
        "neighbors": data.get("neighbors", []),
        "neighbor_class_distribution": data.get("neighbor_class_distribution", {}),
        "confidence_basis": data.get("confidence_basis"),
        "warnings": find_extrapolation_warnings(validated_input),
        "trained_at": data.get("trained_at"),
        "r_version": data.get("r_version"),
        "class_package_version": data.get("class_package_version"),
        "scaling_applied": data.get("scaling_applied"),
        "execution_time_ms": result.get("execution_time_ms"),
        "status_code": 200,
    }

    # Confidence is only surfaced when R actually calculated one. It is derived
    # from the real neighbour vote, never invented.
    confidence = data.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        response["confidence"] = round(float(confidence), 4)

    # Surface the model's real training metrics alongside each prediction so the
    # UI never has to hard-code a performance number.
    for metric_key in ("accuracy", "precision", "recall", "f1_score"):
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
                "Run 'Rscript r_models/knn/train.R' to train the model."
            ),
            "status_code": 503,
        }

    values = metrics.get("metrics", {})
    required = ("accuracy", "precision", "recall", "f1_score")
    if any(values.get(key) is None for key in required):
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "The metrics artifact is incomplete — expected accuracy, "
                "precision, recall and f1_score."
            ),
            "status_code": 500,
        }

    return {
        "success": True,
        "model": metrics.get("model", MODEL_NAME),
        "algorithm": metrics.get("algorithm"),
        "task": metrics.get("task"),
        "trained_at": metrics.get("trained_at"),
        "classes": metrics.get("classes", list(CLASSES)),
        "selected_k": metrics.get("selected_k"),
        "metrics": values,
        "per_class_metrics": metrics.get("per_class_metrics", {}),
        "confusion_matrix": metrics.get("confusion_matrix", {}),
        "dataset": metrics.get("dataset", {}),
        "feature_scaling": metrics.get("feature_scaling", {}),
        "model_info": metrics.get("model_info", {}),
        "class_distribution_note": metrics.get("class_distribution_note", {}),
        "k_comparison": metrics.get("k_comparison", []),
        "model_file": "r_models/knn/model.rds",
        "status_code": 200,
    }


def get_model_config() -> Dict[str, Any]:
    """
    Returns the model's public configuration: selected K, feature names and
    ranges, class set and the K-comparison summary.

    Only information the frontend needs to build and explain the form is
    exposed — no training rows, no scaled matrix and no internal artifact
    layout.
    """
    metrics = _read_json(get_metrics_path())
    if not metrics:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Model configuration is not available. "
                "Run 'Rscript r_models/knn/train.R' to train the model."
            ),
            "status_code": 503,
        }

    selected_k = metrics.get("selected_k")
    if not isinstance(selected_k, int):
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "The metrics artifact does not record a selected K — retrain "
                "with train.R."
            ),
            "status_code": 500,
        }

    model_info = metrics.get("model_info", {}) or {}
    k_selection = model_info.get("k_selection", {}) or {}
    dataset = metrics.get("dataset", {}) or {}
    scaling = metrics.get("feature_scaling", {}) or {}
    note = metrics.get("class_distribution_note", {}) or {}

    return {
        "success": True,
        "model": metrics.get("model", MODEL_NAME),
        "algorithm": metrics.get("algorithm"),
        "task": metrics.get("task"),
        "target": metrics.get("target", "performance"),
        "classes": metrics.get("classes", list(CLASSES)),
        "selected_k": selected_k,
        "k_grid": model_info.get("k_grid", []),
        "k_selection": k_selection,
        "features": dataset.get("features", list(FEATURES)),
        "feature_ranges": dataset.get("feature_ranges", {}),
        "feature_labels": {name: FIELD_LABELS.get(name, name) for name in FEATURES},
        "feature_scaling": {
            "method": scaling.get("method"),
            "reason": scaling.get("reason"),
        },
        "implementation": model_info.get("implementation"),
        "distance": model_info.get("distance"),
        "vote": model_info.get("vote"),
        "rows_train": dataset.get("rows_train"),
        "rows_test": dataset.get("rows_test"),
        "trained_at": metrics.get("trained_at"),
        "class_distribution_note": {
            "classes_without_test_support": note.get("classes_without_test_support", []),
            "classes_unreachable_at_selected_k": note.get(
                "classes_unreachable_at_selected_k", []
            ),
            "message": note.get("message"),
        },
        "model_trained": is_model_trained(),
        "train_script": TRAIN_SCRIPT,
        "status_code": 200,
    }
