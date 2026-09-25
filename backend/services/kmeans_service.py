"""
AI Insight Hub — Phase 6 K-Means Service
Handles K-Means model artifact access, customer input validation, and
orchestration of the R assignment script through the existing RRunner bridge.

Responsibilities:
  - Resolve on-disk locations of the trained model artifacts
  - Validate incoming customer input against the ranges recorded at training time
  - Execute r_models/kmeans/predict.R via RRunner (never retrains)
  - Read the real metrics / clusters / profiles artifacts produced by train.R

K-Means is unsupervised, so this service does not predict a class. It assigns an
incoming customer to the nearest of the SAVED cluster centres, which is the same
nearest-centre rule stats::kmeans() used to assign its own training rows.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from services.r_runner import r_runner

logger = logging.getLogger("ai_insight_hub.kmeans")

# Model identity — the single source of truth for the Phase 6 model
MODEL_NAME = "K-Means"
PREDICT_SCRIPT = "r_models/kmeans/predict.R"
TRAIN_SCRIPT = "r_models/kmeans/train.R"

# Canonical feature order, matching the scaled training matrix in train.R.
# customer_id is deliberately absent: it is the identifier, never a feature.
FEATURES: Tuple[str, ...] = (
    "age",
    "annual_income",
    "spending_score",
    "purchase_frequency",
)

# The identifier column in customers.csv, used to label cluster assignments.
IDENTIFIER_COLUMN = "customer_id"

# Human-readable labels for validation errors, so the frontend can show
# user-friendly messages without duplicating the field list.
FIELD_LABELS: Dict[str, str] = {
    "age": "Age",
    "annual_income": "Annual Income",
    "spending_score": "Spending Score",
    "purchase_frequency": "Purchase Frequency",
}

# Domain constraints the dataset schema itself defines. These match the numeric
# rules in dataset_validator.DATASET_SCHEMAS["customers.csv"] and the checks
# train.R applies. The min/max observed in the trained data are applied on top
# of these (see `_get_feature_ranges`).
FIELD_RULES: Dict[str, Dict[str, Any]] = {
    "age": {"min": 0, "max": 120, "label": "Age"},
    "annual_income": {"min": 0, "label": "Annual Income"},
    "spending_score": {"min": 0, "max": 100, "label": "Spending Score"},
    "purchase_frequency": {"min": 0, "label": "Purchase Frequency"},
}


def get_kmeans_dir() -> str:
    """Returns the absolute path to the r_models/kmeans directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "r_models", "kmeans")


def get_model_path() -> str:
    """Absolute path to the trained model bundle (model.rds)."""
    return os.path.join(get_kmeans_dir(), "model.rds")


def get_metrics_path() -> str:
    """Absolute path to the training metrics artifact (metrics.json)."""
    return os.path.join(get_kmeans_dir(), "metrics.json")


def get_clusters_path() -> str:
    """Absolute path to the per-customer cluster assignment artifact."""
    return os.path.join(get_kmeans_dir(), "clusters.json")


def get_profiles_path() -> str:
    """Absolute path to the per-cluster profile artifact."""
    return os.path.join(get_kmeans_dir(), "profiles.json")


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
    dataset = metrics.get("dataset", {}) or {}
    ranges = dataset.get("feature_ranges", {}) or {}
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
    Returns the input schema for the frontend: each clustering feature with its
    label and the valid range recorded at training time.
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
            "rule_min": rules.get("min"),
            "rule_max": rules.get("max"),
        })
    return {
        "features": fields,
        "identifier_column": IDENTIFIER_COLUMN,
        "model_trained": is_model_trained(),
    }


def get_model_config() -> Dict[str, Any]:
    """
    Returns the trained model's configuration: which K was selected, which
    features the clustering ran on, and the range each feature covered in the
    training data.

    Every value is read from metrics.json, which train.R wrote out of the real
    dataset, so the frontend can build its form and its input hints from the
    actual model rather than from a duplicated constant.

    :returns: A Flask-ready response dict including a 'status_code' key.
    """
    metrics = _read_json(get_metrics_path())
    if not metrics:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Model configuration is not available. "
                "Run 'Rscript r_models/kmeans/train.R' to train the model."
            ),
            "status_code": 503,
        }

    dataset = metrics.get("dataset", {}) or {}
    ranges = dataset.get("feature_ranges", {}) or {}
    scaling = metrics.get("feature_scaling", {}) or {}
    selection = metrics.get("k_selection", {}) or {}
    scaler_center = scaling.get("center", {}) or {}
    scaler_scale = scaling.get("scale", {}) or {}

    features = []
    for name in FEATURES:
        bounds = ranges.get(name, {}) or {}
        features.append({
            "name": name,
            "label": FIELD_LABELS.get(name, name),
            "min": bounds.get("min"),
            "max": bounds.get("max"),
            "mean": scaler_center.get(name),
            "scale": scaler_scale.get(name),
        })

    return {
        "success": True,
        "model": metrics.get("model", MODEL_NAME),
        "algorithm": metrics.get("algorithm"),
        "task": metrics.get("task"),
        "learning_type": metrics.get("learning_type", "unsupervised"),
        "trained_at": metrics.get("trained_at"),
        "model_trained": is_model_trained(),
        "selected_k": metrics.get("selected_k"),
        "k_grid": selection.get("k_grid", []),
        "k_selection_method": selection.get("method"),
        "identifier_column": dataset.get("identifier_column", IDENTIFIER_COLUMN),
        "excluded_from_clustering": dataset.get("excluded_from_clustering", []),
        "feature_names": dataset.get("features", list(FEATURES)),
        "features": features,
        "feature_ranges": ranges,
        "scaling_method": scaling.get("method"),
        "total_records": dataset.get("rows_clean", 0) or 0,
        "model_file": "r_models/kmeans/model.rds",
        "status_code": 200,
    }


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


def validate_customer_input(payload: Any) -> Tuple[Dict[str, float], List[str]]:
    """
    Validates a customer feature payload for cluster assignment.

    Checks, in order of usefulness to the caller:
      - payload is a JSON object
      - every required feature is present
      - every value is a finite number
      - the documented domain bounds (age 0-120, spending_score 0-100, ...)
      - no unrecognised fields

    Values outside the dataset's observed range are NOT rejected — a centroid
    model can still place such a customer, the nearest centre is simply further
    away than usual — and the caller reports them as a warning via
    `find_extrapolation_warnings`.

    :returns: (cleaned_values, errors) where errors is a list of
              user-facing messages (empty when the input is valid).
    """
    errors: List[str] = []
    cleaned: Dict[str, float] = {}

    if not isinstance(payload, dict):
        return cleaned, ["Request body must be a JSON object containing the customer features."]

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
            threshold = rules.get("min")
            if threshold is not None and number < float(threshold):
                errors.append(f"{label} must be {_trim(float(threshold))} or greater.")
                continue
            ceiling = rules.get("max")
            if ceiling is not None and number > float(ceiling):
                errors.append(
                    f"{label} must be at most {_trim(float(ceiling))}."
                )
                continue

        # A value outside the dataset's observed range is still assignable, so it
        # is not rejected here — `find_extrapolation_warnings` reports it
        # alongside the assignment instead.
        cleaned[name] = number

    # Reject unexpected keys so a typo like "income" is not silently ignored.
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

    A centroid model always has a nearest centre, so such a value is still
    answered — but the answer is an extrapolation rather than an interpolation,
    and the caller is told which inputs to be careful about.

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
                    f"was trained on, so this assignment extrapolates beyond the dataset."
                ),
            })

    return warnings


def assign_cluster(validated_input: Dict[str, float]) -> Dict[str, Any]:
    """
    Executes the R assignment script against the saved model via RRunner.

    The centres and the scaling are loaded from model.rds by predict.R — this
    function never retrains the model.

    :returns: A Flask-ready response dict including a 'status_code' key.
    """
    if not is_model_trained():
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "The K-Means model has not been trained yet. "
                "Run 'Rscript r_models/kmeans/train.R' to generate model.rds."
            ),
            "status_code": 503,
        }

    result = r_runner.execute_script(PREDICT_SCRIPT, input_data=validated_input)

    if not result.get("success"):
        logger.error("R assignment script failed: %s", result.get("error"))
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": result.get("error") or "The R assignment engine failed to run.",
            "message": result.get("message") or "R execution failure.",
            "status_code": 502,
        }

    data = result.get("data")
    if not isinstance(data, dict):
        logger.error("R assignment script returned unparseable output: %s", result.get("stdout"))
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

    cluster = data.get("cluster")
    if not isinstance(cluster, int) or isinstance(cluster, bool) or cluster < 1:
        logger.error(f"R returned an invalid cluster id: {cluster!r}")
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": f"The R engine returned an unexpected cluster id: {cluster!r}",
            "status_code": 502,
        }

    # The cluster must be one the model was actually trained on. Anything else
    # means the bundle and the model disagree.
    selected_k = data.get("k")
    if isinstance(selected_k, int) and not isinstance(selected_k, bool) and cluster > selected_k:
        logger.error(f"R returned cluster {cluster} for a model with K = {selected_k}")
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                f"The R engine returned cluster {cluster}, but the model was "
                f"trained with K = {selected_k}."
            ),
            "status_code": 502,
        }

    response: Dict[str, Any] = {
        "success": True,
        "model": data.get("model", MODEL_NAME),
        "algorithm": data.get("algorithm"),
        "task": data.get("task", "Customer Segmentation"),
        "learning_type": data.get("learning_type", "unsupervised"),
        "cluster": cluster,
        "cluster_label": data.get("cluster_label"),
        "cluster_size": data.get("cluster_size"),
        "segment": data.get("segment"),
        "distance": data.get("distance"),
        "distance_original_units": data.get("distance_original_units"),
        "distance_basis": data.get("distance_basis"),
        "runner_up_distance": data.get("runner_up_distance"),
        "separation_ratio": data.get("separation_ratio"),
        "distances": data.get("distances", {}),
        "cluster_centers_original": data.get("cluster_centers_original", {}),
        "inputs": data.get("inputs", validated_input),
        "scaled_inputs": data.get("scaled_inputs", {}),
        "features": data.get("features", list(FEATURES)),
        "k": data.get("k"),
        "projection": data.get("projection"),
        "warnings": find_extrapolation_warnings(validated_input),
        "scaling_applied": data.get("scaling_applied"),
        "trained_at": data.get("trained_at"),
        "r_version": data.get("r_version"),
        "execution_time_ms": result.get("execution_time_ms"),
        "status_code": 200,
    }

    # Surface the model's real training metrics alongside each assignment so the
    # UI never has to hard-code a number.
    for metric_key in ("model_wss", "model_silhouette_score", "model_records_clustered"):
        if data.get(metric_key) is not None:
            response[metric_key] = data[metric_key]

    return response


def get_training_metrics() -> Dict[str, Any]:
    """
    Returns the real metrics produced by train.R, read from metrics.json.

    K-Means is unsupervised, so these are internal clustering diagnostics (WSS,
    between-cluster ratio, silhouette) rather than predictive accuracy.
    """
    metrics = _read_json(get_metrics_path())
    if not metrics:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Model metrics are not available. "
                "Run 'Rscript r_models/kmeans/train.R' to train the model."
            ),
            "status_code": 503,
        }

    values = metrics.get("metrics", {}) or {}
    required = ("wss", "silhouette_score", "between_cluster_ratio")
    if any(values.get(key) is None for key in required):
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "The metrics artifact is incomplete — expected wss, "
                "silhouette_score and between_cluster_ratio."
            ),
            "status_code": 500,
        }

    return {
        "success": True,
        "model": metrics.get("model", MODEL_NAME),
        "algorithm": metrics.get("algorithm"),
        "task": metrics.get("task"),
        "learning_type": metrics.get("learning_type", "unsupervised"),
        "trained_at": metrics.get("trained_at"),
        "r_version": metrics.get("r_version"),
        "random_seed": metrics.get("random_seed"),
        "selected_k": metrics.get("selected_k"),
        "metrics": values,
        "dataset": metrics.get("dataset", {}),
        "feature_scaling": metrics.get("feature_scaling", {}),
        "k_selection": metrics.get("k_selection", {}),
        "k_comparison": metrics.get("k_comparison", []),
        "clusters": metrics.get("clusters", []),
        "cluster_centers": metrics.get("cluster_centers", []),
        "cluster_profiles": metrics.get("cluster_profiles", []),
        "visualization": metrics.get("visualization", {}),
        "reproducibility": metrics.get("reproducibility", {}),
        "model_info": metrics.get("model_info", {}),
        "model_file": "r_models/kmeans/model.rds",
        "status_code": 200,
    }


def get_cluster_assignments() -> Dict[str, Any]:
    """
    Returns the real per-customer cluster assignments from clusters.json.

    The artifact was written by train.R out of the dataset the model actually
    clustered, so every record here carries a real cluster. Only the identifier
    and the clustering features are published — no other column from the source
    file is exposed.
    """
    payload = _read_json(get_clusters_path())
    if not payload:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Cluster assignments are not available. "
                "Run 'Rscript r_models/kmeans/train.R' to cluster the dataset."
            ),
            "status_code": 503,
        }

    records = payload.get("clusters", [])
    if not isinstance(records, list) or not records:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": "The clusters artifact contains no customer records.",
            "status_code": 500,
        }

    return {
        "success": True,
        "model": payload.get("model", MODEL_NAME),
        "task": payload.get("task"),
        "trained_at": payload.get("trained_at"),
        "dataset": payload.get("dataset"),
        "total_records": payload.get("total_records", len(records)),
        "identifier_column": payload.get("identifier_column", IDENTIFIER_COLUMN),
        "features": payload.get("features", list(FEATURES)),
        "selected_k": payload.get("selected_k"),
        "silhouette_score": payload.get("silhouette_score"),
        "wss": payload.get("wss"),
        "cluster_sizes": payload.get("cluster_sizes", {}),
        "cluster_labels": payload.get("cluster_labels", {}),
        "cluster_centers_projected": payload.get("cluster_centers_projected", []),
        "clusters": records,
        "clusters_file": "r_models/kmeans/clusters.json",
        "status_code": 200,
    }


def get_cluster_profiles() -> Dict[str, Any]:
    """
    Returns the per-cluster profiles derived from each cluster's measured centre.

    Read from profiles.json, the dedicated artifact train.R wrote out of the real
    clustering. Each profile carries the cluster's size, its share of the
    dataset, the mean feature values of its members, the centre in both scaled and
    original units, and the automatically generated descriptive label.

    The silhouette and WSS reported alongside the profiles are read from
    metrics.json rather than duplicated into the profiles file, so the two
    artifacts cannot drift apart.
    """
    payload = _read_json(get_profiles_path())
    if not payload:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": (
                "Cluster profiles are not available. "
                "Run 'Rscript r_models/kmeans/train.R' to cluster the dataset."
            ),
            "status_code": 503,
        }

    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        return {
            "success": False,
            "model": MODEL_NAME,
            "error": "The profiles artifact contains no cluster profiles.",
            "status_code": 500,
        }

    summaries = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        summaries.append({
            "cluster": profile.get("cluster"),
            "label": profile.get("label"),
            "label_is_generic": profile.get("label_is_generic"),
            "segment": profile.get("segment"),
            "size": profile.get("size"),
            "size_share": profile.get("size_share"),
            "descriptors": profile.get("descriptors", []),
            "mean_features": profile.get("mean_features", {}),
            "original_mean_features": profile.get("original_mean_features", {}),
            "center_scaled": profile.get("center_scaled", {}),
            "center_original": profile.get("center_original", {}),
            "center_vs_population_sd": profile.get("center_vs_population_sd", {}),
            "wss_share": profile.get("wss_share"),
            "mean_distance_to_center": profile.get("mean_distance_to_center"),
        })

    # The clustering diagnostics are not stored in the profiles file; take them
    # from metrics.json so the two artifacts stay independent.
    metrics = _read_json(get_metrics_path()) or {}
    values = metrics.get("metrics", {}) or {}

    return {
        "success": True,
        "model": payload.get("model", MODEL_NAME),
        "task": payload.get("task"),
        "trained_at": payload.get("trained_at"),
        "selected_k": payload.get("selected_k"),
        "features": payload.get("features", list(FEATURES)),
        "identifier_column": payload.get("identifier_column", IDENTIFIER_COLUMN),
        "total_records": payload.get("total_records", 0) or 0,
        "feature_ranges": payload.get("feature_ranges", {}),
        "naming": payload.get("naming"),
        "comparison": payload.get("comparison", {}),
        "scaling": payload.get("scaling", {}),
        "profiles": summaries,
        "silhouette_score": values.get("silhouette_score"),
        "wss": values.get("wss"),
        "profiles_file": "r_models/kmeans/profiles.json",
        "status_code": 200,
    }
