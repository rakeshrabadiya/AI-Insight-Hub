"""
AI Insight Hub — Phase 3 Linear Regression Test Suite

Tests real behaviour end to end:
  1. Training artifacts exist (train.R, predict.R, model.rds, metrics.json)
  2. The metrics are real numbers produced by the trained R model
  3. Valid predictions return a genuine price from the saved model
  4. Predictions actually change when an input changes (proves no hard-coding)
  5. Predictions are reproducible across identical requests (saved-model reuse)
  6. Invalid input is rejected with clean 400 errors
  7. Missing input is rejected
  8. The prediction response has the documented shape
  9. R execution failures are handled without crashing the API
 10. The metrics, evaluation and schema endpoints serve real artifacts
 11. The model file is actually reused, not retrained per request

Every test exercises the real R engine or the real saved artifacts. No test
asserts a hard-coded price or a stubbed success value.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch

# Add project root and backend to sys.path (same bootstrap as test_backend.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.app import app
from backend.services import regression_service
from backend.services.regression_service import (
    FEATURES,
    MODEL_NAME,
    get_evaluation_data,
    get_metrics_path,
    get_model_path,
    get_training_metrics,
    is_model_trained,
    validate_property_input,
)

# A valid property inside the dataset's observed ranges
VALID_PAYLOAD = {
    "area": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "location_score": 8,
    "property_age": 5
}


class TestTrainingArtifacts(unittest.TestCase):
    """Phase 3 training scripts and the trained model must exist on disk."""

    def test_training_and_prediction_scripts_exist(self):
        """train.R and predict.R must be present in r_models/regression/."""
        for script in ("train.R", "predict.R"):
            path = os.path.join(PROJECT_ROOT, "r_models", "regression", script)
            self.assertTrue(os.path.isfile(path), f"{script} is missing at {path}")
            self.assertGreater(os.path.getsize(path), 0, f"{script} is empty")

    def test_trained_model_file_exists(self):
        """model.rds must exist and be non-empty after training."""
        model_path = get_model_path()
        self.assertTrue(os.path.isfile(model_path), "model.rds not found — run train.R")
        self.assertGreater(os.path.getsize(model_path), 0, "model.rds is empty")
        self.assertTrue(is_model_trained())

    def test_training_script_does_not_hardcode_metrics(self):
        """train.R must compute metrics, never assign a literal r2/rmse/mae."""
        with open(os.path.join(PROJECT_ROOT, "r_models", "regression", "train.R"), "r", encoding="utf-8") as handle:
            source = handle.read()

        # A hard-coded metric would look like: r2 <- 0.9
        for metric in ("r2", "rmse", "mae"):
            for literal in (f"{metric} <- 0.", f"{metric} <- 1.", f"{metric} <- 9"):
                self.assertNotIn(
                    literal, source,
                    f"train.R appears to hard-code {metric} via '{literal}'"
                )

        # The metrics must be derived from model predictions.
        self.assertIn("predict(", source, "train.R must call predict() to generate test predictions")

    def test_prediction_script_loads_saved_model(self):
        """predict.R must load model.rds and must not fit a new model."""
        with open(os.path.join(PROJECT_ROOT, "r_models", "regression", "predict.R"), "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("readRDS", source, "predict.R must load the saved model via readRDS()")
        self.assertIn("model.rds", source, "predict.R must reference model.rds")
        self.assertNotIn("lm(", source, "predict.R must NOT retrain the model with lm()")

    def test_metrics_artifact_is_real_json(self):
        """metrics.json must parse and contain the three required metrics."""
        self.assertTrue(os.path.isfile(get_metrics_path()), "metrics.json not found")
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        self.assertTrue(metrics.get("success"))
        self.assertEqual(metrics.get("model"), MODEL_NAME)

        values = metrics.get("metrics", {})
        for key in ("r2", "rmse", "mae"):
            self.assertIn(key, values, f"metrics.json missing '{key}'")
            self.assertIsInstance(values[key], (int, float), f"'{key}' must be numeric")
            self.assertGreaterEqual(values[key], 0, f"'{key}' cannot be negative")

        # R-squared for a model that predicts must be in a sane range.
        self.assertLessEqual(values["r2"], 1.0, "R2 cannot exceed 1.0")
        self.assertGreater(values["r2"], 0.5, "R2 below 0.5 suggests the model did not learn")


class TestPredictionApi(unittest.TestCase):
    """POST /api/regression/predict must return real model predictions."""

    def setUp(self):
        self.client = app.test_client()

    def test_valid_prediction_success(self):
        """A valid payload returns 200 with a positive numeric prediction."""
        response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertEqual(data["unit"], "INR")
        self.assertIsInstance(data["prediction"], (int, float))
        self.assertGreater(data["prediction"], 0, "A property price must be positive")

    def test_prediction_response_format(self):
        """The success response must carry every documented field."""
        response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)
        data = response.get_json()

        for key in ("success", "model", "prediction", "unit"):
            self.assertIn(key, data, f"Response is missing required key '{key}'")

        # Optional provenance fields the UI displays.
        for key in ("algorithm", "inputs", "trained_at", "r2", "rmse", "mae"):
            self.assertIn(key, data, f"Response is missing provenance key '{key}'")

        # The echoed inputs must match what was sent.
        self.assertEqual(data["inputs"]["area"], VALID_PAYLOAD["area"])
        self.assertEqual(data["inputs"]["bedrooms"], VALID_PAYLOAD["bedrooms"])

    def test_prediction_comes_from_the_model_not_a_constant(self):
        """
        Two different properties must produce two different predictions.
        A hard-coded or faked response would return the same value for both.
        """
        small = dict(VALID_PAYLOAD, area=900, bedrooms=2, bathrooms=1)
        large = dict(VALID_PAYLOAD, area=4000, bedrooms=5, bathrooms=4)

        small_res = self.client.post('/api/regression/predict', json=small).get_json()
        large_res = self.client.post('/api/regression/predict', json=large).get_json()

        self.assertTrue(small_res["success"], small_res)
        self.assertTrue(large_res["success"], large_res)

        self.assertNotEqual(
            small_res["prediction"], large_res["prediction"],
            "Different properties produced an identical price — prediction looks hard-coded"
        )
        # A 4000 sq.ft property must be valued above a 900 sq.ft property.
        self.assertGreater(
            large_res["prediction"], small_res["prediction"],
            "The larger property was not valued above the smaller one"
        )

    def test_prediction_is_reproducible(self):
        """
        The same input must yield the same prediction because predict.R loads a
        fixed saved model rather than retraining on each request.
        """
        first = self.client.post('/api/regression/predict', json=VALID_PAYLOAD).get_json()
        second = self.client.post('/api/regression/predict', json=VALID_PAYLOAD).get_json()

        self.assertEqual(
            first["prediction"], second["prediction"],
            "Identical input produced different predictions — the model is not being reused"
        )

    def test_prediction_surfaces_real_training_metrics(self):
        """The prediction response must carry the real R2/RMSE/MAE from training."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            trained = json.load(handle)["metrics"]

        data = self.client.post('/api/regression/predict', json=VALID_PAYLOAD).get_json()

        self.assertAlmostEqual(data["r2"], trained["r2"], places=4)
        self.assertAlmostEqual(data["rmse"], round(trained["rmse"], 2), places=2)
        self.assertAlmostEqual(data["mae"], round(trained["mae"], 2), places=2)


class TestPredictionValidation(unittest.TestCase):
    """Invalid and missing input must be rejected with clean 400 errors."""

    def setUp(self):
        self.client = app.test_client()

    def assert_rejected(self, payload, expected_fragment):
        response = self.client.post('/api/regression/predict', json=payload)
        self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("errors", data)
        self.assertGreater(len(data["errors"]), 0)
        self.assertIn(
            expected_fragment, " ".join(data["errors"]).lower(),
            f"Error message did not mention '{expected_fragment}'"
        )
        return data

    def test_negative_area_rejected(self):
        """area must be greater than 0."""
        self.assert_rejected(dict(VALID_PAYLOAD, area=-100), "area")

    def test_zero_area_rejected(self):
        """area of exactly 0 is not a real property."""
        self.assert_rejected(dict(VALID_PAYLOAD, area=0), "area")

    def test_zero_bedrooms_rejected(self):
        """bedrooms must be greater than 0."""
        self.assert_rejected(dict(VALID_PAYLOAD, bedrooms=0), "bedrooms")

    def test_zero_bathrooms_rejected(self):
        """bathrooms must be greater than 0."""
        self.assert_rejected(dict(VALID_PAYLOAD, bathrooms=0), "bathrooms")

    def test_negative_property_age_rejected(self):
        """property_age must be 0 or greater."""
        self.assert_rejected(dict(VALID_PAYLOAD, property_age=-5), "property age")

    def test_location_score_out_of_range_rejected(self):
        """location_score must sit inside the range the model was trained on."""
        self.assert_rejected(dict(VALID_PAYLOAD, location_score=99), "location score")

    def test_area_out_of_range_rejected(self):
        """An area far outside the training distribution must be rejected."""
        self.assert_rejected(dict(VALID_PAYLOAD, area=99999), "area")

    def test_non_numeric_value_rejected(self):
        """A non-numeric feature value must be rejected."""
        self.assert_rejected(dict(VALID_PAYLOAD, area="not-a-number"), "area")

    def test_missing_fields_rejected(self):
        """A partial payload must report every missing field."""
        data = self.assert_rejected({"area": 1500}, "required")
        self.assertGreaterEqual(len(data["errors"]), 4, "All four missing fields should be reported")

    def test_empty_body_rejected(self):
        """An empty JSON object must be rejected."""
        self.assert_rejected({}, "required")

    def test_unknown_field_rejected(self):
        """A typo'd field name must not be silently ignored."""
        data = self.assert_rejected(dict(VALID_PAYLOAD, bedroom=3), "unrecognised")
        self.assertIn("bedroom", " ".join(data["errors"]))

    def test_non_json_request_rejected(self):
        """A request without a JSON content type must return 415."""
        response = self.client.post('/api/regression/predict', data="area=1500")
        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.get_json()["success"])

    def test_validate_property_input_unit_level(self):
        """The validation helper returns cleaned floats and error messages."""
        cleaned, errors = validate_property_input(VALID_PAYLOAD)
        self.assertEqual(errors, [])
        self.assertEqual(set(cleaned.keys()), set(FEATURES))
        for value in cleaned.values():
            self.assertIsInstance(value, float)

        _, errors = validate_property_input({"area": 1500})
        self.assertTrue(errors)
        self.assertEqual(len(errors), 4, "One error per missing field")

        _, errors = validate_property_input("not a dict")
        self.assertEqual(len(errors), 1)


class TestRExecutionFailureHandling(unittest.TestCase):
    """Failures inside the R engine must surface as clean API errors, not crashes."""

    def setUp(self):
        self.client = app.test_client()

    def test_r_execution_failure_returns_502(self):
        """When RRunner reports failure, the API returns 502 with a message."""
        failure = {
            "success": False,
            "message": "R script execution failed with exit code 1",
            "error": "Error in predict(): object 'model' not found",
            "return_code": 1
        }
        with patch("backend.services.regression_service.r_runner.execute_script", return_value=failure):
            response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("model' not found", data["error"])

    def test_r_unparseable_output_returns_502(self):
        """Non-JSON stdout from R must produce a clean 502, not a crash."""
        garbage = {
            "success": True,
            "data": None,          # runner could not parse a JSON object
            "stdout": "not json at all",
            "return_code": 0
        }
        with patch("backend.services.regression_service.r_runner.execute_script", return_value=garbage):
            response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.get_json()["success"])

    def test_r_returns_non_numeric_prediction_returns_502(self):
        """A non-numeric prediction from R must be rejected defensively."""
        bad_prediction = {
            "success": True,
            "data": {"success": True, "prediction": "seven lakhs", "unit": "INR"},
            "return_code": 0
        }
        with patch("backend.services.regression_service.r_runner.execute_script", return_value=bad_prediction):
            response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertIn("non-numeric", response.get_json()["error"])

    def test_missing_model_file_returns_503(self):
        """An untrained model must return 503 with retraining instructions."""
        # The guard lives inside predict_price, and app.py puts backend/ on
        # sys.path, so the routes hold this service as 'services.regression_service'.
        with patch("services.regression_service.is_model_trained", return_value=False):
            response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train.R", data["error"], "Should tell the user how to fix it")

    def test_r_script_reports_input_rejection(self):
        """When R itself rejects the input, the API returns 400 with R's message."""
        r_rejection = {
            "success": True,     # the R process exited cleanly
            "return_code": 0,
            "data": {
                "success": False,
                "error": "Field 'area' must be between 669.7 and 4190.5 (model training range).",
                "errors": ["Field 'area' must be between 669.7 and 4190.5 (model training range)."]
            }
        }
        with patch("backend.services.regression_service.r_runner.execute_script", return_value=r_rejection):
            response = self.client.post('/api/regression/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])


class TestMetricsAndEvaluationApi(unittest.TestCase):
    """Metrics, evaluation and schema endpoints must serve the real artifacts."""

    def setUp(self):
        self.client = app.test_client()

    def test_metrics_endpoint(self):
        """GET /api/regression/metrics returns the real R2/RMSE/MAE."""
        response = self.client.get('/api/regression/metrics')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)

        metrics = data["metrics"]
        for key in ("r2", "rmse", "mae"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            self.assertGreater(metrics[key], 0)

        # The endpoint must agree with the artifact on disk.
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)["metrics"]
        self.assertEqual(metrics["r2"], on_disk["r2"])
        self.assertEqual(metrics["rmse"], on_disk["rmse"])
        self.assertEqual(metrics["mae"], on_disk["mae"])

    def test_metrics_endpoint_reports_dataset_provenance(self):
        """Metrics must describe the real training split."""
        data = self.client.get('/api/regression/metrics').get_json()
        dataset = data["dataset"]

        self.assertEqual(dataset["target"], "price")
        self.assertEqual(set(dataset["features"]), set(FEATURES))
        self.assertGreater(dataset["rows_train"], 0)
        self.assertGreater(dataset["rows_test"], 0)
        self.assertEqual(
            dataset["rows_train"] + dataset["rows_test"],
            dataset["rows_clean"],
            "Train + test rows must equal the cleaned dataset size"
        )

    def test_metrics_include_model_coefficients(self):
        """Coefficients from the fitted model must be exposed."""
        data = self.client.get('/api/regression/metrics').get_json()
        coefficients = data["coefficients"]

        self.assertGreaterEqual(len(coefficients), len(FEATURES) + 1, "intercept + one per feature")

        terms = {c["term"] for c in coefficients}
        self.assertIn("(Intercept)", terms)
        for feature in FEATURES:
            self.assertIn(feature, terms)

        for coefficient in coefficients:
            self.assertIsInstance(coefficient["estimate"], (int, float))

    def test_metrics_missing_artifact_returns_503(self):
        """A missing metrics artifact returns 503 with retraining instructions."""
        # app.py puts backend/ on sys.path, so the routes import this service as
        # 'services.regression_service' — a different module object than the
        # 'backend.services.regression_service' alias used by the test imports.
        with patch("services.regression_service.get_metrics_path",
                   return_value="/nonexistent/metrics.json"):
            response = self.client.get('/api/regression/metrics')

        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train.R", data["error"])

    def test_evaluation_endpoint_returns_real_pairs(self):
        """GET /api/regression/evaluation returns actual vs predicted test data."""
        response = self.client.get('/api/regression/evaluation')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["unit"], "INR")
        self.assertGreater(data["count"], 0)
        self.assertEqual(len(data["points"]), data["count"])

        for point in data["points"]:
            for key in ("actual", "predicted", "residual"):
                self.assertIn(key, point)
                self.assertIsInstance(point[key], (int, float))
            # The residual must be the actual minus the predicted value.
            self.assertAlmostEqual(
                point["residual"], round(point["actual"] - point["predicted"], 2), places=2
            )
            self.assertGreater(point["actual"], 0)
            self.assertGreater(point["predicted"], 0)

    def test_evaluation_points_are_distinct_records(self):
        """The chart must not be fed a duplicated single point."""
        data = get_evaluation_data()
        actuals = [point["actual"] for point in data["points"]]
        self.assertGreater(len(set(actuals)), 1, "Evaluation points are all identical")

    def test_schema_endpoint_reports_training_ranges(self):
        """GET /api/regression/schema exposes the real dataset ranges."""
        response = self.client.get('/api/regression/schema')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["model_trained"])

        features = {f["name"]: f for f in data["features"]}
        self.assertEqual(set(features.keys()), set(FEATURES))

        # Ranges must come from the data: location_score spans roughly 3 to 10.
        location = features["location_score"]
        self.assertLess(location["min"], 5)
        self.assertGreater(location["max"], 9)
        self.assertLess(location["min"], location["max"])

    def test_frontend_page_and_assets_exist(self):
        """The Phase 3 page and its assets must be present."""
        for relative in (
            os.path.join("frontend", "regression.html"),
            os.path.join("frontend", "css", "regression.css"),
            os.path.join("frontend", "js", "regression.js"),
        ):
            path = os.path.join(PROJECT_ROOT, relative)
            self.assertTrue(os.path.isfile(path), f"Missing frontend file: {relative}")
            self.assertGreater(os.path.getsize(path), 0, f"Empty frontend file: {relative}")

    def test_dashboard_links_to_regression_page(self):
        """The dashboard's Linear Regression card must open the new page."""
        index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
        with open(index_path, "r", encoding="utf-8") as handle:
            html = handle.read()

        self.assertIn('href="regression.html"', html, "Dashboard does not link to regression.html")
        self.assertIn("OPEN MODEL", html, "Dashboard is missing the OPEN MODEL button")
        self.assertNotIn("Coming in Phase 3", html, "Stale 'Coming in Phase 3' text remains")


class TestPhase2Preservation(unittest.TestCase):
    """Phase 1 and Phase 2 endpoints must keep working after Phase 3."""

    def setUp(self):
        self.client = app.test_client()

    def test_phase1_health_still_works(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "online")

    def test_phase2_r_engine_endpoints_still_work(self):
        for path in ('/api/r-engine/status', '/api/datasets', '/api/datasets/validate'):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"{path} regressed")

    def test_root_endpoint_lists_regression_endpoints(self):
        data = self.client.get('/').get_json()
        self.assertIn("regression_predict", data["endpoints"])
        self.assertIn("Phase 3", data["phase"])


if __name__ == '__main__':
    unittest.main()
