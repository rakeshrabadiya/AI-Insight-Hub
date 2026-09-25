"""
AI Insight Hub — Phase 4 Decision Tree Test Suite

Tests real behaviour end to end:
  1. Training artifacts exist (train.R, predict.R, model.rds, metrics.json)
  2. The metrics are real numbers produced by the trained R model
  3. Valid predictions return a genuine risk class from the saved model
  4. Predictions actually change when an input changes (proves no hard-coding)
  5. Predictions are reproducible across identical requests (saved-model reuse)
  6. Invalid input is rejected with clean 400 errors
  7. Missing input is rejected
  8. The prediction response has the documented shape
  9. R execution failures are handled without crashing the API
 10. The metrics endpoint serves real artifacts
 11. The tree endpoint serves the real fitted tree
 12. All three risk classes are reachable and valid

Every test exercises the real R engine or the real saved artifacts. No test
asserts a hard-coded metric value or a stubbed success value.
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
from backend.services import decision_tree_service
from backend.services.decision_tree_service import (
    CLASSES,
    FEATURES,
    MODEL_NAME,
    get_metrics_path,
    get_model_path,
    get_training_metrics,
    get_tree_path,
    get_tree_structure,
    is_model_trained,
    validate_applicant_input,
)

# A valid applicant inside the dataset's observed ranges
VALID_PAYLOAD = {
    "age": 32,
    "income": 95000,
    "credit_score": 735,
    "existing_loans": 1,
    "employment_years": 5
}

# Applicants chosen from the model's real decision regions so that the tree
# routes each of them to a different terminal class.
LOW_PAYLOAD = {
    "age": 41, "income": 78675, "credit_score": 666,
    "existing_loans": 1, "employment_years": 24
}
MEDIUM_PAYLOAD = {
    "age": 32, "income": 95000, "credit_score": 735,
    "existing_loans": 1, "employment_years": 5
}
HIGH_PAYLOAD = {
    "age": 24, "income": 28361, "credit_score": 412,
    "existing_loans": 5, "employment_years": 11
}


class TestTrainingArtifacts(unittest.TestCase):
    """Phase 4 training scripts and the trained model must exist on disk."""

    def test_training_and_prediction_scripts_exist(self):
        """train.R and predict.R must be present in r_models/decision_tree/."""
        for script in ("train.R", "predict.R"):
            path = os.path.join(PROJECT_ROOT, "r_models", "decision_tree", script)
            self.assertTrue(os.path.isfile(path), f"{script} is missing at {path}")
            self.assertGreater(os.path.getsize(path), 0, f"{script} is empty")

    def test_trained_model_file_exists(self):
        """model.rds must exist and be non-empty after training."""
        model_path = get_model_path()
        self.assertTrue(os.path.isfile(model_path), "model.rds not found — run train.R")
        self.assertGreater(os.path.getsize(model_path), 0, "model.rds is empty")
        self.assertTrue(is_model_trained())

    def test_training_script_uses_rpart(self):
        """train.R must fit a real rpart classification model."""
        with open(os.path.join(PROJECT_ROOT, "r_models", "decision_tree", "train.R"),
                  "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("rpart::rpart(", source, "train.R must call rpart::rpart()")
        self.assertIn('method = "class"', source, "train.R must fit a classification tree")
        self.assertIn("predict(", source, "train.R must call predict() to score the test set")

    def test_training_script_does_not_hardcode_metrics(self):
        """train.R must compute metrics, never assign a literal accuracy/F1."""
        with open(os.path.join(PROJECT_ROOT, "r_models", "decision_tree", "train.R"),
                  "r", encoding="utf-8") as handle:
            source = handle.read()

        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            for literal in (f"{metric} <- 0.", f"{metric} <- 1.", f"{metric} <- 9"):
                self.assertNotIn(
                    literal, source,
                    f"train.R appears to hard-code {metric} via '{literal}'"
                )

    def test_prediction_script_loads_saved_model(self):
        """predict.R must load model.rds and must not retrain the model."""
        with open(os.path.join(PROJECT_ROOT, "r_models", "decision_tree", "predict.R"),
                  "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("readRDS", source, "predict.R must load the saved model via readRDS()")
        self.assertIn("model.rds", source, "predict.R must reference model.rds")
        self.assertNotIn("rpart::rpart(",
                         source, "predict.R must NOT retrain the model with rpart::rpart()")

    def test_metrics_artifact_is_real_json(self):
        """metrics.json must parse and contain the four required metrics."""
        self.assertTrue(os.path.isfile(get_metrics_path()), "metrics.json not found")
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        self.assertTrue(metrics.get("success"))
        self.assertEqual(metrics.get("model"), MODEL_NAME)
        self.assertEqual(metrics.get("classes"), list(CLASSES))

        values = metrics.get("metrics", {})
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(key, values, f"metrics.json missing '{key}'")
            self.assertIsInstance(values[key], (int, float), f"'{key}' must be numeric")
            self.assertGreaterEqual(values[key], 0, f"'{key}' cannot be negative")
            self.assertLessEqual(values[key], 1, f"'{key}' cannot exceed 1")

        # A model that learned something must beat chance on both scores.
        self.assertGreater(values["accuracy"], 0.5, "Accuracy below 0.5 suggests no learning")
        self.assertGreater(values["f1_score"], 0.2, "F1 below 0.2 suggests no learning")

    def test_metrics_confusion_matrix_is_square_and_consistent(self):
        """The confusion matrix must be square and agree with the headline metrics."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        matrix_block = metrics.get("confusion_matrix", {})
        labels = matrix_block.get("labels")
        matrix = matrix_block.get("matrix")

        self.assertEqual(labels, list(CLASSES), "Confusion matrix labels must be the 3 classes")
        self.assertEqual(len(matrix), len(CLASSES), "Matrix must have one row per class")
        for row in matrix:
            self.assertEqual(len(row), len(CLASSES), "Matrix must have one column per class")
            for value in row:
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)

        total = sum(sum(row) for row in matrix)
        self.assertEqual(
            total, metrics["metrics"]["total_predictions"],
            "Confusion matrix cells must sum to the reported prediction count"
        )

        # The reported accuracy must be reproducible from the matrix itself.
        correct = sum(matrix[i][i] for i in range(len(CLASSES)))
        self.assertAlmostEqual(
            metrics["metrics"]["accuracy"], round(correct / total, 6), places=4,
            msg="Accuracy must equal trace(matrix) / sum(matrix)"
        )

    def test_tree_artifact_exists_and_is_a_real_tree(self):
        """tree.json must contain a rooted tree that rpart actually grew."""
        self.assertTrue(os.path.isfile(get_tree_path()), "tree.json not found")
        with open(get_tree_path(), "r", encoding="utf-8") as handle:
            tree = json.load(handle)

        self.assertTrue(tree.get("success"))
        self.assertEqual(tree.get("root", {}).get("id"), 1, "The root node must be node 1")
        self.assertFalse(tree["root"]["is_leaf"], "The root must be a decision node")
        self.assertIn("left", tree["root"], "A decision node must have a left branch")
        self.assertIn("right", tree["root"], "A decision node must have a right branch")

        # Node ids must follow rpart's binary layout: 2n and 2n+1.
        def check_node(node):
            if node.get("is_leaf"):
                self.assertIsNone(node.get("left"), "A leaf must not have children")
                return 1
            left = node["left"]
            right = node["right"]
            self.assertEqual(left["id"], node["id"] * 2,
                             "Left child id must be 2 * parent id")
            self.assertEqual(right["id"], node["id"] * 2 + 1,
                             "Right child id must be 2 * parent id + 1")
            return 1 + check_node(left) + check_node(right)

        self.assertEqual(
            check_node(tree["root"]), tree["total_nodes"],
            "The node count in the artifact must match the nodes actually present"
        )

    def test_tree_node_probabilities_are_real(self):
        """Each node's class probabilities must sum to 1 and match its counts."""
        with open(get_tree_path(), "r", encoding="utf-8") as handle:
            tree = json.load(handle)

        def check_node(node):
            probabilities = node.get("class_probabilities", {})
            counts = node.get("class_counts", {})

            for class_name in CLASSES:
                self.assertIn(class_name, probabilities,
                              f"Node {node['id']} is missing a probability for {class_name}")

            total_probability = sum(float(probabilities[c]) for c in CLASSES)
            self.assertAlmostEqual(total_probability, 1.0, places=3,
                                   msg=f"Node {node['id']} probabilities must sum to 1")

            total_count = sum(int(counts[c]) for c in CLASSES)
            self.assertEqual(total_count, node["samples"],
                             f"Node {node['id']} class counts must sum to its sample count")

            if not node.get("is_leaf"):
                check_node(node["left"])
                check_node(node["right"])

        check_node(tree["root"])


class TestPredictionApi(unittest.TestCase):
    """POST /api/decision-tree/predict must return real model predictions."""

    def setUp(self):
        self.client = app.test_client()

    def test_valid_prediction_success(self):
        """A valid payload returns 200 with a recognised risk class."""
        response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertIn(data["prediction"], CLASSES,
                      "Prediction must be one of the trained risk classes")

    def test_prediction_response_format(self):
        """The success response must carry every documented field."""
        response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)
        data = response.get_json()

        for key in ("success", "model", "prediction"):
            self.assertIn(key, data, f"Response is missing required key '{key}'")

        # Provenance and explanation fields the UI renders.
        for key in ("algorithm", "inputs", "trained_at", "class_probabilities",
                    "decision_path", "classes"):
            self.assertIn(key, data, f"Response is missing provenance key '{key}'")

        # The echoed inputs must match what was sent.
        self.assertEqual(data["inputs"]["age"], VALID_PAYLOAD["age"])
        self.assertEqual(data["inputs"]["income"], VALID_PAYLOAD["income"])

    def test_confidence_comes_from_the_model(self):
        """Confidence, when present, must be the model's own class probability."""
        data = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()

        if "confidence" in data:
            self.assertIsInstance(data["confidence"], (int, float))
            self.assertGreaterEqual(data["confidence"], 0.0)
            self.assertLessEqual(data["confidence"], 1.0)

        # The confidence must equal the predicted class's own probability.
        probabilities = data["class_probabilities"]
        self.assertIn(data["prediction"], probabilities)
        self.assertAlmostEqual(
            data["confidence"], probabilities[data["prediction"]], places=3,
            msg="Confidence must be the model's probability for the predicted class"
        )

    def test_class_probabilities_are_complete_and_normalised(self):
        """The model must report a probability for all three risk classes."""
        data = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()
        probabilities = data["class_probabilities"]

        for class_name in CLASSES:
            self.assertIn(class_name, probabilities,
                          f"Missing probability for {class_name}")
            self.assertGreaterEqual(probabilities[class_name], 0.0)
            self.assertLessEqual(probabilities[class_name], 1.0)

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=3)

    def test_decision_path_is_returned_and_traces_the_real_tree(self):
        """
        The decision path must reference real node ids from the fitted tree and
        end at the leaf that produced the prediction.
        """
        with open(get_tree_path(), "r", encoding="utf-8") as handle:
            tree = json.load(handle)
        real_node_ids = set()

        def collect(node):
            real_node_ids.add(node["id"])
            if not node.get("is_leaf"):
                collect(node["left"])
                collect(node["right"])

        collect(tree["root"])

        data = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()
        path = data["decision_path"]

        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 1, "A decision path needs at least one split and a leaf")

        for step in path:
            self.assertIn(step["node"], real_node_ids,
                          f"Decision path references node {step['node']}, "
                          f"which is not part of the fitted tree")
            self.assertIn("condition", step)

        # The final step must be the leaf that produced the prediction.
        self.assertEqual(path[-1]["branch"], "leaf")
        self.assertIn(data["prediction"], path[-1]["condition"])

    def test_all_three_risk_classes_are_reachable(self):
        """
        LOW, MEDIUM and HIGH must all be producible by the real model, proving
        the classifier is not collapsed onto a single class.
        """
        seen = {}
        for name, payload in (
            ("low", LOW_PAYLOAD), ("medium", MEDIUM_PAYLOAD), ("high", HIGH_PAYLOAD)
        ):
            response = self.client.post('/api/decision-tree/predict', json=payload)
            self.assertEqual(response.status_code, 200,
                             f"{name} payload was rejected: {response.get_data(as_text=True)}")
            data = response.get_json()
            self.assertIn(data["prediction"], CLASSES)
            seen[name] = data["prediction"]

        self.assertEqual(sorted(set(seen.values())), sorted(CLASSES),
                         f"Expected all three classes to be reachable, got {seen}")

    def test_prediction_changes_with_input(self):
        """
        Two clearly different applicants must be classified differently.
        A hard-coded response would return the same class for both.
        """
        low = self.client.post('/api/decision-tree/predict', json=LOW_PAYLOAD).get_json()
        high = self.client.post('/api/decision-tree/predict', json=HIGH_PAYLOAD).get_json()

        self.assertTrue(low["success"], low)
        self.assertTrue(high["success"], high)
        self.assertNotEqual(
            low["prediction"], high["prediction"],
            "Different applicants produced an identical class — prediction looks hard-coded"
        )

    def test_prediction_is_reproducible(self):
        """
        The same input must yield the same prediction because predict.R loads a
        fixed saved model rather than retraining on each request.
        """
        first = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()
        second = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()

        self.assertEqual(
            first["prediction"], second["prediction"],
            "Identical input produced different predictions — the model is not being reused"
        )
        self.assertEqual(
            first.get("confidence"), second.get("confidence"),
            "Identical input produced different confidence values"
        )

    def test_prediction_surfaces_real_training_metrics(self):
        """The prediction response must carry the real metrics from training."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            trained = json.load(handle)["metrics"]

        data = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()

        for metric in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(metric, data, f"Prediction response is missing '{metric}'")
            self.assertAlmostEqual(data[metric], round(trained[metric], 4), places=4)


class TestPredictionValidation(unittest.TestCase):
    """Invalid and missing input must be rejected with clean 400 errors."""

    def setUp(self):
        self.client = app.test_client()

    def assert_rejected(self, payload, expected_fragment):
        response = self.client.post('/api/decision-tree/predict', json=payload)
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

    def test_zero_age_rejected(self):
        """age must be greater than 0."""
        self.assert_rejected(dict(VALID_PAYLOAD, age=0), "age")

    def test_negative_income_rejected(self):
        """income must be greater than 0."""
        self.assert_rejected(dict(VALID_PAYLOAD, income=-50000), "income")

    def test_invalid_credit_score_low_rejected(self):
        """A credit score below 300 is not a real score."""
        self.assert_rejected(dict(VALID_PAYLOAD, credit_score=150), "credit score")

    def test_invalid_credit_score_high_rejected(self):
        """A credit score above 900 is not a real score."""
        self.assert_rejected(dict(VALID_PAYLOAD, credit_score=999), "credit score")

    def test_negative_existing_loans_rejected(self):
        """existing_loans cannot be negative."""
        self.assert_rejected(dict(VALID_PAYLOAD, existing_loans=-1), "existing loans")

    def test_negative_employment_years_rejected(self):
        """employment_years cannot be negative."""
        self.assert_rejected(dict(VALID_PAYLOAD, employment_years=-5), "employment years")

    def test_documented_example_payload_is_accepted(self):
        """
        The example payload published in the API documentation must work.

        Its income of 750,000 sits far above the training range, so the request
        must be answered rather than rejected — the model still classifies it,
        and the response flags the extrapolation instead of refusing.
        """
        response = self.client.post('/api/decision-tree/predict', json={
            "age": 32, "income": 750000, "credit_score": 735,
            "existing_loans": 1, "employment_years": 5
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], CLASSES)

        # The extrapolation must be reported, not hidden.
        warned_fields = {w["field"] for w in data.get("warnings", [])}
        self.assertIn("income", warned_fields,
                      "An out-of-range income must produce an extrapolation warning")

    def test_credit_score_boundaries_accepted(self):
        """Both ends of the documented 300-900 range are valid credit scores."""
        for score in (300, 900):
            response = self.client.post(
                '/api/decision-tree/predict', json=dict(VALID_PAYLOAD, credit_score=score)
            )
            self.assertEqual(response.status_code, 200,
                             f"credit_score={score} was rejected: {response.get_data(as_text=True)}")

    def test_in_range_input_has_no_extrapolation_warning(self):
        """A normal applicant must not be told the prediction was extrapolated."""
        data = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD).get_json()
        out_of_range = [w for w in data.get("warnings", []) if w["field"] in FEATURES]
        self.assertEqual(out_of_range, [],
                         f"In-range input produced warnings: {out_of_range}")

    def test_non_numeric_value_rejected(self):
        """A non-numeric feature value must be rejected."""
        self.assert_rejected(dict(VALID_PAYLOAD, credit_score="excellent"), "credit score")

    def test_missing_fields_rejected(self):
        """A partial payload must report every missing field."""
        data = self.assert_rejected({"age": 32}, "required")
        self.assertGreaterEqual(len(data["errors"]), 4,
                                "All four missing fields should be reported")

    def test_empty_body_rejected(self):
        """An empty JSON object must be rejected."""
        self.assert_rejected({}, "required")

    def test_unknown_field_rejected(self):
        """A typo'd field name must not be silently ignored."""
        data = self.assert_rejected(dict(VALID_PAYLOAD, credit_scor=700), "unrecognised")
        self.assertIn("credit_scor", " ".join(data["errors"]))

    def test_non_json_request_rejected(self):
        """A request without a JSON content type must return 415."""
        response = self.client.post('/api/decision-tree/predict', data="age=32")
        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.get_json()["success"])

    def test_malformed_json_rejected(self):
        """A body that claims to be JSON but is not must return 400."""
        response = self.client.post(
            '/api/decision-tree/predict',
            data="{not valid json",
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_validate_applicant_input_unit_level(self):
        """The validation helper returns cleaned floats and error messages."""
        cleaned, errors = validate_applicant_input(VALID_PAYLOAD)
        self.assertEqual(errors, [])
        self.assertEqual(set(cleaned.keys()), set(FEATURES))
        for value in cleaned.values():
            self.assertIsInstance(value, float)

        _, errors = validate_applicant_input({"age": 32})
        self.assertTrue(errors)
        self.assertEqual(len(errors), 4, "One error per missing field")

        _, errors = validate_applicant_input("not a dict")
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
        with patch("backend.services.decision_tree_service.r_runner.execute_script",
                   return_value=failure):
            response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)

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
        with patch("backend.services.decision_tree_service.r_runner.execute_script",
                   return_value=garbage):
            response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.get_json()["success"])

    def test_r_returns_unknown_risk_class_returns_502(self):
        """
        A risk class the model was never trained on must be rejected rather
        than passed through to the user.
        """
        bad_prediction = {
            "success": True,
            "data": {"success": True, "prediction": "EXTREME", "model": MODEL_NAME},
            "return_code": 0
        }
        with patch("backend.services.decision_tree_service.r_runner.execute_script",
                   return_value=bad_prediction):
            response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertIn("unexpected risk class", response.get_json()["error"].lower())

    def test_missing_model_file_returns_503(self):
        """An untrained model must return 503 with retraining instructions."""
        # app.py puts backend/ on sys.path, so the routes hold this service as
        # 'services.decision_tree_service' — a different module object than the
        # 'backend.services.decision_tree_service' alias used above.
        with patch("services.decision_tree_service.is_model_trained", return_value=False):
            response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)

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
                "error": "Field 'credit_score' must be between 300 and 850.",
                "errors": ["Field 'credit_score' must be between 300 and 850."]
            }
        }
        with patch("backend.services.decision_tree_service.r_runner.execute_script",
                   return_value=r_rejection):
            response = self.client.post('/api/decision-tree/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])


class TestMetricsAndTreeApi(unittest.TestCase):
    """Metrics, tree and schema endpoints must serve the real artifacts."""

    def setUp(self):
        self.client = app.test_client()

    def test_metrics_endpoint(self):
        """GET /api/decision-tree/metrics returns the real classification metrics."""
        response = self.client.get('/api/decision-tree/metrics')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)

        metrics = data["metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            self.assertGreater(metrics[key], 0)

        # The endpoint must agree with the artifact on disk.
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)["metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertEqual(metrics[key], on_disk[key])

    def test_metrics_endpoint_reports_dataset_provenance(self):
        """Metrics must describe the real training split."""
        data = self.client.get('/api/decision-tree/metrics').get_json()
        dataset = data["dataset"]

        self.assertEqual(dataset["target"], "risk")
        self.assertEqual(set(dataset["features"]), set(FEATURES))
        self.assertGreater(dataset["rows_train"], 0)
        self.assertGreater(dataset["rows_test"], 0)
        self.assertEqual(
            dataset["rows_train"] + dataset["rows_test"], dataset["rows_clean"],
            "Train + test rows must equal the cleaned dataset size"
        )

    def test_metrics_include_per_class_and_confusion_matrix(self):
        """Per-class metrics and the confusion matrix must be exposed."""
        data = self.client.get('/api/decision-tree/metrics').get_json()

        per_class = data["per_class_metrics"]
        for class_name in CLASSES:
            self.assertIn(class_name, per_class)
            self.assertIn("precision", per_class[class_name])
            self.assertIn("recall", per_class[class_name])
            self.assertIn("f1_score", per_class[class_name])

        self.assertEqual(data["confusion_matrix"]["labels"], list(CLASSES))

    def test_metrics_missing_artifact_returns_503(self):
        """A missing metrics artifact returns 503 with retraining instructions."""
        with patch("services.decision_tree_service.get_metrics_path",
                   return_value="/nonexistent/metrics.json"):
            response = self.client.get('/api/decision-tree/metrics')

        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train.R", data["error"])

    def test_tree_endpoint_returns_real_tree(self):
        """GET /api/decision-tree/tree returns the actual rpart tree."""
        response = self.client.get('/api/decision-tree/tree')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertEqual(data["target"], "risk")

        root = data["root"]
        self.assertEqual(root["id"], 1)
        self.assertFalse(root["is_leaf"], "The root node must be a decision node")
        self.assertIn(root["variable"], FEATURES,
                      "The root must split on one of the model's features")
        self.assertIsInstance(root["cut"], (int, float))
        self.assertIn("left_condition", root)
        self.assertIn("right_condition", root)

        # The reported counts must match what is actually present.
        def count(node):
            if node.get("is_leaf"):
                return 1
            return 1 + count(node["left"]) + count(node["right"])

        self.assertEqual(count(root), data["total_nodes"])

    def test_tree_endpoint_splits_are_ordered_conditions(self):
        """
        Each decision node must carry two complementary conditions on the same
        feature, which is what makes the branch labels meaningful.
        """
        data = get_tree_structure()
        root = data["root"]

        self.assertIn("<", root["left_condition"] + root["right_condition"])
        self.assertIn(">=", root["left_condition"] + root["right_condition"])
        self.assertIn(root["variable"], root["left_condition"])
        self.assertIn(root["variable"], root["right_condition"])

    def test_tree_endpoint_reports_variable_importance(self):
        """The endpoint must expose the real Gini-gain importance values."""
        data = self.client.get('/api/decision-tree/tree').get_json()
        importance = data["variable_importance"]

        self.assertGreaterEqual(len(importance), 1)
        for entry in importance:
            self.assertIn("feature", entry)
            self.assertIn("importance", entry)
            self.assertIsInstance(entry["importance"], (int, float))
            self.assertGreater(entry["importance"], 0)

        # The most important feature must actually split the root, otherwise
        # the numbers and the drawn tree would disagree.
        if importance:
            self.assertEqual(importance[0]["feature"], data["root"]["variable"])

    def test_tree_missing_artifact_returns_503(self):
        """A missing tree artifact returns 503 with retraining instructions."""
        with patch("services.decision_tree_service.get_tree_path",
                   return_value="/nonexistent/tree.json"):
            response = self.client.get('/api/decision-tree/tree')

        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train.R", data["error"])

    def test_schema_endpoint_reports_training_ranges(self):
        """GET /api/decision-tree/schema exposes the real dataset ranges."""
        response = self.client.get('/api/decision-tree/schema')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["model_trained"])
        self.assertEqual(data["classes"], list(CLASSES))

        features = {f["name"]: f for f in data["features"]}
        self.assertEqual(set(features.keys()), set(FEATURES))

        # Ranges must come from the data, not from a hand-written constant.
        credit = features["credit_score"]
        self.assertGreaterEqual(credit["min"], 300)
        self.assertLessEqual(credit["max"], 850)
        self.assertLess(credit["min"], credit["max"])

    def test_frontend_page_and_assets_exist(self):
        """The Phase 4 page and its assets must be present."""
        for relative in (
            os.path.join("frontend", "decision-tree.html"),
            os.path.join("frontend", "css", "decision-tree.css"),
            os.path.join("frontend", "js", "decision-tree.js"),
        ):
            path = os.path.join(PROJECT_ROOT, relative)
            self.assertTrue(os.path.isfile(path), f"Missing frontend file: {relative}")
            self.assertGreater(os.path.getsize(path), 0, f"Empty frontend file: {relative}")

    def test_dashboard_marks_decision_tree_active(self):
        """The dashboard's Decision Tree card must be ACTIVE and open the page."""
        index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
        with open(index_path, "r", encoding="utf-8") as handle:
            html = handle.read()

        self.assertIn('href="decision-tree.html"', html,
                      "Dashboard does not link to decision-tree.html")
        self.assertNotIn("Coming in Phase 4", html,
                         "Stale 'Coming in Phase 4' text remains on the Decision Tree card")

        # The Linear Regression card must still be intact.
        self.assertIn('href="regression.html"', html,
                      "Dashboard no longer links to the Phase 3 regression page")


class TestPhasePreservation(unittest.TestCase):
    """Phase 1-3 endpoints must keep working after Phase 4."""

    def setUp(self):
        self.client = app.test_client()

    def test_phase1_health_still_works(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "online")

    def test_phase2_endpoints_still_work(self):
        for path in ('/api/r-engine/status', '/api/datasets', '/api/datasets/validate'):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"{path} regressed")

    def test_phase3_regression_prediction_still_works(self):
        """The Phase 3 Linear Regression endpoint must be unaffected."""
        response = self.client.post('/api/regression/predict', json={
            "area": 1500, "bedrooms": 3, "bathrooms": 2,
            "location_score": 8, "property_age": 5
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], "Linear Regression")
        self.assertIsInstance(data["prediction"], (int, float))

    def test_phase3_regression_metrics_still_work(self):
        response = self.client.get('/api/regression/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])

    def test_root_endpoint_lists_both_models(self):
        data = self.client.get('/').get_json()
        endpoints = data["endpoints"]

        for key in ("regression_predict", "decision_tree_predict",
                    "decision_tree_metrics", "decision_tree_tree"):
            self.assertIn(key, endpoints)
        self.assertIn("Phase 3", data["phase"])
        self.assertIn("Phase 4", data["phase"])


if __name__ == '__main__':
    unittest.main()
