"""
AI Insight Hub — Phase 5 KNN Test Suite

Tests real behaviour end to end:
  1. Dataset validation (the real students.csv passes the shared validator)
  2. Training scripts exist on disk
  3. model.rds exists and is non-empty
  4. metrics.json exists and parses
  5. Valid predictions return a genuine performance tier from the saved model
  6. Predictions are one of LOW / MEDIUM / HIGH
  7. Selected K is a positive integer that the K grid actually evaluated
  8. Metrics are internally consistent (confusion matrix reproduces accuracy)
  9. Out-of-range but logically valid input is warned about, not rejected
 10. Invalid values are rejected with clean 400 errors
 11. Missing fields are rejected
 12. Predictions respond to input (nothing is hard-coded)
 13. Predictions are reproducible (saved-model reuse, no retraining per request)
 14. The prediction response has the documented shape
 15. Nearest-neighbour data is real, consistent with the vote, and sorted
 16. The metrics, config and schema endpoints serve the real artifacts
 17. R execution failures are handled without crashing the API
 18. Phases 1-4 still work

Every test exercises the real R engine or the real saved artifacts. No test
asserts a hard-coded metric value or a stubbed success value.

A note on class reachability: students.csv holds exactly one LOW row, so at the
selected K a majority vote can never be won by LOW (that would need more LOW
rows in total than exist). Tests therefore assert the prediction is a *valid
class*, not that all three classes are reachable — asserting reachability would
assert something mathematically false about this dataset.
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
from backend.services.knn_service import (
    CLASSES,
    FEATURES,
    MODEL_NAME,
    get_metrics_path,
    get_model_path,
    is_model_trained,
    validate_student_input,
)
from backend.services.dataset_validator import DATASET_SCHEMAS, validate_dataset

# A valid student inside the dataset's observed ranges (the documented example)
VALID_PAYLOAD = {
    "study_hours": 6.5,
    "attendance": 88,
    "previous_score": 74,
    "assignments_completed": 9,
    "practical_score": 81
}


class TestDataset(unittest.TestCase):
    """The existing students.csv must still satisfy the shared dataset schema."""

    def test_students_dataset_is_valid(self):
        """students.csv passes the Phase 2 shared dataset validator."""
        result = validate_dataset("students.csv")

        self.assertTrue(result["valid"], f"students.csv is invalid: {result['errors']}")
        self.assertGreater(result["row_count"], 0, "students.csv has no data rows")

    def test_students_dataset_has_the_documented_columns(self):
        """The dataset carries the five features plus the performance target."""
        result = validate_dataset("students.csv")
        columns = set(result["columns"])

        for column in FEATURES:
            self.assertIn(column, columns, f"Missing feature column '{column}'")
        self.assertIn("performance", columns)

    def test_dataset_matches_the_service_feature_contract(self):
        """The service's FEATURES tuple matches the dataset's own schema."""
        schema = DATASET_SCHEMAS["students.csv"]
        self.assertEqual(set(schema["required_columns"]), set(FEATURES) | {"performance"})
        self.assertEqual(schema["target_column"], "performance")
        self.assertEqual(sorted(schema["categorical_rules"]["performance"]), sorted(CLASSES))


class TestTrainingArtifacts(unittest.TestCase):
    """Phase 5 training scripts and the trained model must exist on disk."""

    def test_training_and_prediction_scripts_exist(self):
        """train.R and predict.R must be present in r_models/knn/."""
        for script in ("train.R", "predict.R"):
            path = os.path.join(PROJECT_ROOT, "r_models", "knn", script)
            self.assertTrue(os.path.isfile(path), f"{script} is missing at {path}")
            self.assertGreater(os.path.getsize(path), 0, f"{script} is empty")

    def test_trained_model_file_exists(self):
        """model.rds must exist and be non-empty after training."""
        model_path = get_model_path()
        self.assertTrue(os.path.isfile(model_path),
                        "model.rds not found — run 'Rscript r_models/knn/train.R'")
        self.assertGreater(os.path.getsize(model_path), 0, "model.rds is empty")
        self.assertTrue(is_model_trained())

    def test_training_script_uses_class_knn(self):
        """train.R must fit a real classifier with class::knn()."""
        path = os.path.join(PROJECT_ROOT, "r_models", "knn", "train.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("class::knn(", source, "train.R must call class::knn()")
        self.assertIn("K_GRID", source, "train.R must evaluate a grid of K values")
        self.assertIn("scaler_center", source, "train.R must scale / normalise the features")

    def test_training_script_does_not_hardcode_metrics(self):
        """train.R must compute metrics, never assign a literal accuracy/F1."""
        path = os.path.join(PROJECT_ROOT, "r_models", "knn", "train.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            for literal in (f"{metric} <- 0.", f"{metric} <- 1.", f"{metric} <- 9"):
                self.assertNotIn(
                    literal, source,
                    f"train.R appears to hard-code {metric} via '{literal}'"
                )

    def test_prediction_script_loads_saved_model(self):
        """predict.R must load model.rds and must not retrain the model."""
        path = os.path.join(PROJECT_ROOT, "r_models", "knn", "predict.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("readRDS", source, "predict.R must load the saved model via readRDS()")
        self.assertIn("model.rds", source, "predict.R must reference model.rds")
        self.assertIn("class::knn(", source, "predict.R must run the real classifier")
        self.assertIn("scaler", source, "predict.R must apply the saved feature scaling")

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

        # A model that learned something must beat chance.
        self.assertGreater(values["accuracy"], 0.5, "Accuracy below 0.5 suggests no learning")

    def test_metrics_confusion_matrix_is_square_and_consistent(self):
        """The confusion matrix must be square and agree with the headline metrics."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        block = metrics.get("confusion_matrix", {})
        labels = block.get("labels")
        matrix = block.get("matrix")

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

    def test_selected_k_is_valid_and_was_actually_evaluated(self):
        """The shipped K must be a positive integer drawn from the K grid."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        selected_k = metrics.get("selected_k")
        self.assertIsInstance(selected_k, int, "selected_k must be an integer")
        self.assertGreater(selected_k, 0, "selected_k must be positive")
        self.assertLessEqual(selected_k, metrics["dataset"]["rows_train"],
                             "K cannot exceed the number of training rows")

        comparison = metrics.get("k_comparison", [])
        self.assertGreaterEqual(len(comparison), 2, "K must be chosen from a comparison, not guessed")

        evaluated = {entry["k"] for entry in comparison}
        self.assertIn(selected_k, evaluated, "selected_k was not one of the evaluated values")

        # Exactly one candidate may be marked as selected.
        selected_flags = [entry for entry in comparison if entry.get("selected")]
        self.assertEqual(len(selected_flags), 1, "Exactly one K must be marked selected")
        self.assertEqual(selected_flags[0]["k"], selected_k)

        # K was chosen on the training data, never on the test set.
        k_selection = metrics.get("model_info", {}).get("k_selection", {})
        self.assertIn("training", k_selection.get("selected_on", "").lower(),
                      "K must be selected on training data only")

    def test_k_comparison_scores_are_real(self):
        """Each K-comparison row must carry a genuine mean and standard error."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            comparison = json.load(handle)["k_comparison"]

        self.assertGreaterEqual(len(comparison), 4, "K grid should evaluate several values")

        for entry in comparison:
            self.assertIsInstance(entry["k"], int)
            self.assertGreater(entry["k"], 0)
            for key in ("cv_f1_mean", "cv_f1_sd", "cv_f1_se"):
                self.assertIn(key, entry, f"k={entry['k']} is missing '{key}'")
                self.assertIsInstance(entry[key], (int, float))
                self.assertGreaterEqual(entry[key], 0, f"k={entry['k']} '{key}' cannot be negative")

            # A standard error must be consistent with the reported spread.
            self.assertAlmostEqual(
                entry["cv_f1_se"],
                round(entry["cv_f1_sd"] / (entry["folds_evaluated"] ** 0.5), 4),
                delta=0.0002,
                msg=f"k={entry['k']} standard error must be sd / sqrt(folds)"
            )

    def test_class_distribution_note_is_present_and_honest(self):
        """
        The dataset holds one LOW row, so the artifacts must say so rather than
        quietly reporting a tidy three-class model.
        """
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        note = metrics.get("class_distribution_note")
        self.assertIsInstance(note, dict, "A class-distribution note is required")
        self.assertTrue(note.get("message"), "The note must explain the imbalance")

        distribution = metrics["dataset"]["class_distribution"]
        self.assertEqual(sum(distribution.values()), metrics["dataset"]["rows_clean"])
        self.assertEqual(distribution["LOW"], 1,
                         "This dataset holds exactly one LOW row; if that changed, update this test")

        # The unreachability analysis must agree with that fact.
        unreachable = note.get("classes_unreachable_at_selected_k", [])
        votes_to_win = (metrics["selected_k"] + 1) // 2
        for class_name in unreachable:
            self.assertLess(
                distribution[class_name], votes_to_win,
                f"{class_name} was marked unreachable but has enough rows to win a vote"
            )

    def test_feature_scaling_is_recorded(self):
        """The scaling parameters must be stored so prediction can reuse them."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        scaling = metrics.get("feature_scaling")
        self.assertIsInstance(scaling, dict, "Feature scaling must be documented")
        self.assertIn("center", scaling)
        self.assertIn("scale", scaling)

        for name in FEATURES:
            self.assertIn(name, scaling["center"], f"No centre recorded for '{name}'")
            self.assertIn(name, scaling["scale"], f"No scale recorded for '{name}'")
            self.assertGreater(scaling["scale"][name], 0,
                               f"Scale for '{name}' must be positive to avoid dividing by zero")


class TestPredictionApi(unittest.TestCase):
    """POST /api/knn/predict must return real model predictions."""

    def setUp(self):
        self.client = app.test_client()

    def test_valid_prediction_success(self):
        """A valid payload returns 200 with a recognised performance tier."""
        response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertIn(data["prediction"], CLASSES,
                      "Prediction must be one of the trained performance tiers")

    def test_prediction_is_low_medium_or_high(self):
        """The predicted tier must always be one of the three declared classes."""
        data = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()
        self.assertIn(data["prediction"], ("LOW", "MEDIUM", "HIGH"))

    def test_prediction_response_format(self):
        """The success response must carry every documented field."""
        response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)
        data = response.get_json()

        for key in ("success", "model", "prediction", "k", "neighbors"):
            self.assertIn(key, data, f"Response is missing required key '{key}'")

        # Provenance and explanation fields the UI renders.
        for key in ("algorithm", "inputs", "trained_at", "neighbor_class_distribution",
                    "classes", "features", "scaling_applied"):
            self.assertIn(key, data, f"Response is missing provenance key '{key}'")

        # The echoed inputs must match what was sent.
        for name in FEATURES:
            self.assertAlmostEqual(data["inputs"][name], VALID_PAYLOAD[name], places=6)

        # The response must advertise the same K the model was trained with.
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            self.assertEqual(data["k"], json.load(handle)["selected_k"])

    def test_confidence_comes_from_the_real_neighbour_vote(self):
        """
        Confidence, when present, must equal the share of the K neighbours that
        carry the predicted class — not an invented number.
        """
        data = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()

        if "confidence" in data:
            self.assertIsInstance(data["confidence"], (int, float))
            self.assertGreaterEqual(data["confidence"], 0.0)
            self.assertLessEqual(data["confidence"], 1.0)

        distribution = data["neighbor_class_distribution"]
        neighbors = data["neighbors"]
        k = data["k"]

        self.assertEqual(len(neighbors), k, "There must be exactly K neighbours returned")
        self.assertEqual(sum(distribution.values()), k,
                         "The neighbour class distribution must count all K neighbours")

        # The confidence must be the winner's share of the K neighbours.
        winner_count = distribution[data["prediction"]]
        self.assertEqual(winner_count, max(distribution.values()),
                         "The prediction must be the majority class among the neighbours")
        self.assertAlmostEqual(
            data["confidence"], round(winner_count / k, 4), places=4,
            msg="Confidence must be the predicted class's share of the K neighbours"
        )

    def test_neighbor_response_structure(self):
        """
        Each neighbour must carry its rank, class, distance and feature values,
        with distances in ascending order and the listed classes matching the
        reported distribution.
        """
        data = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()
        neighbors = data["neighbors"]
        k = data["k"]

        self.assertIsInstance(neighbors, list)
        self.assertEqual(len(neighbors), k)

        previous_distance = None
        for position, neighbor in enumerate(neighbors):
            self.assertIn("neighbor", neighbor)
            self.assertIn("neighbor_class", neighbor)
            self.assertIn("distance", neighbor)
            self.assertIn("scaled_features", neighbor)

            self.assertEqual(neighbor["neighbor"], position + 1,
                             "Neighbours must be numbered 1..K in nearest-first order")
            self.assertIn(neighbor["neighbor_class"], CLASSES,
                          "A neighbour class must be a trained class")

            distance = neighbor["distance"]
            self.assertIsInstance(distance, (int, float))
            self.assertGreaterEqual(distance, 0, "A Euclidean distance cannot be negative")

            if previous_distance is not None:
                self.assertGreaterEqual(
                    distance, previous_distance - 1e-9,
                    "Neighbours must be ordered by ascending distance"
                )
            previous_distance = distance

            # The neighbour's own feature values must be present for every feature.
            for name in FEATURES:
                self.assertIn(name, neighbor["scaled_features"],
                              f"Neighbour is missing its '{name}' value")

        # The listed classes must match the reported distribution.
        listed = [n["neighbor_class"] for n in neighbors]
        for class_name in CLASSES:
            self.assertEqual(listed.count(class_name), data["neighbor_class_distribution"][class_name])

    def test_prediction_changes_with_input(self):
        """
        Two clearly different students must be classified differently. A
        hard-coded response would return the same tier for both.
        """
        strong = self.client.post('/api/knn/predict', json={
            "study_hours": 34, "attendance": 98, "previous_score": 95,
            "assignments_completed": 9, "practical_score": 95
        }).get_json()
        weak = self.client.post('/api/knn/predict', json={
            "study_hours": 3, "attendance": 60, "previous_score": 38,
            "assignments_completed": 1, "practical_score": 42
        }).get_json()

        self.assertTrue(strong["success"], strong)
        self.assertTrue(weak["success"], weak)
        self.assertNotEqual(
            strong["prediction"], weak["prediction"],
            "Different students produced an identical class — the prediction looks hard-coded"
        )

    def test_prediction_is_reproducible(self):
        """
        The same input must yield the same prediction because predict.R loads a
        fixed saved model rather than retraining on each request.
        """
        first = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()
        second = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()

        self.assertEqual(
            first["prediction"], second["prediction"],
            "Identical input produced different predictions — the model is not being reused"
        )
        self.assertEqual(
            first.get("confidence"), second.get("confidence"),
            "Identical input produced different confidence values"
        )
        self.assertEqual(
            [n["distance"] for n in first["neighbors"]],
            [n["distance"] for n in second["neighbors"]],
            "Identical input produced different neighbour distances"
        )

    def test_prediction_surfaces_real_training_metrics(self):
        """The prediction response must carry the real metrics from training."""
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            trained = json.load(handle)["metrics"]

        data = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()

        for metric in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(metric, data, f"Prediction response is missing '{metric}'")
            self.assertAlmostEqual(data[metric], round(trained[metric], 4), places=4)


class TestPredictionValidation(unittest.TestCase):
    """Invalid and missing input must be rejected with clean 400 errors."""

    def setUp(self):
        self.client = app.test_client()

    def assert_rejected(self, payload, expected_fragment):
        response = self.client.post('/api/knn/predict', json=payload)
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

    def test_negative_study_hours_rejected(self):
        """study_hours must be 0 or greater."""
        self.assert_rejected(dict(VALID_PAYLOAD, study_hours=-1), "study hours")

    def test_invalid_attendance_low_rejected(self):
        """A negative attendance is not a real percentage."""
        self.assert_rejected(dict(VALID_PAYLOAD, attendance=-5), "attendance")

    def test_invalid_attendance_high_rejected(self):
        """An attendance above 100 is not a real percentage."""
        self.assert_rejected(dict(VALID_PAYLOAD, attendance=150), "attendance")

    def test_invalid_previous_score_rejected(self):
        """A previous score outside 0-100 is not a real percentage."""
        self.assert_rejected(dict(VALID_PAYLOAD, previous_score=120), "previous score")

    def test_negative_assignments_rejected(self):
        """assignments_completed cannot be negative."""
        self.assert_rejected(dict(VALID_PAYLOAD, assignments_completed=-1), "assignments")

    def test_invalid_practical_score_rejected(self):
        """A practical score outside 0-100 is not a real percentage."""
        self.assert_rejected(dict(VALID_PAYLOAD, practical_score=101), "practical score")

    def test_non_numeric_value_rejected(self):
        """A non-numeric feature value must be rejected."""
        self.assert_rejected(dict(VALID_PAYLOAD, attendance="excellent"), "attendance")

    def test_empty_string_value_rejected(self):
        """An empty value is not a number."""
        self.assert_rejected(dict(VALID_PAYLOAD, study_hours=""), "study hours")

    def test_missing_fields_rejected(self):
        """A partial payload must report every missing field."""
        data = self.assert_rejected({"study_hours": 5}, "required")
        self.assertGreaterEqual(len(data["errors"]), 4,
                                "All four missing fields should be reported")

    def test_empty_body_rejected(self):
        """An empty JSON object must be rejected."""
        self.assert_rejected({}, "required")

    def test_unknown_field_rejected(self):
        """A typo'd field name must not be silently ignored."""
        data = self.assert_rejected(dict(VALID_PAYLOAD, studhours=7), "unrecognised")
        self.assertIn("studhours", " ".join(data["errors"]))

    def test_non_json_request_rejected(self):
        """A request without a JSON content type must return 415."""
        response = self.client.post('/api/knn/predict', data="study_hours=5")
        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.get_json()["success"])

    def test_malformed_json_rejected(self):
        """A body that claims to be JSON but is not must return 400."""
        response = self.client.post(
            '/api/knn/predict',
            data="{not valid json",
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_percentage_boundaries_accepted(self):
        """Both ends of every 0-100 range are valid percentages."""
        for field in ("attendance", "previous_score", "practical_score"):
            for bound in (0, 100):
                response = self.client.post(
                    '/api/knn/predict', json=dict(VALID_PAYLOAD, **{field: bound})
                )
                self.assertEqual(
                    response.status_code, 200,
                    f"{field}={bound} was rejected: {response.get_data(as_text=True)}"
                )

    def test_zero_study_hours_accepted(self):
        """Zero study hours is logically valid even if unusual."""
        response = self.client.post('/api/knn/predict', json=dict(VALID_PAYLOAD, study_hours=0))
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

    def test_out_of_range_value_warns_but_is_answered(self):
        """
        A logically valid value outside the training range must still be
        classified, and the extrapolation must be reported rather than hidden.
        """
        response = self.client.post('/api/knn/predict', json=dict(
            VALID_PAYLOAD, study_hours=60
        ))
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], CLASSES)

        warned_fields = {w["field"] for w in data.get("warnings", [])}
        self.assertIn("study_hours", warned_fields,
                      "An out-of-range study_hours must produce an extrapolation warning")

    def test_in_range_input_has_no_extrapolation_warning(self):
        """A normal student must not be told the prediction was extrapolated."""
        data = self.client.post('/api/knn/predict', json=VALID_PAYLOAD).get_json()
        out_of_range = [w for w in data.get("warnings", []) if w["field"] in FEATURES]
        self.assertEqual(out_of_range, [],
                         f"In-range input produced warnings: {out_of_range}")

    def test_validate_student_input_unit_level(self):
        """The validation helper returns cleaned floats and error messages."""
        cleaned, errors = validate_student_input(VALID_PAYLOAD)
        self.assertEqual(errors, [])
        self.assertEqual(set(cleaned.keys()), set(FEATURES))
        for value in cleaned.values():
            self.assertIsInstance(value, float)

        _, errors = validate_student_input({"study_hours": 5})
        self.assertTrue(errors)
        self.assertEqual(len(errors), 4, "One error per missing field")

        _, errors = validate_student_input("not a dict")
        self.assertEqual(len(errors), 1)

        # Booleans must not be coerced to 1 / 0.
        _, errors = validate_student_input(dict(VALID_PAYLOAD, study_hours=True))
        self.assertTrue(errors, "A boolean must not be accepted as a number")


class TestRExecutionFailureHandling(unittest.TestCase):
    """Failures inside the R engine must surface as clean API errors, not crashes."""

    def setUp(self):
        self.client = app.test_client()

    def test_r_execution_failure_returns_502(self):
        """When RRunner reports failure, the API returns 502 with a message."""
        failure = {
            "success": False,
            "message": "R script execution failed with exit code 1",
            "error": "Error in class::knn(): object 'train_x' not found",
            "return_code": 1
        }
        with patch("backend.services.knn_service.r_runner.execute_script",
                   return_value=failure):
            response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train_x' not found", data["error"])

    def test_r_unparseable_output_returns_502(self):
        """Non-JSON stdout from R must produce a clean 502, not a crash."""
        garbage = {
            "success": True,
            "data": None,          # runner could not parse a JSON object
            "stdout": "not json at all",
            "return_code": 0
        }
        with patch("backend.services.knn_service.r_runner.execute_script",
                   return_value=garbage):
            response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.get_json()["success"])

    def test_r_returns_unknown_class_returns_502(self):
        """
        A performance tier the model was never trained on must be rejected
        rather than passed through to the user.
        """
        bad_prediction = {
            "success": True,
            "data": {"success": True, "prediction": "EXCEPTIONAL", "model": MODEL_NAME},
            "return_code": 0
        }
        with patch("backend.services.knn_service.r_runner.execute_script",
                   return_value=bad_prediction):
            response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertIn("unexpected performance class", response.get_json()["error"].lower())

    def test_missing_model_file_returns_503(self):
        """An untrained model must return 503 with retraining instructions."""
        # app.py puts backend/ on sys.path, so the routes hold this service as
        # 'services.knn_service' — a different module object than the
        # 'backend.services.knn_service' alias used above.
        with patch("services.knn_service.is_model_trained", return_value=False):
            response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)

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
                "error": "Field 'attendance' must be between 0 and 100.",
                "errors": ["Field 'attendance' must be between 0 and 100."]
            }
        }
        with patch("backend.services.knn_service.r_runner.execute_script",
                   return_value=r_rejection):
            response = self.client.post('/api/knn/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])


class TestMetricsAndConfigApi(unittest.TestCase):
    """Metrics, config and schema endpoints must serve the real artifacts."""

    def setUp(self):
        self.client = app.test_client()

    def test_metrics_endpoint(self):
        """GET /api/knn/metrics returns the real classification metrics."""
        response = self.client.get('/api/knn/metrics')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertIsInstance(data["selected_k"], int)

        metrics = data["metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            self.assertGreaterEqual(metrics[key], 0)

        # The endpoint must agree with the artifact on disk.
        with open(get_metrics_path(), "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)["metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertEqual(metrics[key], on_disk[key])

    def test_metrics_endpoint_reports_dataset_provenance(self):
        """Metrics must describe the real training split."""
        data = self.client.get('/api/knn/metrics').get_json()
        dataset = data["dataset"]

        self.assertEqual(dataset["target"], "performance")
        self.assertEqual(set(dataset["features"]), set(FEATURES))
        self.assertGreater(dataset["rows_train"], 0)
        self.assertGreater(dataset["rows_test"], 0)
        self.assertEqual(
            dataset["rows_train"] + dataset["rows_test"], dataset["rows_clean"],
            "Train + test rows must equal the cleaned dataset size"
        )

    def test_metrics_include_per_class_and_k_comparison(self):
        """Per-class metrics, the confusion matrix and the K grid are exposed."""
        data = self.client.get('/api/knn/metrics').get_json()

        per_class = data["per_class_metrics"]
        for class_name in CLASSES:
            self.assertIn(class_name, per_class)
            self.assertIn("precision", per_class[class_name])
            self.assertIn("recall", per_class[class_name])
            self.assertIn("support", per_class[class_name])

        self.assertEqual(data["confusion_matrix"]["labels"], list(CLASSES))
        self.assertGreaterEqual(len(data["k_comparison"]), 2)
        self.assertIn("feature_scaling", data)

    def test_metrics_missing_artifact_returns_503(self):
        """A missing metrics artifact returns 503 with retraining instructions."""
        with patch("services.knn_service.get_metrics_path",
                   value="/nonexistent/metrics.json"):
            response = self.client.get('/api/knn/metrics')

        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train.R", data["error"])

    def test_config_endpoint_reports_selected_k_and_features(self):
        """GET /api/knn/config exposes the selected K and the feature contract."""
        response = self.client.get('/api/knn/config')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["target"], "performance")
        self.assertEqual(data["classes"], list(CLASSES))
        self.assertIsInstance(data["selected_k"], int)
        self.assertGreater(data["selected_k"], 0)
        self.assertEqual(set(data["features"]), set(FEATURES))
        self.assertTrue(data["model_trained"])

    def test_config_endpoint_does_not_leak_internals(self):
        """
        The config endpoint is for the UI, so it must not expose the stored
        training matrix, its labels, or the raw scaling parameters.
        """
        data = self.client.get('/api/knn/config').get_json()

        # Internal model internals — never part of the public config contract.
        for forbidden in ("train_x", "train_y", "scaler", "center", "scale",
                          "test_predictions", "model_bundle"):
            self.assertNotIn(forbidden, data,
                             f"The config endpoint must not expose '{forbidden}'")

        # The published 'feature_scaling' block carries only a human-readable
        # method and rationale, never the numeric centre/scale values.
        scaling = data.get("feature_scaling", {})
        self.assertEqual(set(scaling.keys()), {"method", "reason"})

    def test_config_endpoint_reports_training_ranges(self):
        """Feature ranges must come from the data, not a hand-written constant."""
        data = self.client.get('/api/knn/config').get_json()
        ranges = data["feature_ranges"]

        self.assertEqual(set(ranges.keys()), set(FEATURES))
        for name in FEATURES:
            self.assertLess(ranges[name]["min"], ranges[name]["max"],
                            f"'{name}' must have a real min/max from the dataset")

        # Percentages must respect their documented 0-100 domain.
        for name in ("attendance", "previous_score", "practical_score"):
            self.assertGreaterEqual(ranges[name]["min"], 0)
            self.assertLessEqual(ranges[name]["max"], 100)

    def test_config_endpoint_reports_class_balance_note(self):
        """The single-LOW-row limitation must be visible to the frontend."""
        data = self.client.get('/api/knn/config').get_json()
        note = data["class_distribution_note"]

        self.assertTrue(note.get("message"), "The class-balance message is required")
        self.assertIn("classes_without_test_support", note)
        self.assertIn("classes_unreachable_at_selected_k", note)

    def test_config_endpoint_missing_artifact_returns_503(self):
        """A missing artifact returns 503 with retraining instructions."""
        with patch("services.knn_service.get_metrics_path",
                   value="/nonexistent/metrics.json"):
            response = self.client.get('/api/knn/config')

        self.assertEqual(response.status_code, 503)
        self.assertIn("train.R", response.get_json()["error"])

    def test_schema_endpoint_reports_training_ranges(self):
        """GET /api/knn/schema exposes the real dataset ranges."""
        response = self.client.get('/api/knn/schema')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["model_trained"])
        self.assertEqual(data["classes"], list(CLASSES))

        features = {f["name"]: f for f in data["features"]}
        self.assertEqual(set(features.keys()), set(FEATURES))

        study_hours = features["study_hours"]
        self.assertGreater(study_hours["min"], 0, "This dataset's minimum study_hours is above 0")
        self.assertLess(study_hours["min"], study_hours["max"])

    def test_frontend_page_and_assets_exist(self):
        """The Phase 5 page and its assets must be present."""
        for relative in (
            os.path.join("frontend", "knn.html"),
            os.path.join("frontend", "css", "knn.css"),
            os.path.join("frontend", "js", "knn.js"),
        ):
            path = os.path.join(PROJECT_ROOT, relative)
            self.assertTrue(os.path.isfile(path), f"Missing frontend file: {relative}")
            self.assertGreater(os.path.getsize(path), 0, f"Empty frontend file: {relative}")

    def test_frontend_page_wires_up_every_required_element(self):
        """
        The page must contain the elements knn.js queries, and the script that
        consumes the API — otherwise the page would silently render nothing.
        """
        with open(os.path.join(PROJECT_ROOT, "frontend", "knn.html"),
                  "r", encoding="utf-8") as handle:
            html = handle.read()

        for element_id in (
            "predictionForm", "btnPredict", "resultPanel", "resultPerformanceBadge",
            "inputStudy_hours", "inputAttendance", "inputPrevious_score",
            "inputAssignments_completed", "inputPractical_score",
            "neighborPanel", "neighborList", "kTableBody",
            "metricAccuracy", "metricPrecision", "metricRecall", "metricF1",
        ):
            self.assertIn(f'id="{element_id}"', html,
                          f"knn.html is missing the element '{element_id}'")

        self.assertIn('src="js/knn.js"', html, "knn.html must load its page script")
        self.assertIn("PREDICT PERFORMANCE", html, "The submit button label is required")
        self.assertIn("HOW KNN REACHED THIS RESULT", html,
                      "The KNN explanation section is required")
        self.assertIn("K Selection", html, "The K comparison section is required")

    def test_frontend_script_calls_the_predict_endpoint(self):
        """The page script must post to the documented prediction endpoint."""
        with open(os.path.join(PROJECT_ROOT, "frontend", "js", "knn.js"),
                  "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("/api/knn/predict", source)
        self.assertIn("/api/knn/metrics", source)
        self.assertIn("/api/knn/config", source)
        self.assertIn("neighbors", source, "The page must render the model's neighbours")

    def test_dashboard_marks_knn_active(self):
        """The dashboard's KNN card must be ACTIVE and open the page."""
        index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
        with open(index_path, "r", encoding="utf-8") as handle:
            html = handle.read()

        self.assertIn('href="knn.html"', html, "Dashboard does not link to knn.html")
        self.assertNotIn("Coming in Phase 5", html,
                         "Stale 'Coming in Phase 5' text remains on the KNN card")

        # The earlier models must still be intact.
        self.assertIn('href="regression.html"', html,
                      "Dashboard no longer links to the Phase 3 regression page")
        self.assertIn('href="decision-tree.html"', html,
                      "Dashboard no longer links to the Phase 4 decision tree page")
        # K-Means stays Phase 6 and untouched.
        self.assertIn("Coming in Phase 6", html,
                      "The Phase 6 K-Means card must still be marked as upcoming")


class TestPhasePreservation(unittest.TestCase):
    """Phase 1-4 endpoints must keep working after Phase 5."""

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

    def test_phase4_decision_tree_still_works(self):
        """The Phase 4 Decision Tree endpoints must be unaffected."""
        response = self.client.post('/api/decision-tree/predict', json={
            "age": 32, "income": 95000, "credit_score": 735,
            "existing_loans": 1, "employment_years": 5
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertTrue(response.get_json()["success"])

        metrics = self.client.get('/api/decision-tree/metrics')
        self.assertEqual(metrics.status_code, 200)
        self.assertTrue(metrics.get_json()["success"])

    def test_root_endpoint_lists_all_three_models(self):
        data = self.client.get('/').get_json()
        endpoints = data["endpoints"]

        for key in ("regression_predict", "decision_tree_predict",
                    "knn_predict", "knn_metrics", "knn_config"):
            self.assertIn(key, endpoints)
        self.assertIn("Phase 3", data["phase"])
        self.assertIn("Phase 4", data["phase"])
        self.assertIn("Phase 5", data["phase"])

    def test_kmeans_is_not_implemented(self):
        """Phase 6 is out of scope, so no K-Means training script may exist."""
        self.assertFalse(
            os.path.isfile(os.path.join(PROJECT_ROOT, "r_models", "kmeans", "train.R")),
            "K-Means must not be implemented in Phase 5"
        )


if __name__ == '__main__':
    unittest.main()
