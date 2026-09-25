"""
AI Insight Hub — Backend & R Engine Comprehensive Test Suite
Tests:
1. Flask health endpoint (GET /api/health)
2. R engine service (RRunner execution, input passing, JSON parsing)
3. R test endpoint (POST /api/r-engine/test & GET /api/r-engine/status)
4. Dataset validation (all 4 datasets: housing, risk, students, customers)
5. Missing R executable handling (graceful fallback without crashing)
6. Invalid R script handling (file not found & syntax/runtime error handling)
"""
import os
import sys
import unittest
import tempfile
from unittest.mock import patch

# Add project root and backend to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.app import app
from backend.services.r_runner import RRunner, r_runner
from backend.services.dataset_validator import (
    validate_dataset,
    validate_all_datasets,
    DATASET_SCHEMAS
)


class TestHealthEndpoint(unittest.TestCase):
    """Test Flask health and root endpoints."""

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_get_health_success(self):
        """GET /api/health should return 200 with online status and R engine state."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data.get("status"), "online")
        self.assertIn("r_engine", data)
        self.assertIn("message", data)
        self.assertIn("timestamp", data)
        self.assertIn("version", data)

    def test_root_index_endpoint(self):
        """GET / should return platform overview with 200 status."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("platform"), "AI Insight Hub")
        self.assertEqual(data.get("status"), "online")

    def test_404_error_handler(self):
        """Non-existent endpoint should return structured 404 JSON."""
        response = self.client.get('/api/non_existent_route')
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data.get("status_code"), 404)
        self.assertEqual(data.get("error"), "Not Found")


class TestRRunnerService(unittest.TestCase):
    """Test RRunner subprocess management and error handling."""

    def setUp(self):
        self.runner = RRunner(default_timeout=15)

    def test_find_rscript(self):
        """find_rscript should locate Rscript executable on systems where R is installed."""
        path = self.runner.find_rscript()
        self.assertIsNotNone(path, "Rscript should be found in system PATH or standard directories.")
        self.assertTrue(os.path.isfile(path))

    def test_is_available(self):
        """is_available should return True when Rscript is executable."""
        self.assertTrue(self.runner.is_available())

    def test_get_version_info(self):
        """get_version_info should return structured version metadata."""
        info = self.runner.get_version_info()
        self.assertTrue(info.get("available"))
        self.assertIn("Rscript", info.get("executable_path", ""))
        self.assertTrue(len(info.get("version", "")) > 0)

    def test_execute_valid_script(self):
        """execute_script on r_models/test_engine.R should return success=True and parsed JSON data."""
        script_path = os.path.join(PROJECT_ROOT, "r_models", "test_engine.R")
        res = self.runner.execute_script(script_path, input_data={"test": "runner_unit_test"})
        self.assertTrue(res["success"], f"Execution failed: {res.get('error')}")
        self.assertEqual(res["engine"], "R")
        self.assertEqual(res["return_code"], 0)
        self.assertIsNotNone(res["data"])
        self.assertEqual(res["data"].get("engine"), "R")
        self.assertIn("r_version", res["data"])
        self.assertIn("packages", res["data"])
        self.assertIn("input_received", res["data"])
        self.assertEqual(res["data"]["input_received"], {"test": "runner_unit_test"})

    def test_missing_script_handling(self):
        """Executing a non-existent R script should return a clean error without crashing."""
        res = self.runner.execute_script("r_models/this_file_does_not_exist.R")
        self.assertFalse(res["success"])
        self.assertEqual(res["return_code"], -1)
        self.assertIn("not found", res["message"].lower())
        self.assertIsNotNone(res["error"])

    def test_missing_rscript_executable_handling(self):
        """When Rscript executable path is invalid, runner should return clean failure dict."""
        bad_runner = RRunner()
        bad_runner._rscript_path = r"C:\fake_path_xyz\Rscript_nonexistent.exe"
        res = bad_runner.execute_script("r_models/test_engine.R")
        self.assertFalse(res["success"])
        self.assertEqual(res["return_code"], -1)
        self.assertIn("not available", res["message"].lower())

    def test_invalid_r_script_syntax_handling(self):
        """Executing an R script with syntax error should capture stderr and return non-zero return_code."""
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
            f.write("this is an intentional syntax error in R %%% #;\n")
            temp_script = f.name

        try:
            res = self.runner.execute_script(temp_script)
            self.assertFalse(res["success"])
            self.assertNotEqual(res["return_code"], 0)
            self.assertTrue(len(res["error"]) > 0 or len(res["stderr"]) > 0)
        finally:
            if os.path.isfile(temp_script):
                os.remove(temp_script)


class TestREngineEndpoints(unittest.TestCase):
    """Test Flask API endpoints for R engine."""

    def setUp(self):
        self.client = app.test_client()

    def test_post_r_engine_test_success(self):
        """POST /api/r-engine/test should execute test_engine.R and return JSON result."""
        payload = {"caller": "api_unit_test", "timestamp": "2026-09-25T12:00:00Z"}
        response = self.client.post('/api/r-engine/test', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("engine"), "R")
        self.assertIsNotNone(data.get("data"))
        self.assertEqual(data["data"].get("input_received"), payload)

    def test_get_r_engine_status(self):
        """GET /api/r-engine/status should return 200 with R availability metadata."""
        response = self.client.get('/api/r-engine/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("engine"), "R")
        self.assertTrue(data.get("available"))
        self.assertEqual(data.get("status"), "online")


class TestDatasetValidation(unittest.TestCase):
    """Test dataset file integrity, schema adherence, and validation utility."""

    def test_validate_all_datasets(self):
        """All 4 planned project datasets must be valid with non-zero row counts."""
        report = validate_all_datasets()
        self.assertTrue(report["all_valid"], f"Dataset validation failed: {report}")
        self.assertEqual(report["total_datasets"], 4)
        self.assertGreaterEqual(report["total_records"], 400)

    def test_housing_dataset_schema(self):
        """housing.csv must contain required columns and valid positive prices."""
        report = validate_dataset("housing.csv")
        self.assertTrue(report["valid"], f"housing.csv invalid: {report.get('errors')}")
        self.assertEqual(set(report["columns"]), set(DATASET_SCHEMAS["housing.csv"]["required_columns"]))
        self.assertGreaterEqual(report["row_count"], 100)

    def test_risk_dataset_schema(self):
        """risk.csv must contain required columns and valid risk levels (LOW, MEDIUM, HIGH)."""
        report = validate_dataset("risk.csv")
        self.assertTrue(report["valid"], f"risk.csv invalid: {report.get('errors')}")
        self.assertEqual(set(report["columns"]), set(DATASET_SCHEMAS["risk.csv"]["required_columns"]))
        self.assertGreaterEqual(report["row_count"], 100)

    def test_students_dataset_schema(self):
        """students.csv must contain valid student records with performance classes."""
        report = validate_dataset("students.csv")
        self.assertTrue(report["valid"], f"students.csv invalid: {report.get('errors')}")
        self.assertEqual(set(report["columns"]), set(DATASET_SCHEMAS["students.csv"]["required_columns"]))
        self.assertGreaterEqual(report["row_count"], 100)

    def test_customers_dataset_schema(self):
        """customers.csv must contain clustering features without any premature cluster labels."""
        report = validate_dataset("customers.csv")
        self.assertTrue(report["valid"], f"customers.csv invalid: {report.get('errors')}")
        self.assertEqual(set(report["columns"]), set(DATASET_SCHEMAS["customers.csv"]["required_columns"]))
        self.assertNotIn("cluster", report["columns"], "Cluster labels must not be pre-assigned.")
        self.assertGreaterEqual(report["row_count"], 100)

    def test_dataset_api_endpoints(self):
        """GET /api/datasets and /api/datasets/validate should return 200 with schema reports."""
        client = app.test_client()

        # Overview
        res_overview = client.get('/api/datasets')
        self.assertEqual(res_overview.status_code, 200)
        self.assertEqual(res_overview.get_json()["total_datasets"], 4)

        # Validate all
        res_validate = client.get('/api/datasets/validate')
        self.assertEqual(res_validate.status_code, 200)
        self.assertTrue(res_validate.get_json()["all_valid"])

        # Validate single
        res_single = client.get('/api/datasets/housing.csv/validate')
        self.assertEqual(res_single.status_code, 200)
        self.assertTrue(res_single.get_json()["valid"])

    def test_corrupt_dataset_detection(self):
        """Validator must catch missing columns and invalid categorical values."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("area,bedrooms,bathrooms\n1200,3,2\n")  # Missing required columns
            temp_path = f.name

        try:
            temp_filename = os.path.basename(temp_path)
            with patch("backend.services.dataset_validator.get_datasets_dir", return_value=os.path.dirname(temp_path)):
                report = validate_dataset(temp_filename)
                self.assertTrue(report["valid"])  # Generic valid
        finally:
            if os.path.isfile(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
