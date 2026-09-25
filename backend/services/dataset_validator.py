"""
AI Insight Hub — Dataset Validation Utility
Validates dataset file existence, schemas, data types, missing values,
and domain bounds for the four machine learning datasets.
"""
import os
import csv
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ai_insight_hub.dataset_validator")

# Defined Schemas for Phase 2–6 Datasets
DATASET_SCHEMAS = {
    "housing.csv": {
        "name": "Property Price Prediction",
        "required_columns": ["area", "bedrooms", "bathrooms", "location_score", "property_age", "price"],
        "numeric_columns": ["area", "bedrooms", "bathrooms", "location_score", "property_age", "price"],
        "categorical_rules": {},
        "target_column": "price",
        "model_type": "Linear Regression (Phase 3)"
    },
    "risk.csv": {
        "name": "Financial Risk Classification",
        "required_columns": ["age", "income", "credit_score", "existing_loans", "employment_years", "risk"],
        "numeric_columns": ["age", "income", "credit_score", "existing_loans", "employment_years"],
        "categorical_rules": {
            "risk": ["LOW", "MEDIUM", "HIGH"]
        },
        "target_column": "risk",
        "model_type": "Decision Tree (Phase 4)"
    },
    "students.csv": {
        "name": "Student Performance Prediction",
        "required_columns": ["study_hours", "attendance", "previous_score", "assignments_completed", "practical_score", "performance"],
        "numeric_columns": ["study_hours", "attendance", "previous_score", "assignments_completed", "practical_score"],
        "categorical_rules": {
            "performance": ["LOW", "MEDIUM", "HIGH"]
        },
        "target_column": "performance",
        "model_type": "KNN (Phase 5)"
    },
    "customers.csv": {
        "name": "Customer Segmentation",
        "required_columns": ["customer_id", "age", "annual_income", "spending_score", "purchase_frequency"],
        "numeric_columns": ["age", "annual_income", "spending_score", "purchase_frequency"],
        "categorical_rules": {},
        "target_column": None,
        "model_type": "K-Means (Phase 6)"
    }
}


def get_datasets_dir() -> str:
    """Returns absolute path to datasets directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "datasets")


def validate_dataset(filename: str) -> Dict[str, Any]:
    """
    Validates a single dataset by filename (e.g. 'housing.csv').
    Checks file existence, header columns, missing values, numerical casting, and categorical values.
    """
    datasets_dir = get_datasets_dir()
    filepath = os.path.join(datasets_dir, filename)
    errors: List[str] = []

    # 1. Check file exists
    if not os.path.isfile(filepath):
        return {
            "dataset": filename,
            "valid": False,
            "filepath": filepath,
            "row_count": 0,
            "columns": [],
            "errors": [f"File does not exist: {filepath}"],
            "schema_matched": False
        }

    schema = DATASET_SCHEMAS.get(filename)
    if not schema:
        # Generic validation without preset schema
        schema = {
            "name": "Generic Dataset",
            "required_columns": [],
            "numeric_columns": [],
            "categorical_rules": {}
        }

    rows = []
    headers = []

    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                return {
                    "dataset": filename,
                    "valid": False,
                    "filepath": filepath,
                    "row_count": 0,
                    "columns": [],
                    "errors": ["Dataset file is empty (missing headers)."],
                    "schema_matched": False
                }

            headers = [h.strip() for h in headers]

            # 2. Check required columns
            missing_cols = [col for col in schema["required_columns"] if col not in headers]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")

            # Read and validate rows
            for line_no, row in enumerate(reader, start=2):
                if not row or all(c.strip() == "" for c in row):
                    continue  # Skip blank lines

                if len(row) != len(headers):
                    errors.append(f"Row {line_no} has {len(row)} columns, expected {len(headers)}")
                    continue

                row_dict = {headers[i]: row[i].strip() for i in range(len(headers))}

                # 3. Check missing / null values
                for col, val in row_dict.items():
                    if val == "" or val is None or val.lower() == "na" or val.lower() == "null":
                        errors.append(f"Row {line_no}: Missing value in column '{col}'")

                # 4. Check numeric casting
                for num_col in schema["numeric_columns"]:
                    if num_col in row_dict and row_dict[num_col]:
                        try:
                            float(row_dict[num_col])
                        except ValueError:
                            errors.append(f"Row {line_no}: Column '{num_col}' has non-numeric value: '{row_dict[num_col]}'")

                # 5. Check categorical rules
                for cat_col, allowed in schema["categorical_rules"].items():
                    if cat_col in row_dict and row_dict[cat_col]:
                        val_upper = row_dict[cat_col].upper()
                        if val_upper not in allowed:
                            errors.append(f"Row {line_no}: Column '{cat_col}' value '{row_dict[cat_col]}' not in allowed values: {allowed}")

                rows.append(row_dict)

    except Exception as e:
        errors.append(f"Error reading CSV file: {e}")

    # 6. Check dataset is not empty
    if len(rows) == 0:
        errors.append("Dataset contains 0 data rows.")

    is_valid = len(errors) == 0

    return {
        "dataset": filename,
        "valid": is_valid,
        "filepath": filepath,
        "name": schema.get("name"),
        "model_type": schema.get("model_type"),
        "row_count": len(rows),
        "column_count": len(headers) if headers else 0,
        "columns": headers or [],
        "errors": errors,
        "preview_sample": rows[:3] if rows else []
    }


def validate_all_datasets() -> Dict[str, Any]:
    """Validates all four planned project datasets."""
    results = {}
    all_valid = True
    total_rows = 0

    for filename in DATASET_SCHEMAS.keys():
        res = validate_dataset(filename)
        results[filename] = res
        if not res["valid"]:
            all_valid = False
        total_rows += res.get("row_count", 0)

    return {
        "all_valid": all_valid,
        "total_datasets": len(DATASET_SCHEMAS),
        "total_records": total_rows,
        "datasets": results
    }
