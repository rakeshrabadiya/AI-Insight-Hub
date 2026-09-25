# AI Insight Hub — Multi-Model Machine Learning Analytics Platform

[![Platform](https://img.shields.io/badge/Platform-AI%20Insight%20Hub-00f0ff.svg)](#)
[![Phase](https://img.shields.io/badge/Status-Phase%203%20Linear%20Regression%20Active-10b981.svg)](#)
[![Backend](https://img.shields.io/badge/Backend-Python%20Flask-3b82f6.svg)](#)
[![Frontend](https://img.shields.io/badge/Frontend-Vanilla%20HTML%20%2F%20CSS%20%2F%20JS-f59e0b.svg)](#)
[![ML-Engine](https://img.shields.io/badge/ML%20Engine-R%204.6.1%20Connected-6366f1.svg)](#)
[![Academic](https://img.shields.io/badge/Academic-GCF%20Training%20%7C%20Ethnotech%20%7C%20Parul%20University-8b5cf6.svg)](#)

---

## 📌 Project Overview

**AI Insight Hub** is a multi-model machine learning analytics platform engineered to deliver unified predictions, risk classifications, student performance insights, and unsupervised customer clustering. 

Built as part of the **GCF Training by Ethnotech at Parul University**, the platform combines a lightweight, high-performance vanilla web frontend with a modular Python Flask REST API backend, designed to orchestrate statistical and machine learning algorithms written in **R**.

---

## 🎯 Project Objectives

1. **Multi-Model Machine Learning Architecture:** Host four distinct ML algorithms spanning supervised regression, supervised classification, and unsupervised clustering under a unified interface.
2. **Zero-Bloat Full-Stack Integration:** Deliver an ultra-responsive client interface using pure vanilla web standards (HTML5, CSS3, ES6+ JavaScript) connected to a Python Flask REST backend.
3. **Robust R Model Execution:** Utilize R for statistical computation, data preprocessing, and model fitting, while exposing clean JSON REST endpoints via Flask.
4. **Interactive Diagnostics & Telemetry:** Provide real-time health checks, latency monitoring, and transparent payload inspection across system layers.

---

## 🛠️ Technology Stack

| Layer | Technologies & Tools | Rationale |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3 (Custom Properties & Glassmorphism), Vanilla JavaScript (ES6+) | Framework-free, zero runtime overhead, instant browser rendering, pure standards. |
| **Backend** | Python 3.13, Flask 3.x, Flask-CORS | Lightweight microframework with modular blueprint routing and cross-origin REST capabilities. |
| **R Execution Subsystem** | R 4.6.1, `Rscript` Subprocess Bridge (`backend/services/r_runner.py`) | Decoupled subprocess execution, timeout enforcement, stdin/args serialization, stderr isolation. |
| **Machine Learning (R)** | `rpart`, `class`, `cluster`, `ggplot2`, `jsonlite`, `readr` | Statistical modeling algorithms and serialization tools. |
| **Database** | SQLite (*Planned for later persistence phases*) | Zero-configuration relational storage for model metrics and inference logs. |
| **Version Control** | Git & GitHub | Distributed version control and milestone-driven commit tracking. |

> **Strict Constraints:** No React, No Tailwind CSS, No Node.js, and No frontend build frameworks are used in this project.

---

## 🚀 Current Project Phase

### **Phase 3: Linear Regression — Property Price Prediction (ACTIVE / COMPLETED)**

Phase 3 delivers the **first real machine learning model** in the platform. A genuine
multiple linear regression model is trained in R, persisted to disk, and served through
the Flask API to an interactive frontend page.

> ⚠️ **Educational Notice:** This is an **educational machine learning implementation**
> built for the GCF Training curriculum. The model is trained on a synthetic dataset for
> academic demonstration. It is **not** a production valuation tool and must not be used
> for real financial or property decisions.

#### What Phase 3 Delivers

- ✅ **Real R Linear Regression Model:** `stats::lm()` Ordinary Least Squares fitted on `datasets/housing.csv`.
- ✅ **Training Script (`r_models/regression/train.R`):** Loads and validates the dataset, cleans invalid rows, splits into train/test, fits the model, predicts on the test set, and computes real metrics.
- ✅ **Saved Model (`r_models/regression/model.rds`):** The fitted model is persisted once. Predictions **load** this file — the model is never retrained per request.
- ✅ **Prediction Script (`r_models/regression/predict.R`):** Loads `model.rds`, validates input against the training ranges, and returns a structured JSON prediction.
- ✅ **Flask API:** `POST /api/regression/predict`, `GET /api/regression/metrics`, `GET /api/regression/evaluation`, `GET /api/regression/schema`.
- ✅ **Interactive Frontend (`frontend/regression.html`):** Prediction form, premium result card, live model performance tiles, actual-vs-predicted scatter chart, and a fitted-coefficient panel.
- ✅ **Dashboard Integration:** The Linear Regression model card is now marked **ACTIVE** with an **OPEN MODEL** button linking to the new page.
- ✅ **Comprehensive Tests:** 59 Python tests + 50 frontend logic tests, all passing.

#### Dataset Features & Target Variable

| Role | Column | Description | Observed Range |
| :--- | :--- | :--- | :--- |
| **Target** | `price` | Property market value in Indian Rupees (INR) | 149,126 – 724,382 |
| Feature | `area` | Built-up area in square feet | 669.7 – 4,190.5 |
| Feature | `bedrooms` | Number of bedrooms | 1 – 5 |
| Feature | `bathrooms` | Number of bathrooms | 1 – 4 |
| Feature | `location_score` | Location desirability score | 3.1 – 9.7 |
| Feature | `property_age` | Property age in years | 1 – 30 |

The model formula is `price ~ area + bedrooms + bathrooms + location_score + property_age`.
All ranges above are **computed from the dataset during training**, not hard-coded.

#### Training Process

`r_models/regression/train.R` performs these steps, in order:

1. **Load** `datasets/housing.csv` (120 rows).
2. **Validate** that every required column is present.
3. **Clean** — coerce columns to numeric, drop incomplete rows and rows with non-positive area or price, and report how many rows were dropped.
4. **Split** — 80 / 20 train/test with a fixed seed (`42`) for reproducibility → **96 training rows, 24 test rows**.
5. **Train** — `stats::lm(price ~ area + bedrooms + bathrooms + location_score + property_age)`.
6. **Predict** on the held-out test set with `predict()`.
7. **Evaluate** — compute R², RMSE and MAE from the real predictions.
8. **Persist** — `model.rds`, `metrics.json` and `evaluation.json`.

#### Evaluation Metrics (Real, From the Trained Model)

These are the actual values produced by `train.R` on the 24 held-out test records:

| Metric | Test Set Value | Training Set Value | Meaning |
| :--- | ---: | ---: | :--- |
| **R²** | **0.9967** | 0.9980 | 99.67% of the price variance is explained by the features. |
| **Adjusted R²** | **0.9979** | — | R² penalised for the number of predictors. |
| **RMSE** | **₹5,646.30** | ₹5,765.38 | Typical prediction error, in Rupees. |
| **MAE** | **₹5,005.34** | ₹4,902.24 | Average absolute error, in Rupees. |

Fitted coefficients (all statistically significant at p < 0.001 except the intercept):

| Term | Estimate | Std. Error | t-value | p-value |
| :--- | ---: | ---: | ---: | ---: |
| `(Intercept)` | −988.39 | 2,979.15 | −0.33 | 0.7408 |
| `area` | 111.07 | 0.59 | 189.74 | < 0.0001 |
| `bedrooms` | 17,821.36 | 509.82 | 34.96 | < 0.0001 |
| `bathrooms` | 11,270.06 | 711.41 | 15.84 | < 0.0001 |
| `location_score` | 14,836.62 | 345.16 | 42.98 | < 0.0001 |
| `property_age` | −1,062.57 | 67.90 | −15.65 | < 0.0001 |

**Interpretation:** each additional square foot adds roughly ₹111, each bedroom adds
roughly ₹17,821, and each additional year of property age subtracts roughly ₹1,063 from
the estimated price. The model is behaving exactly as a real-estate regression should.

#### How to Retrain the Model

Retraining is a single command and is required only when the dataset or the formula changes:

```bash
Rscript r_models/regression/train.R
```

This rewrites `model.rds`, `metrics.json` and `evaluation.json`. The API picks up the new
model on the very next prediction — no server restart is needed.

#### How to Test the Model

```bash
# Full Python suite (backend, R bridge, regression API, validation) — 59 tests
python -m unittest discover -s tests -p "test_*.py" -v

# Frontend logic suite (formatting, validation, prediction flow, chart) — 50 tests
# Requires the Flask backend to be running on 127.0.0.1:5000
node tests/frontend_logic_test.js
```

The prediction endpoint can also be exercised directly:

```bash
curl -X POST http://127.0.0.1:5000/api/regression/predict \
  -H "Content-Type: application/json" \
  -d '{"area":1500,"bedrooms":3,"bathrooms":2,"location_score":8,"property_age":5}'
```

Output:
```json
{
  "success": true,
  "model": "Linear Regression",
  "algorithm": "stats::lm (Ordinary Least Squares)",
  "prediction": 355005.39,
  "unit": "INR",
  "r2": 0.9967,
  "rmse": 5646.3,
  "mae": 5005.34,
  "trained_at": "2026-09-25T12:36:57Z"
}
```

#### Phase 3 API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/regression/predict` | Validate input, run `predict.R` against the saved model, return a price. |
| `GET` | `/api/regression/metrics` | Real R², RMSE, MAE, dataset provenance and coefficients from `metrics.json`. |
| `GET` | `/api/regression/evaluation` | Real held-out test-set actual vs predicted pairs for the chart. |
| `GET` | `/api/regression/schema` | Each input feature with the valid range recorded at training time. |

**Status codes for `/api/regression/predict`:**

| Code | Meaning |
| :---: | :--- |
| `200` | Prediction produced successfully. |
| `400` | Invalid or missing input (with a per-field `errors` list). |
| `415` | Request was not sent as JSON. |
| `502` | The R engine failed to execute or returned an unreadable response. |
| `503` | The model has not been trained yet (`model.rds` missing). |

#### Input Validation Rules

Validation is enforced in **both** Python and R, so a request cannot bypass it by calling
the R script directly:

| Field | Rule |
| :--- | :--- |
| `area` | `> 0` and within 669.7 – 4,190.5 |
| `bedrooms` | `> 0` and within 1 – 5 |
| `bathrooms` | `> 0` and within 1 – 4 |
| `location_score` | within 3.1 – 9.7 (the dataset's valid range) |
| `property_age` | `>= 0` and within 1 – 30 |

Unknown fields are rejected rather than silently ignored, and booleans are refused rather
than being coerced to 1 or 0.

---

### Phase 2: R Engine Bridge & Dataset Foundation (COMPLETED)
- ✅ **R Environment Verified:** System `R` and `Rscript` discovered and connected (`R version 4.6.1`).
- ✅ **R Dependencies Installed & Verified:** `rpart`, `class`, `cluster`, `ggplot2`, `jsonlite`, `readr`.
- ✅ **R Subprocess Runner (`r_runner.py`):** Robust subprocess orchestration, automatic Rscript discovery, timeout enforcement, stderr capture, and JSON parsing.
- ✅ **R Health Diagnostic Script (`test_engine.R`):** Execution probe returning environment state, package inventory, and input echo.
- ✅ **Flask R REST API (`POST /api/r-engine/test`):** Bridge endpoint for live R diagnostics with structured error handling.
- ✅ **4 Realistic Datasets Generated (480 total records):**
  1. `datasets/housing.csv` (120 rows) — Regression
  2. `datasets/risk.csv` (120 rows) — Classification
  3. `datasets/students.csv` (120 rows) — KNN Classification
  4. `datasets/customers.csv` (120 rows) — Unsupervised Clustering
- ✅ **Dataset Validation Utility (`dataset_validator.py`):** Schema verification, null checks, numeric parsing, and categorical domain verification.
- ✅ **Frontend Interactive Testing:** "TEST R ENGINE" diagnostic button and multi-tab live payload viewer.
- ✅ **Comprehensive Test Suite (`tests/test_backend.py`):** 19 unit tests passing with 100% success rate.

---

## 📊 Dataset Registry & Schemas

All datasets are stored in `datasets/` and validated for integrity:

| Dataset File | Domain Application | Target ML Model | Records | Features / Columns |
| :--- | :--- | :--- | :---: | :--- |
| **`housing.csv`** | Property Price Prediction | **Linear Regression (Phase 3 — ACTIVE)** | 120 | `area`, `bedrooms`, `bathrooms`, `location_score`, `property_age`, `price` (Target) |
| **`risk.csv`** | Financial Risk Classification | Decision Tree (*Phase 4*) | 120 | `age`, `income`, `credit_score`, `existing_loans`, `employment_years`, `risk` (`LOW`, `MEDIUM`, `HIGH`) |
| **`students.csv`** | Student Performance | K-Nearest Neighbors (*Phase 5*) | 120 | `study_hours`, `attendance`, `previous_score`, `assignments_completed`, `practical_score`, `performance` (`LOW`, `MEDIUM`, `HIGH`) |
| **`customers.csv`** | Customer Segmentation | K-Means Clustering (*Phase 6*) | 120 | `customer_id`, `age`, `annual_income`, `spending_score`, `purchase_frequency` *(No pre-assigned labels)* |

---

## 🏗️ System & R Engine Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Browser Client (Frontend)                  │
│   index.html (Dashboard)  •  regression.html (Predictor)      │
│   CSS3 Glassmorphism  •  Vanilla JS  •  Inline SVG chart     │
│   [Ping API]  [Test R Engine]  [Predict Property Price]       │
└───────────────────────────────┬───────────────────────────────┘
                                │
                         HTTP / JSON REST
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                    Python Flask REST Backend                  │
│   app.py  •  routes/health.py  •  routes/r_engine.py          │
│   routes/datasets.py  •  routes/regression.py  (Phase 3)     │
│   services/r_runner.py  •  services/dataset_validator.py      │
│   services/regression_service.py  (validation + orchestration)│
└───────────────────────────────┬───────────────────────────────┘
                                │
                 Subprocess IPC (stdin / args / stdout)
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                    R Machine Learning Subsystem               │
│   Rscript.exe (v4.6.1)  •  r_models/test_engine.R             │
│   Phase 3: r_models/regression/train.R  → model.rds          │
│            r_models/regression/predict.R → predict()          │
│   Packages: rpart • class • cluster • ggplot2 • jsonlite      │
│   Datasets: datasets/housing.csv, risk.csv, students.csv, ... │
└───────────────────────────────────────────────────────────────┘
```

**Phase 3 prediction flow (end to end):**

```
Form input → POST /api/regression/predict → validate_property_input()
   → RRunner.execute_script("r_models/regression/predict.R")
   → readRDS("model.rds") → stats::predict()
   → JSON back to Flask → JSON to browser → premium result card
```

---

## 📁 Repository Directory Structure

```text
AI-Insight-Hub/
├── frontend/                     # Pure client-side application
│   ├── index.html                # Main dashboard UI with R test trigger & model registry
│   ├── regression.html           # Phase 3: Property Price Predictor page
│   ├── css/
│   │   ├── style.css             # Main styling, design tokens & glassmorphism
│   │   ├── responsive.css        # Adaptive mobile & tablet breakpoints
│   │   └── regression.css        # Phase 3: predictor form, result card, chart
│   └── js/
│       ├── app.js                # Dynamic API communication, R engine tests & tabs
│       └── regression.js         # Phase 3: prediction flow, validation & SVG chart
│
├── backend/                      # Python Flask REST API
│   ├── app.py                    # Flask server entrypoint & CORS config
│   ├── requirements.txt          # Python dependencies (Flask, Flask-CORS)
│   ├── routes/                   # API Blueprint route definitions
│   │   ├── __init__.py           # Blueprint package initialization
│   │   ├── health.py             # Health check endpoint (/api/health)
│   │   ├── r_engine.py           # R engine execution endpoint (/api/r-engine/test)
│   │   ├── datasets.py           # Dataset validation endpoint (/api/datasets/validate)
│   │   └── regression.py         # Phase 3: predict / metrics / evaluation / schema
│   ├── services/                 # Business logic and ML orchestration services
│   │   ├── __init__.py           # Service package initializer
│   │   ├── r_runner.py           # R execution engine subprocess runner
│   │   ├── dataset_validator.py  # Dataset schema and data integrity validator
│   │   └── regression_service.py # Phase 3: input validation & model orchestration
│   └── database/                 # SQLite database storage directory (future)
│       └── .gitkeep
│
├── r_models/                     # R Machine Learning source scripts
│   ├── install_packages.R        # Automated R package installer
│   ├── test_engine.R             # R environment diagnostic probe
│   ├── regression/               # Linear regression (Phase 3)
│   │   ├── train.R               # Model training, evaluation & artifact generation
│   │   ├── predict.R             # Loads model.rds and returns a JSON prediction
│   │   ├── model.rds             # Serialised fitted model + training metadata
│   │   ├── metrics.json          # Real R2 / RMSE / MAE + coefficients
│   │   └── evaluation.json       # Held-out test actual vs predicted pairs
│   ├── decision_tree/            # Decision tree classification scripts (Phase 4)
│   ├── knn/                      # K-Nearest Neighbors scripts (Phase 5)
│   └── kmeans/                   # K-Means clustering scripts (Phase 6)
│
├── datasets/                     # Training and testing datasets (4 CSV files)
│   ├── housing.csv               # Property price regression dataset (120 rows)
│   ├── risk.csv                  # Financial risk classification dataset (120 rows)
│   ├── students.csv              # Student performance KNN dataset (120 rows)
│   └── customers.csv             # Customer segmentation K-Means dataset (120 rows)
│
├── tests/                        # Automated test suites
│   ├── test_backend.py           # 19 tests: health, R engine, dataset validation
│   ├── test_regression.py        # 40 tests: Phase 3 model, API, validation, failures
│   └── frontend_logic_test.js    # 50 tests: page logic, validation & chart rendering
│
├── .gitignore                    # Git pattern exclusion rules
└── README.md                     # Comprehensive platform documentation
```

---

## ⚙️ Installation & Setup Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.13)
- **R 4.0+** (Tested on R 4.6.1) — [Download from CRAN](https://cran.r-project.org/)
- **Git**
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

---

### Step 1: Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd AI-Insight-Hub
```

---

### Step 2: Configure Python Virtual Environment & Install Dependencies

#### On Windows:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Git Bash)
source .venv/Scripts/activate

# Or in Command Prompt:
.venv\Scripts\activate

# Install requirements
pip install -r backend/requirements.txt
```

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

---

### Step 3: Install Required R Packages

Execute the automated R package installer script:
```bash
Rscript r_models/install_packages.R
```

Or from within an interactive R terminal:
```R
install.packages(c("rpart", "class", "cluster", "ggplot2", "jsonlite", "readr"), repos="https://cloud.r-project.org")
```

---

### Step 4: Train the Linear Regression Model (Phase 3)

Train the model and generate its artifacts (`model.rds`, `metrics.json`, `evaluation.json`):
```bash
Rscript r_models/regression/train.R
```

Expected output:
```text
[4/8] Split dataset: 96 training rows / 24 test rows (80% / 20%)
[7/8] Test metrics  -> R2 = 0.9967 | RMSE = 5646.30 | MAE = 5005.34
PHASE 3 LINEAR REGRESSION TRAINING COMPLETE
```

> The Phase 3 model artifacts are committed to the repository, so this step only needs to
> be repeated if you change the dataset or the model formula.

---

### Step 5: Run the Automated Test Suite

Verify that all endpoints, services, dataset validators, R execution bridges, and the
Linear Regression model pass:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected output:
```text
Ran 59 tests in 2.7s
OK
```

With the Flask backend running, also run the frontend logic suite:
```bash
node tests/frontend_logic_test.js
```

Expected output:
```text
=== Results: 50 passed, 0 failed ===
```

---

### Step 6: Start the Flask Backend Server

```bash
python backend/app.py
```

---

### Step 7: Test the REST API Endpoints

#### 1. System Health Check:
```bash
curl http://127.0.0.1:5000/api/health
```
Output:
```json
{
  "message": "AI Insight Hub API is running",
  "r_engine": "online",
  "status": "online",
  "timestamp": "2026-09-25T12:07:21.858675+00:00",
  "version": "1.0.0"
}
```

#### 2. R Engine Execution Test:
```bash
curl -X POST http://127.0.0.1:5000/api/r-engine/test -H "Content-Type: application/json" -d "{\"test\":\"ping\"}"
```
Output:
```json
{
  "data": {
    "engine": "R",
    "message": "R engine executed successfully",
    "packages": {
      "class": true,
      "cluster": true,
      "ggplot2": true,
      "jsonlite": true,
      "readr": true,
      "rpart": true
    },
    "platform": "x86_64-w64-mingw32",
    "r_version": "4.6.1 ()",
    "success": true
  },
  "engine": "R",
  "execution_time_ms": 628.13,
  "message": "R engine executed successfully",
  "return_code": 0,
  "success": true
}
```

#### 3. Dataset Validation Check:
```bash
curl http://127.0.0.1:5000/api/datasets/validate
```

#### 4. Linear Regression Prediction (Phase 3):
```bash
curl -X POST http://127.0.0.1:5000/api/regression/predict \
  -H "Content-Type: application/json" \
  -d '{"area":1500,"bedrooms":3,"bathrooms":2,"location_score":8,"property_age":5}'
```
Output:
```json
{
  "success": true,
  "model": "Linear Regression",
  "algorithm": "stats::lm (Ordinary Least Squares)",
  "prediction": 355005.39,
  "unit": "INR",
  "r2": 0.9967,
  "rmse": 5646.3,
  "mae": 5005.34
}
```

#### 5. Linear Regression Model Metrics (Phase 3):
```bash
curl http://127.0.0.1:5000/api/regression/metrics
```

#### 6. Actual vs Predicted Test Data (Phase 3):
```bash
curl http://127.0.0.1:5000/api/regression/evaluation
```

---

### Step 8: Launch the Frontend

Open `frontend/index.html` in your browser, or start a local static server:
```bash
python -m http.server 8000
```
Then navigate to: `http://127.0.0.1:8000/frontend/`

**Interactive Dashboard Actions:**
- Click **"TEST R ENGINE"** in the R Machine Learning Engine panel to trigger `POST /api/r-engine/test` and view real-time execution metrics.
- Toggle between console tabs (**GET /api/health**, **POST /api/r-engine/test**, and **GET /api/datasets/validate**) to inspect formatted JSON payloads.
- Click **"OPEN MODEL"** on the Linear Regression card to open the **Property Price Predictor** page at `frontend/regression.html`.

**Property Price Predictor Actions:**
- Enter property attributes and click **"PREDICT PROPERTY PRICE"** to run the real R model.
- View the live R², RMSE and MAE tiles (loaded from the trained model).
- Hover the Actual vs Predicted scatter plot to inspect individual test records.

---

## 🗺️ Future Development Roadmap

| Phase | Milestone Name | Status | Scope & Deliverables |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **Foundation & API Integration** | ✅ Complete | Directory layout, Flask REST server, health telemetry, glassmorphism dashboard, dynamic status monitoring. |
| **Phase 2** | **R Execution Bridge & Datasets** | ✅ Complete | Subprocess RRunner, package verification, test R script, test endpoint (`/api/r-engine/test`), 4 datasets (480 records), dataset validator, test suite. |
| **Phase 3** | **Linear Regression** | ✅ Complete | Property price model in R (`lm`), training pipeline with real R²/RMSE/MAE, persisted `model.rds`, prediction/metrics/evaluation API, interactive predictor page with scatter chart. |
| **Phase 4** | **Decision Tree** | ⏳ Planned | Financial risk classification model in R (`rpart`), tree visualization, risk scoring matrix. |
| **Phase 5** | **KNN Classification** | ⏳ Planned | Student performance model in R (`class::knn`), distance-weighted inference, dynamic K-tuning. |
| **Phase 6** | **K-Means Clustering** | ⏳ Planned | Customer segmentation model in R (`kmeans`), elbow method optimization, 2D centroid scatter visualizer. |

---

## 👥 Academic & Training Attribution

- **Program:** GCF Training
- **Conducted by:** Ethnotech
- **Institution:** Parul University
- **Developer:** Rakesh Rabadiya

---

## 📄 License
This project is licensed for academic, educational, and training purposes within the GCF Training curriculum.

