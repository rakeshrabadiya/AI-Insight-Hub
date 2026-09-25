# AI Insight Hub — Multi-Model Machine Learning Analytics Platform

[![Platform](https://img.shields.io/badge/Platform-AI%20Insight%20Hub-00f0ff.svg)](#)
[![Phase](https://img.shields.io/badge/Status-Phase%202%20R%20Bridge%20Active-10b981.svg)](#)
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

### **Phase 2: R Engine Bridge & Dataset Foundation (ACTIVE / COMPLETED)**
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
- ⏳ *Note: The four specific R ML model algorithms (training & inference) will be implemented in Phases 3 to 6.*

---

## 📊 Dataset Registry & Schemas

All datasets are stored in `datasets/` and validated for integrity:

| Dataset File | Domain Application | Target ML Model | Records | Features / Columns |
| :--- | :--- | :--- | :---: | :--- |
| **`housing.csv`** | Property Price Prediction | Linear Regression (*Phase 3*) | 120 | `area`, `bedrooms`, `bathrooms`, `location_score`, `property_age`, `price` (Target) |
| **`risk.csv`** | Financial Risk Classification | Decision Tree (*Phase 4*) | 120 | `age`, `income`, `credit_score`, `existing_loans`, `employment_years`, `risk` (`LOW`, `MEDIUM`, `HIGH`) |
| **`students.csv`** | Student Performance | K-Nearest Neighbors (*Phase 5*) | 120 | `study_hours`, `attendance`, `previous_score`, `assignments_completed`, `practical_score`, `performance` (`LOW`, `MEDIUM`, `HIGH`) |
| **`customers.csv`** | Customer Segmentation | K-Means Clustering (*Phase 6*) | 120 | `customer_id`, `age`, `annual_income`, `spending_score`, `purchase_frequency` *(No pre-assigned labels)* |

---

## 🏗️ System & R Engine Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Browser Client (Frontend)                  │
│       HTML5 Dashboard  •  CSS3 Glassmorphism  •  Vanilla JS   │
│         [Ping API Health]      [Test R Engine Button]         │
└───────────────────────────────┬───────────────────────────────┘
                                │
                         HTTP / JSON REST
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                    Python Flask REST Backend                  │
│   app.py  •  routes/health.py  •  routes/r_engine.py          │
│   services/r_runner.py  •  services/dataset_validator.py      │
└───────────────────────────────┬───────────────────────────────┘
                                │
                 Subprocess IPC (stdin / args / stdout)
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                    R Machine Learning Subsystem               │
│   Rscript.exe (v4.6.1)  •  r_models/test_engine.R             │
│   Packages: rpart • class • cluster • ggplot2 • jsonlite      │
│   Datasets: datasets/housing.csv, risk.csv, students.csv, ... │
└───────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```text
AI-Insight-Hub/
├── frontend/                     # Pure client-side application
│   ├── index.html                # Main dashboard UI with R test trigger
│   ├── css/
│   │   ├── style.css             # Main styling, design tokens & glassmorphism
│   │   └── responsive.css        # Adaptive mobile & tablet breakpoints
│   └── js/
│       └── app.js                # Dynamic API communication, R engine tests & tabs
│
├── backend/                      # Python Flask REST API
│   ├── app.py                    # Flask server entrypoint & CORS config
│   ├── requirements.txt          # Python dependencies (Flask, Flask-CORS)
│   ├── routes/                   # API Blueprint route definitions
│   │   ├── __init__.py           # Blueprint package initialization
│   │   ├── health.py             # Health check endpoint (/api/health)
│   │   ├── r_engine.py           # R engine execution endpoint (/api/r-engine/test)
│   │   └── datasets.py           # Dataset validation endpoint (/api/datasets/validate)
│   ├── services/                 # Business logic and ML orchestration services
│   │   ├── __init__.py           # Service package initializer
│   │   ├── r_runner.py           # R execution engine subprocess runner
│   │   └── dataset_validator.py  # Dataset schema and data integrity validator
│   └── database/                 # SQLite database storage directory (future)
│       └── .gitkeep
│
├── r_models/                     # R Machine Learning source scripts
│   ├── install_packages.R        # Automated R package installer
│   ├── test_engine.R             # R environment diagnostic probe
│   ├── regression/               # Linear regression scripts (Phase 3)
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
├── tests/                        # Backend & R Engine Automated Test Suite
│   └── test_backend.py           # 19 Unit tests covering health, R engine, datasets
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

### Step 4: Run the Automated Test Suite

Verify that all endpoints, services, dataset validators, and R execution bridges pass:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected output:
```text
Ran 19 tests in 1.5s
OK
```

---

### Step 5: Start the Flask Backend Server

```bash
python backend/app.py
```

---

### Step 6: Test the REST API Endpoints

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

---

### Step 7: Launch the Frontend Dashboard

Open `frontend/index.html` in your browser, or start a local static server:
```bash
python -m http.server 8000
```
Then navigate to: `http://127.0.0.1:8000/frontend/`

**Interactive Dashboard Actions:**
- Click **"TEST R ENGINE"** in the R Machine Learning Engine panel to trigger `POST /api/r-engine/test` and view real-time execution metrics.
- Toggle between console tabs (**GET /api/health**, **POST /api/r-engine/test**, and **GET /api/datasets/validate**) to inspect formatted JSON payloads.

---

## 🗺️ Future Development Roadmap

| Phase | Milestone Name | Status | Scope & Deliverables |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **Foundation & API Integration** | ✅ Complete | Directory layout, Flask REST server, health telemetry, glassmorphism dashboard, dynamic status monitoring. |
| **Phase 2** | **R Execution Bridge & Datasets** | ✅ Complete | Subprocess RRunner, package verification, test R script, test endpoint (`/api/r-engine/test`), 4 datasets (480 records), dataset validator, test suite. |
| **Phase 3** | **Linear Regression** | ⏳ Planned | Property price prediction model in R (`lm`), multi-variable regression training & inference endpoint, interactive prediction UI. |
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

