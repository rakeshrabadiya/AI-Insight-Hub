# AI Insight Hub — Multi-Model Machine Learning Analytics Platform

[![Platform](https://img.shields.io/badge/Platform-AI%20Insight%20Hub-00f0ff.svg)](#)
[![Phase](https://img.shields.io/badge/Status-Phase%201%20Foundation%20Active-10b981.svg)](#)
[![Backend](https://img.shields.io/badge/Backend-Python%20Flask-3b82f6.svg)](#)
[![Frontend](https://img.shields.io/badge/Frontend-Vanilla%20HTML%20%2F%20CSS%20%2F%20JS-f59e0b.svg)](#)
[![ML-Engine](https://img.shields.io/badge/ML%20Engine-R%20(Upcoming)-6366f1.svg)](#)
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
| **Machine Learning** | R Statistical Language (*Integration in Phases 3–6*) | High-precision statistical modeling, native matrix computation, and ML libraries (`lm`, `rpart`, `class`, `cluster`). |
| **Database** | SQLite (*Planned for later persistence phases*) | Zero-configuration relational storage for model metrics and inference logs. |
| **Version Control** | Git & GitHub | Distributed version control and milestone-driven commit tracking. |

> **Strict Constraints:** No React, No Tailwind CSS, No Node.js, and No frontend build frameworks are used in this project.

---

## 🚀 Current Project Phase

### **Phase 1: Project Foundation & Flask-Frontend Integration (ACTIVE)**
- ✅ Complete directory structure initialized with modular routing and service placeholders.
- ✅ Python Flask backend operational with Cross-Origin Resource Sharing (CORS) and error handling.
- ✅ Dynamic REST API health endpoint (`GET /api/health`).
- ✅ Premium dark navy/cyan glassmorphism dashboard built with vanilla HTML/CSS/JS.
- ✅ Real-time frontend-to-backend communication, latency tracking, and interactive JSON diagnostics console.
- ⏳ *Note: Machine learning models in R are not yet executed in Phase 1 and are planned for subsequent phases.*

---

## 📊 Planned Machine Learning Models

The platform will incorporate four machine learning models across upcoming development phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AI INSIGHT HUB                                  │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Model Algorithm          │ Domain Application       │ Planned Milestone     │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ 1. Linear Regression     │ Property Price Prediction│ Phase 3               │
│ 2. Decision Tree         │ Financial Risk Analysis  │ Phase 4               │
│ 3. K-Nearest Neighbors   │ Student Performance      │ Phase 5               │
│ 4. K-Means Clustering    │ Customer Segmentation    │ Phase 6               │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

1. **Linear Regression (Phase 3):** Continuous valuation estimation for real estate properties based on area, rooms, amenities, and location score. Evaluated via $R^2$ and RMSE.
2. **Decision Tree (Phase 4):** Hierarchical risk profile classification for financial applicants using debt-to-income ratios, credit history, and collateral metrics.
3. **K-Nearest Neighbors - KNN (Phase 5):** Academic performance and retention tier prediction based on student study hours, attendance percentages, and test results.
4. **K-Means Clustering (Phase 6):** Unsupervised behavioral segmentation of customer cohorts using annual expenditure, transaction volume, and recency indicators.

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Browser Client (Frontend)                  │
│       HTML5 Dashboard  •  CSS3 Glassmorphism  •  Vanilla JS   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                         HTTP / JSON REST
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                    Python Flask REST API                      │
│        app.py  •  routes/health.py  •  services/ (Bridge)     │
└───────────────────────────────┬───────────────────────────────┘
                                │
                   Subprocess / rpy2 Execution
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                    R Machine Learning Engine                  │
│   regression/  •  decision_tree/  •  knn/  •  kmeans/         │
└───────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```text
AI-Insight-Hub/
├── frontend/                     # Pure client-side application
│   ├── index.html                # Main dashboard UI
│   ├── css/
│   │   ├── style.css             # Main styling, design tokens & glassmorphism
│   │   └── responsive.css        # Adaptive mobile & tablet breakpoints
│   └── js/
│       └── app.js                # Dynamic API communication & state management
│
├── backend/                      # Python Flask REST API
│   ├── app.py                    # Flask server entrypoint & CORS config
│   ├── requirements.txt          # Python dependencies (Flask, Flask-CORS)
│   ├── routes/                   # API Blueprint route definitions
│   │   ├── __init__.py           # Blueprint package initialization
│   │   └── health.py             # Health check endpoint (/api/health)
│   ├── services/                 # Business logic and ML orchestration services
│   │   └── __init__.py           # Service package initializer
│   └── database/                 # SQLite database storage directory (future)
│       └── .gitkeep
│
├── r_models/                     # R Machine Learning source scripts
│   ├── regression/               # Linear regression scripts (Phase 3)
│   ├── decision_tree/            # Decision tree classification scripts (Phase 4)
│   ├── knn/                      # K-Nearest Neighbors scripts (Phase 5)
│   └── kmeans/                   # K-Means clustering scripts (Phase 6)
│
├── datasets/                     # Training and testing datasets (CSV / RData)
│   └── .gitkeep
│
├── .gitignore                    # Git pattern exclusion rules
└── README.md                     # Comprehensive platform documentation
```

---

## ⚙️ Installation & Setup Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.13)
- **Git**
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

---

### Step 1: Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd AI-Insight-Hub
```

---

### Step 2: Create & Activate Python Virtual Environment

#### On Windows (Command Prompt / PowerShell / Git Bash):
```bash
# Create virtual environment named .venv
python -m venv .venv

# Activate in Command Prompt:
.venv\Scripts\activate

# Or in PowerShell:
.venv\Scripts\Activate.ps1

# Or in Git Bash:
source .venv/Scripts/activate
```

#### On macOS / Linux:
```bash
# Create virtual environment named .venv
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

---

### Step 3: Install Backend Dependencies
```bash
pip install -r backend/requirements.txt
```

Verify installed packages:
```bash
pip list
```

---

### Step 4: Start the Flask Backend Server
```bash
python backend/app.py
```

The Flask development server will launch at:
```text
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

### Step 5: Verify the REST API Endpoint
Open your browser or run curl:
```bash
curl http://127.0.0.1:5000/api/health
```

Expected JSON Response:
```json
{
  "message": "AI Insight Hub API is running",
  "r_engine": "not_connected",
  "status": "online",
  "timestamp": "2026-03-30T10:00:00.000000+00:00",
  "version": "1.0.0"
}
```

---

### Step 6: Launch and Use the Frontend

You can open the frontend using any of the following methods:

#### Option A: Direct File Open
Double-click `frontend/index.html` or open it directly in your browser:
```text
file:///C:/Users/.../AI-Insight-Hub/frontend/index.html
```

#### Option B: Simple HTTP Server (Recommended)
In a new terminal window:
```bash
# From the project root
python -m http.server 8000
```
Then navigate to:
```text
http://127.0.0.1:8000/frontend/
```

#### Option C: VS Code Live Server
Right-click `frontend/index.html` and click **"Open with Live Server"**.

---

## 🚦 Verifying Dynamic Online / Offline Status

1. **When Flask Server is Running:**
   - The top status pill shows `Flask API: ONLINE` with a glowing green pulse.
   - The Diagnostics Console displays HTTP Status `200 OK` with real-time latency in milliseconds.
   - The live JSON response is formatted and syntax-highlighted.

2. **When Flask Server is Stopped (`Ctrl + C`):**
   - The status pill dynamically updates to `Flask API: OFFLINE` with a red pulse.
   - An alert banner appears with instructions to restart the Flask server.
   - Clicking **"Ping API"** or **"Retry Connection"** attempts reconnection.

---

## 🗺️ Future Development Roadmap

| Phase | Milestone Name | Scope & Deliverables |
| :---: | :--- | :--- |
| **Phase 1** | **Foundation & API Integration** | Directory layout, Flask REST server, health telemetry, glassmorphism dashboard, dynamic status monitoring. |
| **Phase 2** | **R Execution Bridge** | Python-to-R subprocess/rpy2 execution pipeline, dataset loaders, error handling wrapper. |
| **Phase 3** | **Linear Regression** | Property price prediction dataset, R model fitting (`lm`), inference endpoint, interactive UI form. |
| **Phase 4** | **Decision Tree** | Financial risk classification dataset, R tree modeling (`rpart`), tree visualization, risk scoring matrix. |
| **Phase 5** | **KNN Classification** | Student performance dataset, R KNN modeling (`class::knn`), dynamic K parameter tuning, metrics display. |
| **Phase 6** | **K-Means Clustering** | Customer segmentation dataset, R clustering (`kmeans`), elbow method optimization, 2D scatter visualization. |

---

## 👥 Academic & Training Attribution

- **Program:** GCF Training
- **Conducted by:** Ethnotech
- **Institution:** Parul University
- **Developer:** Rakesh Rabadiya

---

## 📄 License
This project is licensed for academic, educational, and training purposes within the GCF Training curriculum.
