# AI Insight Hub — Multi-Model Machine Learning Analytics Platform

[![Platform](https://img.shields.io/badge/Platform-AI%20Insight%20Hub-00f0ff.svg)](#)
[![Phase](https://img.shields.io/badge/Status-All%206%20Phases%20Complete%20%C2%B7%204%20Models-10b981.svg)](#)
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

### **Phase 5: K-Nearest Neighbors — Student Performance Prediction (ACTIVE / COMPLETED)**

Phase 5 adds the platform's **second classification model**, this time a *lazy*,
instance-based one: rather than learning parameters, a KNN model simply memorises its
training set and classifies a new record by looking at the students most similar to it.
The model is trained in R with `class::knn()`, persisted to disk, and served through the
same Flask → `Rscript` bridge as Phases 3 and 4.

> ⚠️ **Educational Notice:** This is an **educational machine learning implementation**.
> The model is trained on a synthetic dataset for academic demonstration. It is **not**
> a real assessment tool and must not be used to judge an actual student.

- ✅ **Real `class::knn()` Model:** genuine Euclidean-distance KNN with majority voting.
- ✅ **No Retraining at Inference:** `predict.R` does `readRDS()` and reuses the stored
  scaled training matrix — it never refits anything.
- ✅ **Data-Driven K:** K is chosen from a grid by repeated cross-validation on the
  **training data only**, so the reported test metrics stay an honest estimate.
- ✅ **Feature Scaling:** every feature is standardised using training-set parameters
  that are saved and re-applied verbatim at prediction time.
- ✅ **Real Neighbours:** the K nearest training rows behind each prediction are returned
  with their actual classes and actual distances — and are self-checked against
  `class::knn()`'s own output before anything is written to disk.
- ✅ **Full Stack:** four API endpoints plus a Student Performance Predictor page that
  explains the vote, the neighbours and the model's real performance.

#### What a KNN Classifier Is, and Why One Here

KNN predicts nothing of its own. To classify a new record it measures the distance from
that record to every training row, takes the **K closest**, and returns their **majority
class**. There is no fitted coefficient, no decision boundary and no training-time
abstraction — which is exactly why the neighbours are the explanation: you can read the
reasoning straight off the model.

The task is multi-class student performance: `LOW`, `MEDIUM` or `HIGH`.

#### Dataset Features & Performance Classes

Source: `datasets/students.csv` — **120 rows, 5 numeric features, 1 class target.**

| Feature | Type | Training range | What it means |
| :--- | :--- | :--- | :--- |
| `study_hours` | numeric | 2.2 – 34.8 | Hours studied per week |
| `attendance` | numeric (0–100) | 56.7 – 99.9 | Class attendance percentage |
| `previous_score` | numeric (0–100) | 36.1 – 96.3 | Most recent assessment score |
| `assignments_completed` | numeric | 1 – 10 | Assignments submitted |
| `practical_score` | numeric (0–100) | 40.4 – 100.0 | Practical / lab assessment score |
| **`performance`** | **class** | — | **`LOW` / `MEDIUM` / `HIGH`** |

**Observed class distribution: MEDIUM = 78, HIGH = 41, LOW = 1.**

> ⚠️ **This dataset is severely imbalanced, and it materially limits the model.**
> Two consequences are reported throughout rather than hidden:
>
> 1. **The single LOW row must be used for training** (a class needs at least one
>    training row), so the held-out test set contains **no LOW rows at all**.
> 2. **At the selected K = 5, a majority vote needs 3 of the 5 nearest rows to be LOW.**
>    Only one LOW row exists, so **LOW is mathematically unreachable** — the model can
>    only ever return `MEDIUM` or `HIGH`.
>
> Both facts are computed at training time and published in `metrics.json` under
> `class_distribution_note`, and shown on the prediction page.

#### Why Feature Scaling Is Required

KNN compares records with a straight-line **Euclidean distance**, and that distance is
computed directly on the feature values. The five features are on completely different
scales:

| Feature | Standard deviation |
| :--- | --- |
| `assignments_completed` | 3.03 |
| `study_hours` | 9.28 |
| `previous_score` | 17.03 |
| `practical_score` | 18.25 |

Left unscaled, the two large-variance columns (`previous_score`, `practical_score`)
would dominate every distance calculation — roughly **6× the spread** of
`assignments_completed` — while `assignments_completed` would contribute almost nothing
to "who is this student most like?". The model would silently be a function of two
features out of five.

Every feature is therefore **standardised (z-score) using training-set parameters only**:

```
scaled_value = (value - training_mean) / training_standard_deviation
```

Fitting the centre and scale on the *training half only* is deliberate — computing them
over the full dataset would leak test information into training. Those exact parameters
are saved in `model.rds` and re-applied verbatim by `predict.R`, so the incoming record
and the stored training matrix are always on the same footing. After scaling, all five
features have mean 0 and standard deviation 1, and each contributes what it should.

#### Training Process

`r_models/knn/train.R` runs eleven verified steps:

1. **Load** `datasets/students.csv` (120 rows).
2. **Validate** the six required columns are present.
3. **Clean** — coerce features to numeric, trim/upper-case the target, drop incomplete,
   domain-invalid (percentages outside 0–100) and unknown-class rows, and count every
   drop. Any range check guards against `sample()`'s length-1 gotcha, which would
   otherwise corrupt the split for the single LOW row.
4. **Split** stratified by class: **97 training / 23 test rows**, then assert the two
   halves neither overlap nor drop a row.
5. **Scale** the features from training-set centre/scale only.
6. **Choose K** by 10 repeats of 5-fold stratified cross-validation on the **training
   data only**, with the one-standard-error rule.
7. **Train** the real classifier with `class::knn(train_x, test_x, train_y, k = K)`.
8. **Predict** on the held-out test set.
9. **Extract the K nearest rows** per test record with the same Euclidean metric
   `class::knn()` uses internally, then **self-check** that the majority class of those
   extracted rows reproduces `class::knn()`'s prediction *and* its `prob` attribute.
   The run aborts rather than write misleading neighbour data.
10. **Compute metrics** from the confusion matrix built out of those real predictions.
11. **Save** `model.rds` and `metrics.json`.

#### K Selection — How K Was Chosen

K is **not** hand-picked and the test set is **not** consulted. Each candidate was scored
by repeated stratified cross-validation over the training data:

| K | Mean CV Macro F1 | Std. Error | Folds |
| :--- | ---: | ---: | ---: |
| 3 | 0.5428 | ± 0.0086 | 50 |
| **5** | **0.5611** | **± 0.0079** | **50** ✅ **selected** |
| 7 | 0.5575 | ± 0.0087 | 50 |
| 9 | 0.5490 | ± 0.0084 | 50 |

The best mean CV F1 (0.5611) defines a one-standard-error threshold of **0.5532**. Every
K from 3 to 9 clears that bar, so the 1-SE rule takes the **smallest K statistically
indistinguishable from the best: K = 5**.

> **Note on honesty:** picking K by its *test* accuracy would have selected K = 3, 7 or 9
> (0.9565 rather than 0.9130), but that uses the test set for model selection and makes
> the reported test score an over-estimate. K was chosen on training data only, and the
> 0.9130 below is an honest held-out number.

#### Evaluation Metrics (Real, From the Trained Model)

| Metric | Value | How it is computed |
| :--- | ---: | :--- |
| **Accuracy** | **0.9130** | 21 of 23 held-out test records correct |
| **Precision** | **0.6028** | Macro average across all 3 declared tiers |
| **Recall** | **0.6028** | Macro average across all 3 declared tiers |
| **F1 Score** | **0.6028** | Macro average across all 3 declared tiers |
| F1 (supported tiers only) | 0.9042 | Macro across MEDIUM + HIGH only |
| Majority-class baseline | 0.6522 | Always predicting `MEDIUM` |

**Confusion matrix** (rows = actual, columns = predicted):

| actual \ predicted | LOW | MEDIUM | HIGH | Total |
| :--- | ---: | ---: | ---: | ---: |
| **LOW** | 0 | 0 | 0 | 0 |
| **MEDIUM** | 0 | 14 | 1 | 15 |
| **HIGH** | 0 | 1 | 7 | 8 |

> **Reading these numbers honestly.** Accuracy of 0.9130 is genuine, and it comfortably
> beats the 0.6522 majority-class baseline. But the **macro F1 of 0.6028 is the number to
> trust for a "how good is this?" judgement**, and the gap between the two is entirely
> explained by `LOW` scoring 0 — it has no test rows at all, so it can contribute
> neither true positives nor recall. The `0.9042` supported-only F1 shows how well the
> model actually does on the two tiers that can be evaluated. Both numbers are published;
> neither is hidden, and the class distribution is reported next to them.

#### Phase 5 API Endpoints

##### Prediction — `POST /api/knn/predict`

```json
{
  "study_hours": 6.5,
  "attendance": 88,
  "previous_score": 74,
  "assignments_completed": 9,
  "practical_score": 81
}
```

Real response (abridged — values are exactly what the R model returned):

```json
{
  "success": true,
  "model": "K-Nearest Neighbors",
  "algorithm": "class::knn (Euclidean distance, majority vote)",
  "prediction": "MEDIUM",
  "k": 5,
  "confidence": 0.8,
  "confidence_basis": "Share of the 5 nearest training neighbours belonging to the predicted class, as reported by class::knn() and re-derived from the neighbour list.",
  "neighbor_class_distribution": { "LOW": 0, "MEDIUM": 4, "HIGH": 1 },
  "neighbors": [
    {
      "neighbor": 1,
      "neighbor_class": "MEDIUM",
      "distance": 1.1951,
      "scaled_features": { "study_hours": 4.4, "attendance": 98.8, "previous_score": 75.5,
                           "assignments_completed": 8, "practical_score": 67.9 }
    }
  ],
  "scaling_applied": "standardisation (z-score) fitted on the training set",
  "warnings": []
}
```

**On the confidence value.** `class::knn()` returns a `prob` attribute defined as *the
share of the K nearest neighbours belonging to the winning class*. That is what
`confidence` reports, and `predict.R` re-derives the same share from the neighbour list it
built, **refusing to answer if the two disagree**. It is a real count of real neighbours
— not an invented model probability.

##### Metrics — `GET /api/knn/metrics`

Returns the real accuracy, macro precision/recall/F1, supported-only macro averages,
per-class metrics, the confusion matrix, the class-distribution note, the feature-scaling
parameters, the full K comparison and every held-out test prediction with its neighbours.

##### Config — `GET /api/knn/config`

Returns the selected K, the evaluated K grid, feature names with their real training
ranges, the class set, the scaling method and the class-balance caveat. Internal
artifacts (the stored training matrix, its labels and the numeric centre/scale values) are
deliberately **not** exposed.

##### Schema — `GET /api/knn/schema`

Returns each input field with its label and the valid range recorded at training time.

```bash
curl -X POST http://127.0.0.1:5000/api/knn/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours":6.5,"attendance":88,"previous_score":74,"assignments_completed":9,"practical_score":81}'
```

#### Nearest-Neighbour Explanation

Every prediction comes with the K training students that produced it. For the example
above, the five nearest rows to the submitted record were:

| Neighbor | Class | Distance |
| :--- | :--- | ---: |
| 1 | MEDIUM | 1.1951 |
| 2 | MEDIUM | 1.1961 |
| 3 | HIGH | 1.3771 |
| 4 | MEDIUM | 1.3786 |
| 5 | MEDIUM | 1.4381 |

Four MEDIUM rows beat one HIGH row, so the model answers **MEDIUM** with a neighbour
agreement of 4/5 = **80%**. Nothing else was consulted.

Those rows are recovered with the same Euclidean metric `class::knn()` uses internally,
via `||a−b||² = |a|² − 2a·b + |b|²`. That identity is arithmetically identical to summing
squared differences but avoids an n×p loop. Crucially, the training script **verifies** the
extraction against `class::knn()` itself: for every test record, the majority class of the
extracted neighbours must equal the model's prediction, and the derived winning-class
proportion must equal the model's own `prob` value. If either check fails, the training run
aborts rather than write metrics that misrepresent the model.

Distances are measured in **standardised feature space**, so they are comparable across
features but are not percentages.

#### Input Validation Rules

| Field | Rule |
| :--- | :--- |
| `study_hours` | `>= 0` |
| `attendance` | `0 – 100` |
| `previous_score` | `0 – 100` |
| `assignments_completed` | `>= 0` |
| `practical_score` | `0 – 100` |

Rejected with **HTTP 400** and a per-field error list: missing fields, empty values,
non-numeric values, negative counts/hours, percentages outside 0–100, unrecognised field
names, and malformed JSON. A non-JSON `Content-Type` returns **415**.

**Warnings, not errors (HTTP 200):** a logically valid value *outside the training range*
is still classified, and the response flags it in `warnings`. This follows the Phase 4
behaviour — a KNN model always has a nearest neighbour, so an out-of-range record is
answered, just as an extrapolation rather than an interpolation.

```json
"warnings": [
  "Study Hours (60) is outside the 2.2 - 34.8 range the model was trained on, so this prediction is extrapolated."
]
```

#### How to Retrain the Model

```bash
Rscript r_models/knn/train.R
```

This regenerates `model.rds` and `metrics.json` in place. The run is deterministic
(`set.seed(42)`), so an unchanged dataset reproduces the same split, the same selected K
and the same metrics exactly — retraining is only needed after editing
`datasets/students.csv` or changing the training logic.

To use different data or a different target, edit the `TARGET`, `FEATURES`, `CLASS_LEVELS`
and `K_GRID` constants at the top of `train.R`; the API's validation rules, the saved
feature ranges and the frontend schema all follow from those constants rather than being
duplicated by hand.

---

### **Phase 6: K-Means — Customer Segmentation (ACTIVE / COMPLETED)**

**This completes the platform: all four planned machine learning models are now implemented.**

```text
Phase 1 ✅    Phase 2 ✅    Phase 3 ✅    Phase 4 ✅    Phase 5 ✅    Phase 6 ✅
```

| # | Model | Learning type | Phase | Page |
| :--: | :--- | :--- | :--: | :--- |
| 1 | **Linear Regression** | supervised regression | 3 | `regression.html` |
| 2 | **Decision Tree** | supervised classification | 4 | `decision-tree.html` |
| 3 | **K-Nearest Neighbors** | supervised classification | 5 | `knn.html` |
| 4 | **K-Means** | **unsupervised clustering** | **6** | **`kmeans.html`** |

Phase 6 is the platform's **first unsupervised model**, and the only one with no target
column. The first three models answer *"what is the value / class?"*. K-Means answers a
different question: *"what groups exist in this data at all?"*

> ⚠️ **Educational Notice:** This is an **educational machine learning implementation**.
> The model is trained on a synthetic dataset for academic demonstration. The segments it
> finds are real clusters in that data, not validated commercial customer categories.

#### What Unsupervised Learning Is, and How It Differs Here

A supervised model is given the answer column and learns to reproduce it. Phases 3, 4 and
5 each had a `target` in their dataset — `price`, `risk`, `performance` — and were scored
by comparing their answers to it.

**Unsupervised learning starts with no answers at all.** There is no target column, no
train/test split and no accuracy score, because there is nothing to be right or wrong
about. Instead of predicting a label, the algorithm looks for structure that is already in
the data and reports it. For K-Means that structure is *groups of customers that behave
similarly to each other and differently to everyone else*.

This has a direct consequence for the metrics: **an unsupervised model cannot report
accuracy**, and this one does not pretend to. The honest quality measures are internal —
how tightly customers group (WSS) and how far apart the groups are (silhouette) — and
both are computed from the real fit.

#### Dataset & Features Used

Source: `datasets/customers.csv` — **120 rows, 4 numeric features, 1 identifier, no target.**

| Column | Type | Training range | Role |
| :--- | :--- | :--- | :--- |
| `customer_id` | string identifier | — | **Labels rows only — never clustered on** |
| `age` | numeric | 18 – 68 | Clustering feature |
| `annual_income` | numeric | 15.6 – 149.3 | Clustering feature |
| `spending_score` | numeric | 7 – 99 | Clustering feature |
| `purchase_frequency` | numeric | 2 – 50 | Clustering feature |

**Why `customer_id` is excluded.** It is a unique identifier with no behavioural meaning,
and it is ordered by registration. Clustering on it would recover *"which accounts were
opened first"* rather than finding customer types. It is used to label output rows and
nothing else. `train.R` asserts the identifier is absent from the feature list, and a test
enforces that at the API and frontend layers too.

#### Feature Scaling

K-Means minimises a **squared Euclidean distance**, and the four features are on very
different numeric ranges:

| Feature | Standard deviation |
| :--- | ---: |
| `age` | 11.21 |
| `annual_income` | 35.99 |
| `spending_score` | 27.95 |
| `purchase_frequency` | 14.79 |

Unscaled, `annual_income` would dominate every distance and the other three features would
barely influence the result. Every feature is therefore standardised to **mean 0 and
standard deviation 1** using the cleaned dataset, and `predict.R` applies the **identical**
saved transformation to an incoming customer. Without that, the two sides of the
comparison would be on different scales and the nearest-centre answer would be meaningless.

#### Selecting K: Elbow Method + Silhouette Score

K-Means needs to be told how many groups to look for. That number — **K** — is a modelling
decision, so it is chosen from evidence rather than picked to make the output look tidy.
Five candidates were fitted and scored:

```bash
Rscript r_models/kmeans/train.R
```

```text
[6/12] Scored 5 candidate K values (kmeans, nstart = 25)
      K             WSS   Silhouette  Between ratio  Stability WSS
      2        277.3377       0.4217         0.4174       284.2622
      3        160.4258       0.4358         0.6630       168.2142
      4        126.2634       0.4390         0.7347       127.8777
      5         94.1189       0.3987         0.8023       111.1381
      6         83.4317       0.3584         0.8247        93.7305
      Elbow (largest proportional WSS drop) -> K = 5
      Best average silhouette width          -> K = 4
```

**The Elbow method.** WSS always falls as K rises — splitting a cluster in two can only
reduce the sum of squared distances — so the lowest WSS is always the largest K and WSS
alone cannot choose. The *elbow* is where the curve **flattens**: the K whose proportional
reduction in WSS is the largest. Here that is **K = 5**.

**The Silhouette score.** For each customer, `s(i) = (b − a) / max(a, b)`, where `a` is the
mean distance to the other members of its own cluster and `b` is the smallest mean distance
to any *other* cluster. It approaches 1 when a customer is tightly packed and far from
everything else, and 0 when it sits on a boundary. Crucially — unlike WSS — **it is not
monotonic**, so its maximum is a genuine optimum. Here that is **K = 4** at 0.4390.

**How the two were combined.** The silhouette is the primary criterion, because it is the
only one of the two that can identify an optimum on its own; the elbow breaks a close tie.
The elbow's silhouette (0.3987) sits well below the peak (0.4390), so preferring it would
mean trading measurable separation for a curve-shape heuristic. **K = 4 was selected**, and
the full reasoning is written to `metrics.json` under `k_selection.reason` so the decision
can be audited rather than taken on trust.

Each K is additionally fitted 10 more times from independent seeds, and the spread of those
restarts is reported as `stability_wss_sd` — a K whose solution depends on a lucky restart
is not a trustworthy segmentation.

#### Selected K = 4: The Cluster Profiles

| Cluster | Size | Share | Generated label | Centre (original units) |
| :--: | --: | --: | :--- | :--- |
| 1 | 37 | 30.8% | Younger / Lower Income | age 30.9, income 47.3, spend 68.3, freq 23.3 |
| 2 | 18 | 15.0% | Low Spending / Rare Buyers | age 38.4, income 63.6, spend 23.6, freq 10.6 |
| 3 | 14 | 11.7% | Older / Lower Income / Low Spending / Rare Buyers | age 59.6, income 35.6, spend 20.6, freq 4.6 |
| 4 | 51 | 42.5% | High Income / High Spending / Frequent Buyers | age 44.2, income 102.5, spend 77.9, freq 37.2 |

**These labels are derived, not written by hand.** `train.R` compares each cluster's centre
against the population mean *feature by feature* and marks a feature as separating when the
centre sits at least **0.5 standard deviations** away. The label is then assembled from
those actual descriptors. A cluster separated on fewer than two features is genuinely
ambiguous, so it falls back to the neutral name `Cluster N` rather than inventing a
persona — a test asserts that relationship in both directions.

The dataset's headline numbers: **WSS = 126.2634**, **between-cluster ratio = 0.7347**
(73.5% of the total variation lies *between* clusters), **mean silhouette = 0.4390**,
across **120 clustered / 0 dropped** rows.

#### Phase 6 API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/kmeans/predict` | Assign one customer to the nearest trained cluster centre |
| `GET` | `/api/kmeans/metrics` | WSS, silhouette, cluster sizes, centres, K evaluation, selection reasoning |
| `GET` | `/api/kmeans/clusters` | Every customer's real cluster assignment, plus projected centres |
| `GET` | `/api/kmeans/profiles` | Per-cluster size, feature means, centres and generated labels |
| `GET` | `/api/kmeans/config` | Selected K, K grid, feature names, real training ranges, identifier held out |
| `GET` | `/api/kmeans/schema` | Each feature's label and real training range, for the frontend form |

##### Customer Assignment — `POST /api/kmeans/predict`

K-Means does not *predict* a class, so this endpoint **assigns**: it scales the incoming
customer with the model's saved scaler, measures the Euclidean distance to all four
centres, and returns the nearest one. That nearest-centre rule is exactly what
`stats::kmeans()` used on its own training rows, so the assignment and the model cannot
disagree.

Request:
```json
{
  "age": 42,
  "annual_income": 85,
  "spending_score": 55,
  "purchase_frequency": 28
}
```

```bash
curl -X POST http://127.0.0.1:5000/api/kmeans/predict \
  -H "Content-Type: application/json" \
  -d '{"age":42,"annual_income":85,"spending_score":55,"purchase_frequency":28}'
```

Real response (abridged — exactly what the R model returned):

```json
{
  "success": true,
  "model": "K-Means",
  "cluster": 4,
  "cluster_label": "High Income / High Spending / Frequent Buyers",
  "cluster_size": 51,
  "segment": "Core",
  "distance": 1.1572,
  "distance_original_units": 30.3991,
  "runner_up_distance": 1.8501,
  "separation_ratio": 0.6255,
  "distance_basis": "Euclidean distance in standardised feature space, the same space kmeans() clustered in",
  "distances": {
    "Cluster 1": 1.8501,
    "Cluster 2": 2.2807,
    "Cluster 3": 2.6957,
    "Cluster 4": 1.1572
  },
  "scaled_inputs": { "age": 0.22, "annual_income": 0.4674,
                     "spending_score": -0.2158, "purchase_frequency": 0.1572 },
  "warnings": []
}
```

The `distances` map is the full ranking, so the answer is auditable: cluster 4 is nearest at
1.1572 and the runner-up is 1.8501 away. The `separation_ratio` (0.6255) is that distance
over the runner-up's — well below 1, so the nearest centre is clearly the closest rather
than a near-tie.

**No retraining at inference.** `predict.R` does `readRDS()` and reuses the stored centres
and scaler. A request costs the same whether the model is a day old or a year old.

#### Visualization Explanation

The clustering is genuinely **four-dimensional** (four features), which no screen can show.
The dashboard therefore projects the customers onto their **two leading principal
components** (PC1 and PC2) and draws the scatter plot there.

> ⚠️ **The clustering itself is performed on all four scaled features — never on the
> components.** The PCA is computed *for visualisation only*, and `metrics.json` records
> that explicitly in `visualization.purpose`. Replacing K-Means with PCA clustering would
> silently change the model into a different, lower-variance one; this implementation does
> not do that. PC1 explains 55.0% of the variance and PC2 a further 30.8%, so the two
> components together show 85.8% of it.

Consequently, the distances shown in the scatter plot are **PCA** distances, whereas the
`distance` reported under Customer Segmentation is the **four-feature** distance the model
actually minimises. The page states this on the plot itself so the two are never confused.
Each cluster centre is drawn as a labelled, ringed marker with a halo, so it reads as an
average rather than a data point.

#### How to Retrain the Model

```bash
Rscript r_models/kmeans/train.R
```

This rewrites `model.rds`, `metrics.json`, `clusters.json` and `profiles.json` in place, and
the API picks up the new model on the very next request — no server restart needed. The run
is deterministic
(`set.seed(42)` with `nstart = 25`); `train.R` re-fits the model a second time under the same
seed and **refuses to write anything** if the assignment differs, so the artifacts are
verified reproducible rather than assumed to be.

Expected output:
```text
[7/12] Selected K = 4 - The elbow method suggests K = 5 but its average silhouette width
       (0.3987) is materially below the peak of 0.4390 at K = 4. WSS always falls as K rises,
       so the elbow is the weaker criterion here, and the silhouette maximum is taken.
[8/12] Final kmeans() model: K = 4, WSS = 126.2634, between-cluster ratio = 0.7347, mean silhouette = 0.4390
      Cluster sizes: 1=37, 2=18, 3=14, 4=51
      Reproducible under the same seed: TRUE
PHASE 6 K-MEANS TRAINING COMPLETE
```

To use different data or a different feature set, edit the `IDENTIFIER_COLUMN`, `FEATURES`,
`K_GRID`, `N_START` and `RANDOM_SEED` constants at the top of `train.R`; the API's validation
rules, the saved feature ranges and the frontend schema all follow from those constants.

#### How to Test the Model

```bash
# Phase 6 suite (artifacts, assignment, K evaluation, profiles, failure handling)
python -m unittest tests.test_kmeans -v

# Frontend behaviour suite (form, cards, profiles, scatter, elbow chart) — 119 assertions
# Requires the Flask backend to be running on 127.0.0.1:5000
node tests/kmeans_frontend_test.js
```

---

### **Phase 4: Decision Tree — Financial Risk Classification (ACTIVE / COMPLETED)**

Phase 4 delivers the platform's **first classification model**. A real CART Decision Tree is
grown in R with `rpart`, persisted to disk, and served through the same Flask → `Rscript`
bridge Phase 2 built and Phase 3 proved, this time classifying a financial applicant into a
risk tier.

- ✅ **Real `rpart` Model:** `rpart::rpart(..., method = "class")` — CART, Gini impurity.
- ✅ **No Retraining at Inference:** `predict.R` does `readRDS()` and calls `predict()`.
- ✅ **Real Metrics:** accuracy, precision, recall, F1 and the confusion matrix are all
  computed from held-out test-set predictions, never hard-coded.
- ✅ **Real Tree Exposed:** `tree.json` holds the node-by-node structure of the *fitted*
  model, self-checked to route records identically to `predict()`.
- ✅ **Full Stack:** API endpoints, an interactive Financial Risk Analyzer page, and a
  decision-tree visualisation — all driven by live model output.

#### What a Decision Tree Is, and Why One Here

A decision tree learns a chain of `if/then` rules by repeatedly picking the feature split
that best separates the classes. For this problem that means it answers questions like:

```text
Is existing_loans < 1.5 ?
├── yes → Is credit_score >= 540.5 ?
│         ├── yes → Is employment_years < 20 ? → MEDIUM
│         └── no  → MEDIUM
└── no  → HIGH
```

That matters here because the model is a **classifier**, not a regressor. Linear Regression
(Phase 3) predicts a continuous price; `rpart` with `method = "class"` predicts one of three
discrete classes, and returns the class probability for each — which is where the
confidence shown in the UI comes from.

#### Dataset Features & Risk Classes

Dataset: **`datasets/risk.csv`** (120 rows, 0 rows dropped as invalid)

| Feature | Type | Description | Trained range |
| :--- | :--- | :--- | :--- |
| `age` | numeric | Applicant age in years | 21 – 65 |
| `income` | numeric | Annual income (INR) | 23,887.75 – 144,947.41 |
| `credit_score` | numeric | Credit score | 401 – 845 |
| `existing_loans` | numeric | Number of existing loans | 0 – 5 |
| `employment_years` | numeric | Years employed | 0 – 30 |
| **`risk`** | **factor (target)** | **The predicted class** | — |

**Target classes:** `LOW`, `MEDIUM`, `HIGH` — held in a factor with that fixed level order
so the class-probability columns always mean the same thing.

The dataset is **heavily imbalanced** (`HIGH` = 81, `MEDIUM` = 32, `LOW` = 7). That single
fact drives most of the design decisions below, so it is worth stating plainly: on an
imbalanced dataset a model that ignores its inputs and always answers "HIGH" already scores
about 70% accuracy. Accuracy alone is therefore not a useful headline here, which is why the
page also shows balanced accuracy and the full confusion matrix.

#### Training Process

`r_models/decision_tree/train.R` performs these steps, in order:

1. **Load** `datasets/risk.csv`; **validate** that all six required columns exist.
2. **Clean**: coerce every feature to numeric, drop incomplete rows, drop rows violating the
   domain rules, drop rows whose target is outside the three known classes — each count is
   reported, never silently discarded.
3. **Factor** the target with the fixed `LOW`/`MEDIUM`/`HIGH` level order.
4. **Stratified 80/20 split** with a fixed seed (`set.seed(42)`). The split is done *inside
   each class*, so every risk tier appears on both sides — a naive random split can leave
   `LOW` out of the test set entirely, and then its precision and recall cannot be computed
   at all.
5. **Cross-validate** 30 `(maxdepth, minsplit)` combinations — 10 repeats of 5 stratified
   folds — **on the training data only**, and select by the one-standard-error rule
   (simplest tree within 1 SE of the best mean macro-F1). This keeps the test set an honest
   estimate rather than something the hyperparameters were tuned against.
6. **Train** the final model on the full training set with the selected settings.
7. **Predict** on the held-out test set and compute the real metrics.
8. **Extract** the fitted tree node by node, then **self-check** it two ways before writing
   anything: the hand-extracted routing must agree with `predict()` on 100% of test
   records, and the extracted leaf probabilities must match `predict(type = "prob")` on
   100% of them. If either check fails the script refuses to write `tree.json`, because a
   decorative tree would misrepresent the model.
9. **Save** `model.rds`, `metrics.json` and `tree.json`.

#### Evaluation Metrics (Real, From the Trained Model)

Seed 42, 97 training rows / 23 test rows, confusion matrix read as rows = actual,
columns = predicted:

| | LOW | MEDIUM | HIGH | Total |
| :--- | ---: | ---: | ---: | ---: |
| **LOW** | 0 | 1 | 0 | 1 |
| **MEDIUM** | 1 | 5 | 0 | 6 |
| **HIGH** | 0 | 6 | 10 | 16 |

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Accuracy** | **0.6522** | 15 of 23 correct |
| **Precision** (macro) | **0.4722** | |
| **Recall** (macro) | **0.4861** | |
| **F1 Score** (macro) | **0.4416** | |
| Balanced accuracy | 0.4861 | mean of the per-class recalls |
| Majority-class baseline | 0.6957 | "always predict HIGH" |

> **Reading these honestly.** The model is *below* the majority-class baseline on accuracy.
> That is the real result, not a bug: with 7 `LOW` rows in 120, the test set contains a
> single `LOW` example, and the tree misclassifies it. The numbers are reported exactly as
> computed so the limitation is visible — this is an educational demonstration on a small,
> imbalanced dataset, not a production credit-scoring system, and it is not financial advice.

#### Phase 4 API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/decision-tree/predict` | Validate input, run `predict.R` against the saved model, return the risk class. |
| `GET` | `/api/decision-tree/metrics` | Real accuracy, precision, recall, F1, per-class metrics, confusion matrix and the full CV grid. |
| `GET` | `/api/decision-tree/tree` | The real fitted tree: every node, split rule, branch condition and class probability. |
| `GET` | `/api/decision-tree/schema` | Each input feature with the valid range recorded at training time. |

**Prediction example — request:**
```json
{
  "age": 32,
  "income": 750000,
  "credit_score": 735,
  "existing_loans": 1,
  "employment_years": 5
}
```

**Prediction example — response (abridged):**
```json
{
  "success": true,
  "model": "Decision Tree",
  "prediction": "MEDIUM",
  "confidence": 0.8,
  "class_probabilities": { "LOW": 0.15, "MEDIUM": 0.8, "HIGH": 0.05 },
  "decision_path": [
    { "node": 1, "condition": "existing_loans < 1.5" },
    { "node": 2, "condition": "credit_score >= 540.5" },
    { "node": 4, "condition": "employment_years < 20" },
    { "node": 9, "condition": "Reached leaf node 9 -> MEDIUM RISK" }
  ]
}
```

`confidence` is the model's own probability for the class it predicted, read from
`predict(type = "prob")`. It is reported only when R actually calculated one.

**Status codes for `/api/decision-tree/predict`:**

| Code | Meaning |
| :---: | :--- |
| `200` | Prediction produced successfully. |
| `400` | Invalid or missing input (with a per-field `errors` list). |
| `415` | Request was not sent as JSON. |
| `502` | The R engine failed to execute or returned an unreadable result. |
| `503` | The model has not been trained yet (`model.rds` missing). |

#### Input Validation Rules

Validation is enforced in **both** Python and R, so a request cannot bypass it by calling the
R script directly:

| Field | Rule |
| :--- | :--- |
| `age` | `> 0` |
| `income` | `> 0` |
| `credit_score` | within `300` – `900` |
| `existing_loans` | `>= 0` |
| `employment_years` | `>= 0` |

Missing fields, empty strings, non-numeric values and unknown field names are all rejected;
booleans are refused rather than being coerced to 1 or 0.

**Out-of-range values are a warning, not an error.** A value outside the range the model was
trained on (`income` above 144,947, for instance) is still classified — a CART tree always
routes a record down *some* branch — and the response carries a `warnings` array naming the
field and its trained range:

```json
"warnings": [
  {
    "field": "income",
    "value": 750000.0,
    "trained_min": 23887.75,
    "trained_max": 144947.41,
    "message": "Annual Income (750000) is outside the 23887.8 – 144947 range the model was trained on, so this prediction is extrapolated."
  }
]
```

This keeps the documented example payload working while still telling the user the answer
is an extrapolation rather than an interpolation.

#### How the Tree Structure Is Exposed

`GET /api/decision-tree/tree` returns the fitted tree as a nested structure — each node
carries its `id`, `variable`, `cut` (the threshold), the `left_condition` and
`right_condition` in human-readable form, its `class_counts`, `class_probabilities` and
`samples`, and its two children (`2n` and `2n+1`, rpart's binary layout).

The frontend renders this as a real decision diagram — one card per node, connected by
labelled branches — plus the variable-importance ranking. Because the payload is the fitted
model rather than a hand-drawn illustration, the diagram changes whenever the model is
retrained.

The **decision path** returned with every prediction is the same structure walked for one
specific applicant: the chain of rules that record actually satisfied, ending at the leaf
that produced its class. It is a trace of the model, not a description of it.

#### How to Retrain the Model

```bash
Rscript r_models/decision_tree/train.R
```

This regenerates `model.rds`, `metrics.json` and `tree.json` in place. The run is
deterministic (`set.seed(42)`), so an unchanged dataset reproduces the same tree and the
same metrics exactly — retraining is only needed after editing `datasets/risk.csv` or
changing the training logic.

To use different data or a different target, edit the `TARGET`, `FEATURES` and
`CLASS_LEVELS` constants at the top of `train.R`; the API's validation rules, the saved
feature ranges and the frontend schema all follow from those constants rather than being
duplicated by hand.

---

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
│   ├── decision-tree.html        # Phase 4: Financial Risk Analyzer page
│   ├── knn.html                  # Phase 5: Student Performance Predictor page
│   ├── kmeans.html               # Phase 6: Customer Segmentation page
│   ├── css/
│   │   ├── style.css             # Main styling, design tokens & glassmorphism
│   │   ├── responsive.css        # Adaptive mobile & tablet breakpoints
│   │   ├── regression.css        # Phase 3: predictor form, result card, chart
│   │   ├── decision-tree.css     # Phase 4: risk form, result card, tree diagram
│   │   ├── knn.css               # Phase 5: performance form, result card, neighbours, K table
│   │   └── kmeans.css            # Phase 6: stat row, cluster cards, scatter, K chart, profiles
│   └── js/
│       ├── app.js                # Dynamic API communication, R engine tests & tabs
│       ├── regression.js         # Phase 3: prediction flow, validation & SVG chart
│       ├── decision-tree.js      # Phase 4: risk form, prediction & tree rendering
│       └── knn.js                # Phase 5: prediction flow, neighbour trace, K table & metrics
│
├── backend/                      # Python Flask REST API
│   ├── app.py                    # Flask server entrypoint & CORS config
│   ├── requirements.txt          # Python dependencies (Flask, Flask-CORS)
│   ├── routes/                   # API Blueprint route definitions
│   │   ├── __init__.py           # Blueprint package initialization
│   │   ├── health.py             # Health check endpoint (/api/health)
│   │   ├── r_engine.py           # R engine execution endpoint (/api/r-engine/test)
│   │   ├── datasets.py           # Dataset validation endpoint (/api/datasets/validate)
│   │   ├── regression.py         # Phase 3: predict / metrics / evaluation / schema
│   │   ├── decision_tree.py      # Phase 4: predict / metrics / tree / schema
│   │   └── knn.py                # Phase 5: predict / metrics / config / schema
│   ├── services/                 # Business logic and ML orchestration services
│   │   ├── __init__.py           # Service package initializer
│   │   ├── r_runner.py           # R execution engine subprocess runner
│   │   ├── dataset_validator.py  # Dataset schema and data integrity validator
│   │   ├── regression_service.py # Phase 3: input validation & model orchestration
│   │   ├── decision_tree_service.py # Phase 4: risk input validation & R orchestration
│   │   └── knn_service.py        # Phase 5: student input validation & R orchestration
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
│   ├── decision_tree/            # Decision tree classification (Phase 4)
│   │   ├── train.R               # Trains rpart CART tree, evaluates & writes artifacts
│   │   ├── predict.R             # Loads model.rds and returns a JSON risk prediction
│   │   ├── model.rds             # Serialised fitted rpart model + metadata + tree
│   │   ├── metrics.json          # Real accuracy / precision / recall / F1 + confusion matrix
│   │   └── tree.json             # Node-by-node structure of the fitted tree
│   ├── knn/                      # K-Nearest Neighbors classification (Phase 5)
│   │   ├── train.R               # Trains class::knn, selects K by CV & writes artifacts
│   │   ├── predict.R             # Loads model.rds, scales, classifies & returns neighbours
│   │   ├── model.rds             # Serialised scaled training matrix, labels, scaler & K
│   │   └── metrics.json          # Real accuracy / precision / recall / F1 + K comparison
│   └── kmeans/                   # K-Means clustering scripts (Phase 6)
│       ├── train.R               # Trains stats::kmeans, selects K by silhouette + elbow & writes artifacts
│       ├── predict.R             # Loads model.rds, applies the saved scaler, assigns the nearest centre
│       ├── model.rds             # Serialised kmeans object, centres, scaler, PCA basis & profiles
│       ├── metrics.json          # Real WSS, silhouette, K comparison, cluster profiles & PCA projection
│       ├── clusters.json         # Every customer with its real cluster, distances & projected position
│       └── profiles.json         # Per-cluster size, means, centres, relative position & label rule
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
│   ├── test_decision_tree.py     # 55 tests: Phase 4 model, API, validation, tree, failures
│   ├── test_knn.py               # 66 tests: Phase 5 model, API, validation, neighbours, failures
│   ├── test_kmeans.py            # 77 tests: Phase 6 model, K evaluation, assignment, profiles, failures
│   ├── frontend_logic_test.js    # 50 tests: Phase 3 page logic, validation & chart
│   ├── decision_tree_frontend_test.js  # 69 tests: Phase 4 form, result, metrics & tree UI
│   ├── knn_frontend_test.js      # 73 tests: Phase 5 form, neighbours, K table & metrics UI
│   ├── frontend_dom_contract_test.js   # 32 tests: every page's script/markup id contract
│   └── kmeans_frontend_test.js   # 119 tests: Phase 6 form, cards, profiles, scatter & elbow chart UI
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

### Step 4b: Train the Decision Tree Model (Phase 4)

Train the classifier and generate its artifacts (`model.rds`, `metrics.json`, `tree.json`):
```bash
Rscript r_models/decision_tree/train.R
```

Expected output:
```text
[4/11] Stratified split: 97 training rows / 23 test rows (80% / 20%)
[5/11] Cross-validated 30 (maxdepth, minsplit) combinations over 10 repeats of 5 stratified folds
      Selected by 1-SE rule: maxdepth = 3, minsplit = 5 (CV F1 = 0.6599 +/- 0.0248)
[8/11] Test metrics -> Accuracy = 0.6522 | Precision = 0.4722 | Recall = 0.4861 | F1 = 0.4416
[10/11] Tree routing self-check passed: extracted tree matches predict() on 23/23 test records
PHASE 4 DECISION TREE TRAINING COMPLETE
```

> The Phase 4 model artifacts are committed too, and the run is deterministic
> (`set.seed(42)`), so this only needs repeating after editing `datasets/risk.csv` or the
> training logic.

---

### Step 4c: Train the KNN Model (Phase 5)

Train the classifier and generate its artifacts (`model.rds`, `metrics.json`):
```bash
Rscript r_models/knn/train.R
```

Expected output:
```text
[4/11] Stratified split: 97 training rows / 23 test rows (81% / 19%)
      Training class counts: LOW=1, MEDIUM=63, HIGH=33
      Test class counts:     LOW=0, MEDIUM=15, HIGH=8
[6/11] Cross-validated K in {3, 5, 7, 9} over 10 repeats of 5 stratified folds (training data only)
      K =  5  mean CV macro F1 = 0.5611  (sd 0.0555, se 0.0079, 50 folds)   <- SELECTED
      Selected K = 5 by the one-standard-error rule
[9/11] Neighbour self-check passed: extracted neighbours reproduce
       class::knn()'s predictions AND probabilities on 23/23 test records
      Class reachability at K = 5 (a majority vote needs ceil(K/2) = 3 neighbours):
        LOW      training rows =   1  ->  UNREACHABLE
        MEDIUM   training rows =  63  ->  reachable
        HIGH     training rows =  33  ->  reachable
[10/11] Test metrics -> Accuracy = 0.9130 | Precision = 0.6028 | Recall = 0.6028 | F1 = 0.6028
PHASE 5 KNN TRAINING COMPLETE
```

> The Phase 5 model artifacts are committed too, and the run is deterministic
> (`set.seed(42)`), so this only needs repeating after editing `datasets/students.csv`
> or the training logic.

---

### Step 5: Run the Automated Test Suite

Verify that all endpoints, services, dataset validators, R execution bridges, the Linear
Regression model, the Decision Tree model and the KNN model pass:
```bash
python -m pytest tests/ -q
```

Expected output:
```text
180 passed
```

With the Flask backend running, also run the frontend logic suites:
```bash
node tests/frontend_logic_test.js
node tests/decision_tree_frontend_test.js
node tests/knn_frontend_test.js
```

Expected output:
```text
=== Results: 50 passed, 0 failed ===
=== Results: 69 passed, 0 failed ===
=== Results: 73 passed, 0 failed ===
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

#### 7. Financial Risk Prediction (Phase 4):
```bash
curl -X POST http://127.0.0.1:5000/api/decision-tree/predict \
  -H "Content-Type: application/json" \
  -d '{"age":32,"income":95000,"credit_score":735,"existing_loans":1,"employment_years":5}'
```

Expected response:
```json
{
  "success": true,
  "model": "Decision Tree",
  "prediction": "MEDIUM",
  "confidence": 0.8,
  "class_probabilities": { "LOW": 0.15, "MEDIUM": 0.8, "HIGH": 0.05 }
}
```

#### 8. Decision Tree Model Metrics (Phase 4):
```bash
curl http://127.0.0.1:5000/api/decision-tree/metrics
```

#### 9. Fitted Decision Tree Structure (Phase 4):
```bash
curl http://127.0.0.1:5000/api/decision-tree/tree
```

#### 10. Student Performance Prediction (Phase 5):
```bash
curl -X POST http://127.0.0.1:5000/api/knn/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours":6.5,"attendance":88,"previous_score":74,"assignments_completed":9,"practical_score":81}'
```

Expected response (abridged):
```json
{
  "success": true,
  "model": "K-Nearest Neighbors",
  "prediction": "MEDIUM",
  "k": 5,
  "confidence": 0.8,
  "neighbor_class_distribution": { "LOW": 0, "MEDIUM": 4, "HIGH": 1 },
  "neighbors": [ { "neighbor": 1, "neighbor_class": "MEDIUM", "distance": 1.1951 } ]
}
```

#### 11. KNN Model Metrics & Config (Phase 5):
```bash
curl http://127.0.0.1:5000/api/knn/metrics
curl http://127.0.0.1:5000/api/knn/config
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
- Click **"OPEN MODEL"** on the Decision Tree card to open the **Financial Risk Analyzer** page at `frontend/decision-tree.html`.
- Click **"OPEN MODEL"** on the KNN card to open the **Student Performance Predictor** page at `frontend/knn.html`.

**Property Price Predictor Actions:**
- Enter property attributes and click **"PREDICT PROPERTY PRICE"** to run the real R model.
- View the live R², RMSE and MAE tiles (loaded from the trained model).
- Hover the Actual vs Predicted scatter plot to inspect individual test records.

**Financial Risk Analyzer Actions:**
- Enter the applicant profile and click **"ANALYZE FINANCIAL RISK"** to run the real `rpart` classifier.
- View the live accuracy, precision, recall and F1 tiles plus the confusion matrix (all loaded from the trained model).
- Inspect the decision diagram — every node, threshold and branch comes from the fitted tree.
- Read the per-applicant decision path showing the exact rules that record satisfied.

**Student Performance Predictor Actions:**
- Enter the student profile and click **"PREDICT PERFORMANCE"** to run the real `class::knn` classifier.
- View the live accuracy, precision, recall and F1 tiles plus the confusion matrix (all loaded from the trained model).
- Read the **K nearest training students** that voted on the prediction, with their real classes and real distances.
- Inspect the **K comparison table** showing the cross-validation score for every candidate K and which one shipped.
- Note the on-page caveat explaining that this dataset's single LOW row makes LOW unreachable at K = 5.

---

## 🗺️ Future Development Roadmap

| Phase | Milestone Name | Status | Scope & Deliverables |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **Foundation & API Integration** | ✅ Complete | Directory layout, Flask REST server, health telemetry, glassmorphism dashboard, dynamic status monitoring. |
| **Phase 2** | **R Execution Bridge & Datasets** | ✅ Complete | Subprocess RRunner, package verification, test R script, test endpoint (`/api/r-engine/test`), 4 datasets (480 records), dataset validator, test suite. |
| **Phase 3** | **Linear Regression** | ✅ Complete | Property price model in R (`lm`), training pipeline with real R²/RMSE/MAE, persisted `model.rds`, prediction/metrics/evaluation API, interactive predictor page with scatter chart. |
| **Phase 4** | **Decision Tree** | ✅ Complete | Financial risk classification model in R (`rpart::rpart`, CART), stratified split + cross-validated depth selection, real accuracy/precision/recall/F1 + confusion matrix, persisted `model.rds`, prediction/metrics/tree API, decision-tree visualisation, Financial Risk Analyzer page. |
| **Phase 5** | **KNN Classification** | ✅ Complete | Student performance model in R (`class::knn`, Euclidean + majority vote), stratified split, training-only feature scaling, cross-validated K selection (K=5), real accuracy/precision/recall/F1 + confusion matrix + K comparison, persisted `model.rds`, predict/metrics/config/schema API, real nearest-neighbour trace, Student Performance Predictor page. |
| **Phase 6** | **K-Means Clustering** | ✅ Complete | Customer segmentation model in R (`stats::kmeans`), 4-feature standardisation, K selected from {2,3,4,5,6} by silhouette with the elbow as tie-breaker (**K = 4**, WSS = 126.26, silhouette = 0.4390), clusters 37/18/14/51 with labels derived from each measured centre, persisted `model.rds` + `metrics.json` + `clusters.json` + `profiles.json`, predict/metrics/clusters/profiles/config/schema API, hand-drawn canvas scatter plot with PCA projection, Customer Segmentation page. **All four planned models are now implemented.** |

---

## 👥 Academic & Training Attribution

- **Program:** GCF Training
- **Conducted by:** Ethnotech
- **Institution:** Parul University
- **Developer:** Rakesh Rabadiya

---

## 📄 License
This project is licensed for academic, educational, and training purposes within the GCF Training curriculum.

