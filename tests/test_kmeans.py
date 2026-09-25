"""
AI Insight Hub — Phase 6 K-Means Test Suite

Tests real behaviour end to end:
  1. Dataset validation (the real customers.csv passes the shared validator)
  2. Training scripts exist on disk
  3. model.rds exists and is non-empty
  4. metrics.json exists and parses
  5. clusters.json exists and parses
  6. Selected K is a positive integer drawn from the evaluated K grid
  7. The fit produces exactly the selected K, no degenerate clusters
  8. Every usable record carries a real cluster assignment
  9. Cluster sizes sum exactly to the dataset size
 10. Metrics endpoint
 11. Clusters endpoint
 12. Profiles endpoint
 13. Valid customer assignment
 14. Invalid input is rejected
 15. Missing input is rejected
 16. R execution failures are handled without crashing the API
 17. Prediction response structure
 18. Cluster profile response structure
 19. K evaluation data structure
 20. Repeated identical input gives a consistent cluster
 21. Different valid inputs can produce different clusters
 22. Phases 1-5 still work

Every test exercises the real R engine or the real saved artifacts. No test
asserts a hard-coded metric value or a stubbed success value.

A note on metrics: K-Means is unsupervised, so there is no held-out accuracy to
assert. The tests therefore check the clustering's own internal consistency
(sizes sum to the dataset, WSS recomputes from the stored centres, the reported
cluster is genuinely the nearest one) rather than inventing a predictive score.
"""
import json
import math
import os
import re
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
from backend.services.kmeans_service import (
    FEATURES,
    IDENTIFIER_COLUMN,
    MODEL_NAME,
    get_clusters_path,
    get_metrics_path,
    get_model_path,
    is_model_trained,
    validate_customer_input,
)
from backend.services.dataset_validator import DATASET_SCHEMAS, validate_dataset

# A valid customer inside the dataset's observed ranges
VALID_PAYLOAD = {
    "age": 42,
    "annual_income": 85,
    "spending_score": 55,
    "purchase_frequency": 28
}

# A customer at the opposite end of the dataset, used to prove the assignment
# actually responds to input rather than returning a fixed cluster.
CONTRAST_PAYLOAD = {
    "age": 65,
    "annual_income": 20,
    "spending_score": 10,
    "purchase_frequency": 3
}


def load_metrics():
    """Reads the on-disk metrics artifact, failing loudly if it is missing."""
    with open(get_metrics_path(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_clusters():
    """Reads the on-disk clusters artifact, failing loudly if it is missing."""
    with open(get_clusters_path(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(relative_path):
    """Reads any JSON artifact by its repo-relative path."""
    with open(os.path.join(PROJECT_ROOT, relative_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestDataset(unittest.TestCase):
    """The existing customers.csv must satisfy the shared dataset schema."""

    def test_customers_dataset_is_valid(self):
        """customers.csv passes the Phase 2 shared dataset validator."""
        result = validate_dataset("customers.csv")

        self.assertTrue(result["valid"], f"customers.csv is invalid: {result['errors']}")
        self.assertGreater(result["row_count"], 0, "customers.csv has no data rows")

    def test_customers_dataset_has_the_documented_columns(self):
        """The dataset carries the four numeric features plus the identifier."""
        result = validate_dataset("customers.csv")
        columns = set(result["columns"])

        for column in FEATURES:
            self.assertIn(column, columns, f"Missing feature column '{column}'")
        self.assertIn(IDENTIFIER_COLUMN, columns)

    def test_dataset_matches_the_service_feature_contract(self):
        """The service's FEATURES tuple matches the dataset's own schema."""
        schema = DATASET_SCHEMAS["customers.csv"]

        self.assertEqual(set(schema["numeric_columns"]), set(FEATURES))
        self.assertIn(IDENTIFIER_COLUMN, schema["required_columns"])
        # K-Means is unsupervised, so the dataset must NOT declare a target.
        self.assertIsNone(schema["target_column"],
                          "customers.csv must have no target column for clustering")

    def test_identifier_is_excluded_from_the_clustering_features(self):
        """customer_id is an identifier, never a clustering feature."""
        self.assertNotIn(IDENTIFIER_COLUMN, FEATURES,
                         "The identifier must never be used as a clustering feature")


class TestTrainingArtifacts(unittest.TestCase):
    """Phase 6 training scripts and the trained artifacts must exist on disk."""

    def test_training_and_prediction_scripts_exist(self):
        """train.R and predict.R must be present in r_models/kmeans/."""
        for script in ("train.R", "predict.R"):
            path = os.path.join(PROJECT_ROOT, "r_models", "kmeans", script)
            self.assertTrue(os.path.isfile(path), f"{script} is missing at {path}")
            self.assertGreater(os.path.getsize(path), 0, f"{script} is empty")

    def test_trained_model_file_exists(self):
        """model.rds must exist and be non-empty after training."""
        model_path = get_model_path()
        self.assertTrue(os.path.isfile(model_path),
                        "model.rds not found — run 'Rscript r_models/kmeans/train.R'")
        self.assertGreater(os.path.getsize(model_path), 0, "model.rds is empty")
        self.assertTrue(is_model_trained())

    def test_training_script_uses_kmeans(self):
        """train.R must fit a real clustering with stats::kmeans()."""
        path = os.path.join(PROJECT_ROOT, "r_models", "kmeans", "train.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("kmeans(", source, "train.R must call kmeans()")
        self.assertIn("K_GRID", source, "train.R must evaluate a grid of K values")
        self.assertIn("scaler_center", source, "train.R must scale the features")
        self.assertIn("set.seed", source, "train.R must fix the seed for reproducibility")

    def test_training_script_computes_rather_than_hardcodes_metrics(self):
        """train.R must compute its metrics, never assign a literal score."""
        path = os.path.join(PROJECT_ROOT, "r_models", "kmeans", "train.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        for metric in ("recomputed_wss", "FINAL_SILHOUETTE", "SELECTED_K"):
            for literal in (f"{metric} <- 0", f"{metric} <- 1", f"{metric} <- 0.",
                            f"{metric} <- 1."):
                self.assertNotIn(
                    literal, source,
                    f"train.R appears to hard-code {metric} via '{literal}'"
                )

    def test_training_script_excludes_the_identifier_from_clustering(self):
        """train.R must never put customer_id into the scaled feature matrix."""
        path = os.path.join(PROJECT_ROOT, "r_models", "kmeans", "train.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("FEATURES <- c(", source, "train.R must declare its feature list")
        feature_line = next(line for line in source.splitlines()
                            if line.strip().startswith("FEATURES <- c("))
        self.assertNotIn(IDENTIFIER_COLUMN, feature_line,
                         "The identifier must not appear in the clustering feature list")

    def test_prediction_script_loads_saved_model(self):
        """predict.R must load model.rds and must not retrain the model."""
        path = os.path.join(PROJECT_ROOT, "r_models", "kmeans", "predict.R")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("readRDS", source, "predict.R must load the saved model via readRDS()")
        self.assertIn("model.rds", source, "predict.R must reference model.rds")
        self.assertIn("scaler", source, "predict.R must apply the saved feature scaling")

        # Comments explain which rule is being applied, and one response field
        # quotes "kmeans()" in prose. Neither refits anything, so both the
        # comments and the string literals are stripped before checking — any
        # kmeans( left in the remaining code would be a real call.
        code_lines = [line for line in source.splitlines()
                      if not line.lstrip().startswith("#")]
        code = re.sub(r'"[^"\n]*"', '""', "\n".join(code_lines))
        self.assertNotIn("kmeans(", code,
                         "predict.R must not call kmeans() on every request")

    def test_metrics_artifact_is_real_json(self):
        """metrics.json must parse and carry the real clustering diagnostics."""
        self.assertTrue(os.path.isfile(get_metrics_path()), "metrics.json not found")
        metrics = load_metrics()

        self.assertTrue(metrics.get("success"))
        self.assertEqual(metrics.get("model"), MODEL_NAME)
        self.assertEqual(metrics.get("learning_type"), "unsupervised")

        values = metrics.get("metrics", {})
        for key in ("wss", "silhouette_score", "between_cluster_ratio", "cluster_count"):
            self.assertIn(key, values, f"metrics.json missing '{key}'")
            self.assertIsInstance(values[key], (int, float), f"'{key}' must be numeric")

        # Silhouette is only meaningful in the documented -1..1 range, and a
        # clustering whose members overlap heavily would sit near zero.
        self.assertGreaterEqual(values["silhouette_score"], -1)
        self.assertLessEqual(values["silhouette_score"], 1)
        self.assertGreater(values["silhouette_score"], 0.2,
                           "A silhouette at or below 0.2 suggests the clusters overlap heavily")

        # WSS is a sum of squared distances, so it can never be negative.
        self.assertGreater(values["wss"], 0)

        # The between-cluster share is a ratio of the total variation.
        self.assertGreater(values["between_cluster_ratio"], 0)
        self.assertLessEqual(values["between_cluster_ratio"], 1)

    def test_clusters_artifact_is_real_json(self):
        """clusters.json must parse and carry one record per clustered row."""
        self.assertTrue(os.path.isfile(get_clusters_path()), "clusters.json not found")
        payload = load_clusters()

        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("model"), MODEL_NAME)
        self.assertEqual(payload.get("identifier_column"), IDENTIFIER_COLUMN)
        self.assertEqual(payload.get("features"), list(FEATURES))

        records = payload.get("clusters")
        self.assertIsInstance(records, list)
        self.assertEqual(
            len(records), payload.get("total_records"),
            "The artifact must hold one record per clustered row"
        )

        # Every record must carry the identifier, a cluster and its own features.
        for record in records:
            self.assertIn("id", record)
            self.assertIn("cluster", record)
            for name in FEATURES:
                self.assertIn(name, record, f"Record {record.get('id')} is missing '{name}'")

    def test_selected_k_is_valid_and_was_actually_evaluated(self):
        """The shipped K must be a positive integer drawn from the K grid."""
        metrics = load_metrics()
        selected_k = metrics.get("selected_k")

        self.assertIsInstance(selected_k, int, "selected_k must be an integer")
        self.assertGreater(selected_k, 1, "K = 1 is not a segmentation")
        self.assertLess(selected_k, metrics["dataset"]["rows_clean"],
                        "K cannot reach the number of usable rows")

        comparison = metrics.get("k_comparison", [])
        self.assertGreaterEqual(len(comparison), 3,
                                "K must be chosen from a real comparison, not guessed")

        evaluated = {entry["k"] for entry in comparison}
        self.assertIn(selected_k, evaluated, "selected_k was not one of the evaluated values")

        selected_flags = [entry for entry in comparison if entry.get("selected")]
        self.assertEqual(len(selected_flags), 1, "Exactly one K must be marked selected")
        self.assertEqual(selected_flags[0]["k"], selected_k)

    def test_k_evaluation_data_structure(self):
        """Each K row must carry WSS, silhouette and stability figures."""
        comparison = load_metrics()["k_comparison"]

        self.assertGreaterEqual(len(comparison), 3)
        for entry in comparison:
            self.assertIsInstance(entry["k"], int)
            self.assertGreater(entry["k"], 0)
            for key in ("wss", "silhouette_mean", "between_ratio",
                        "stability_wss_mean", "stability_wss_sd", "smallest_cluster"):
                self.assertIn(key, entry, f"k={entry['k']} is missing '{key}'")
                self.assertIsInstance(entry[key], (int, float), f"k={entry['k']} '{key}' must be numeric")
                self.assertGreaterEqual(entry[key], 0,
                                        f"k={entry['k']} '{key}' cannot be negative")

        # WSS must fall as K rises — the whole reason the elbow exists.
        ordered = sorted(comparison, key=lambda entry: entry["k"])
        wss_values = [entry["wss"] for entry in ordered]
        for earlier, later in zip(wss_values, wss_values[1:]):
            self.assertLess(later, earlier,
                            "WSS must decrease as K increases for K-Means")

        # A K of 1 member is not a segment, so the grid must never produce one.
        for entry in comparison:
            self.assertGreater(entry["smallest_cluster"], 0,
                               f"k={entry['k']} produced an empty cluster")

    def test_k_selection_records_both_criteria_and_the_reasoning(self):
        """The chosen K must be justified by real numbers, not asserted."""
        metrics = load_metrics()
        selection = metrics["k_selection"]

        self.assertEqual(selection["selected_k"], metrics["selected_k"])
        self.assertIn(selection["elbow_k"], selection["k_grid"])
        self.assertIn(selection["silhouette_k"], selection["k_grid"])
        self.assertTrue(selection.get("basis"), "The selection basis must be recorded")
        self.assertTrue(selection.get("reason"), "The selection reasoning must be recorded")
        self.assertTrue(selection.get("explain", {}).get("elbow"),
                        "The elbow method must be explained to the user")
        self.assertTrue(selection.get("explain", {}).get("silhouette"),
                        "The silhouette method must be explained to the user")

        # The selected K must genuinely be the best on whichever criterion won.
        comparison = {entry["k"]: entry for entry in metrics["k_comparison"]}
        if "silhouette" in selection["basis"]:
            best = max(comparison, key=lambda k: comparison[k]["silhouette_mean"])
            self.assertEqual(selection["selected_k"], best,
                             "A silhouette-based selection must take the silhouette maximum")

    def test_cluster_count_matches_selected_k(self):
        """The fit must produce exactly the K that was selected."""
        metrics = load_metrics()
        selected_k = metrics["selected_k"]

        self.assertEqual(metrics["metrics"]["cluster_count"], selected_k)
        self.assertEqual(len(metrics["clusters"]), selected_k)
        self.assertEqual(len(metrics["cluster_centers"]), selected_k)
        self.assertEqual(len(metrics["cluster_profiles"]), selected_k)

        for expected in range(1, selected_k + 1):
            self.assertIn(expected, [c["cluster"] for c in metrics["clusters"]],
                          f"Cluster {expected} is missing from the metrics artifact")

    def test_every_valid_record_has_a_cluster(self):
        """Every usable dataset row must carry a cluster within 1..K."""
        payload = load_clusters()
        metrics = load_metrics()
        selected_k = metrics["selected_k"]

        self.assertEqual(len(payload["clusters"]), metrics["dataset"]["rows_clean"],
                         "Every cleaned row must be assigned a cluster")

        seen_ids = set()
        for record in payload["clusters"]:
            cluster = record["cluster"]
            self.assertIsInstance(cluster, int, "A cluster id must be an integer")
            self.assertGreaterEqual(cluster, 1, "Cluster ids are 1-based")
            self.assertLessEqual(cluster, selected_k, "A cluster id cannot exceed the model's K")
            self.assertNotIn(record["id"], seen_ids,
                             f"Customer {record['id']} appears twice in the artifact")
            seen_ids.add(record["id"])

    def test_cluster_sizes_sum_to_dataset_size(self):
        """The reported sizes must account for every clustered customer."""
        metrics = load_metrics()
        payload = load_clusters()
        total = metrics["dataset"]["rows_clean"]

        sizes = payload["cluster_sizes"]
        self.assertEqual(sum(int(value) for value in sizes.values()), total,
                         "The cluster sizes must sum to the clustered dataset size")

        counted = {}
        for record in payload["clusters"]:
            counted[record["cluster"]] = counted.get(record["cluster"], 0) + 1

        self.assertEqual(counted, {int(k): int(v) for k, v in sizes.items()},
                         "Each reported size must match the records actually assigned to it")

        # The per-profile sizes in metrics.json must agree with the artifact too.
        for profile in metrics["clusters"]:
            self.assertEqual(profile["size"], sizes[str(profile["cluster"])],
                             f"Cluster {profile['cluster']} size disagrees between artifacts")

    def test_cluster_sizes_are_reported_in_the_profiles(self):
        """No cluster may be a singleton, and every share must be real."""
        metrics = load_metrics()
        total = metrics["dataset"]["rows_clean"]

        for profile in metrics["cluster_profiles"]:
            self.assertIsInstance(profile["size"], int)
            self.assertGreater(profile["size"], 1,
                               f"Cluster {profile['cluster']} is a singleton, not a segment")
            # jsonlite writes numbers at 4 decimal places by default, so the
            # share is compared to within that resolution rather than exactly.
            self.assertAlmostEqual(
                profile["size_share"],
                profile["size"] / total,
                delta=0.0001,
                msg="A cluster's share must equal its size over the dataset size"
            )

    def test_wss_is_reproducible_from_the_stored_centres(self):
        """
        The reported WSS must be recoverable from the saved centres and the
        stored scaling — i.e. it is the model's own objective value, not a
        number typed into the JSON.
        """
        metrics = load_metrics()
        centers = metrics["cluster_centers"]
        scaling = metrics["feature_scaling"]

        self.assertIn("center", scaling)
        self.assertIn("scale", scaling)
        for name in FEATURES:
            self.assertIn(name, scaling["center"], f"No centre recorded for '{name}'")
            self.assertIn(name, scaling["scale"], f"No scale recorded for '{name}'")
            self.assertGreater(scaling["scale"][name], 0,
                               f"Scale for '{name}' must be positive to avoid dividing by zero")

        # Sum the squared distance from every customer to its own cluster centre,
        # in scaled space, and compare with the WSS the artifact reports.
        total = 0.0
        for record in load_clusters()["clusters"]:
            cluster = record["cluster"]
            centre = next(entry for entry in centers if entry["cluster"] == cluster)
            for name in FEATURES:
                scaled = (float(record[name]) - scaling["center"][name]) / scaling["scale"][name]
                difference = scaled - float(centre["center_scaled"][name])
                total += difference * difference

        self.assertAlmostEqual(
            metrics["metrics"]["wss"], round(total, 4), places=1,
            msg="WSS must equal the sum of squared distances to the stored centres"
        )

    def test_cluster_profiles_are_derived_from_their_own_centre(self):
        """
        A profile's centre must sit where its members actually are: recomputing
        the members' mean from clusters.json must reproduce the reported
        original-unit mean, so the label rests on real statistics.
        """
        metrics = load_metrics()
        profiles = {p["cluster"]: p for p in metrics["cluster_profiles"]}

        grouped = {}
        for record in load_clusters()["clusters"]:
            grouped.setdefault(record["cluster"], []).append(record)

        for cluster_id, members in grouped.items():
            profile = profiles[cluster_id]

            for name in FEATURES:
                expected = sum(float(m[name]) for m in members) / len(members)
                reported = float(profile["original_mean_features"][name])
                self.assertAlmostEqual(
                    reported, round(expected, 4), places=2,
                    msg=f"Cluster {cluster_id} mean for '{name}' does not match its members"
                )

    def test_cluster_labels_are_derived_not_arbitrary(self):
        """
        A generated label must be backed by real offsets, and a cluster that is
        not distinct on at least two features must fall back to a neutral name.
        """
        metrics = load_metrics()

        for profile in metrics["cluster_profiles"]:
            offsets = profile["center_vs_population_sd"]
            self.assertEqual(set(offsets.keys()), set(FEATURES))

            # A descriptor is only claimed where the centre is >= 0.5 sd from
            # the population on that feature, which is the rule train.R applies.
            separating = [name for name in FEATURES
                          if abs(float(offsets[name])) >= 0.5]

            if profile["label_is_generic"]:
                self.assertEqual(profile["label"], f"Cluster {profile['cluster']}",
                                 "An ambiguous cluster must use the neutral name")
                self.assertLess(len(separating), 2,
                                "A cluster separated on 2+ features must get a descriptive label")
            else:
                self.assertGreaterEqual(len(separating), 2,
                                        "A descriptive label needs 2+ separating features")
                self.assertNotEqual(profile["label"], f"Cluster {profile['cluster']}")
                self.assertTrue(profile["descriptors"],
                                "A descriptive label must list the descriptors behind it")

    def test_visualization_is_declared_as_pca_for_display_only(self):
        """
        The 2-D plot must be labelled as a projection, and the model must state
        that the clustering itself used the original scaled features.
        """
        viz = load_metrics()["visualization"]

        self.assertIn("PCA", viz["method"])
        self.assertIn("visualisation only", viz["purpose"].lower())
        self.assertIn("never on the components", viz["purpose"].lower())

        points = viz["points"]
        for axis in ("PC1", "PC2", "cluster"):
            self.assertIn(axis, points, f"The projection must carry '{axis}'")

        self.assertEqual(len(points["PC1"]), load_metrics()["dataset"]["rows_clean"],
                         "Every clustered customer must appear in the projection")
        self.assertEqual(len(points["PC1"]), len(points["PC2"]))
        self.assertEqual(len(points["PC1"]), len(points["cluster"]))

        for value in points["PC1"] + points["PC2"]:
            self.assertTrue(math.isfinite(float(value)),
                            "A projection coordinate must be a finite number")

    def test_reproducibility_is_verified_not_assumed(self):
        """The seed, nstart and the verified re-run must all be recorded."""
        reproducibility = load_metrics()["reproducibility"]

        self.assertTrue(reproducibility["same_seed_same_clusters"],
                        "train.R must verify the fit reproduces under the same seed")
        self.assertIsInstance(reproducibility["seed"], int)
        self.assertGreater(reproducibility["n_start"], 1,
                           "A single-start kmeans() fit is not reproducible enough")

    def test_clusters_artifact_does_not_leak_other_columns(self):
        """Only the identifier, the features and model outputs may be published."""
        allowed = {"id", "cluster", "distance_to_center", "separation_ratio", "PC1", "PC2"}
        allowed.update(FEATURES)

        for record in load_clusters()["clusters"]:
            self.assertEqual(set(record.keys()), allowed,
                             f"Unexpected fields published: {set(record.keys()) - allowed}")

    def test_frontend_page_and_assets_exist(self):
        """The Phase 6 page and its assets must be present."""
        for relative in (
            os.path.join("frontend", "kmeans.html"),
            os.path.join("frontend", "css", "kmeans.css"),
            os.path.join("frontend", "js", "kmeans.js"),
            os.path.join("tests", "kmeans_frontend_test.js"),
        ):
            path = os.path.join(PROJECT_ROOT, relative)
            self.assertTrue(os.path.isfile(path), f"Missing file: {relative}")
            self.assertGreater(os.path.getsize(path), 0, f"Empty file: {relative}")

    def test_frontend_page_wires_up_every_required_element(self):
        """
        The page must contain the elements kmeans.js queries, and the script that
        consumes the API — otherwise the page would silently render nothing.
        """
        with open(os.path.join(PROJECT_ROOT, "frontend", "kmeans.html"),
                  "r", encoding="utf-8") as handle:
            html = handle.read()

        for element_id in (
            "predictionForm", "btnPredict", "resultPanel", "resultClusterBadge",
            "inputAge", "inputAnnual_income", "inputSpending_score",
            "inputPurchase_frequency",
            "metricRecords", "metricSelectedK", "metricClusterCount",
            "metricSilhouette", "metricWss",
            "clusterCardGrid", "profileGrid", "scatterWrap", "vizLegend",
            "elbowWrap", "kTableBody", "resultDistance", "distanceList",
        ):
            self.assertIn(f'id="{element_id}"', html,
                          f"kmeans.html is missing the element '{element_id}'")

        self.assertIn('src="js/kmeans.js"', html, "kmeans.html must load its page script")
        self.assertIn("ASSIGN CUSTOMER TO CLUSTER", html, "The submit button label is required")
        self.assertIn("CUSTOMER SEGMENTATION", html, "The segmentation section is required")
        self.assertIn("K-MEANS CLUSTERING", html, "The model name is required")
        self.assertIn("OPTIMAL K ANALYSIS", html, "The K analysis section is required")

    def test_frontend_script_calls_every_documented_endpoint(self):
        """The page script must read the endpoints the API contract defines."""
        with open(os.path.join(PROJECT_ROOT, "frontend", "js", "kmeans.js"),
                  "r", encoding="utf-8") as handle:
            source = handle.read()

        for endpoint in ("/api/kmeans/predict", "/api/kmeans/metrics",
                         "/api/kmeans/clusters", "/api/kmeans/profiles",
                         "/api/kmeans/schema"):
            self.assertIn(endpoint, source, f"The page script must call {endpoint}")

        # The page must not cluster on the identifier.
        fields_line = next(line for line in source.splitlines()
                           if line.strip().startswith("const FIELDS"))
        self.assertNotIn(IDENTIFIER_COLUMN, fields_line,
                         "The frontend must not send the identifier as a feature")

    def test_dashboard_marks_kmeans_active(self):
        """The dashboard's K-Means card must be ACTIVE and open the page."""
        index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
        with open(index_path, "r", encoding="utf-8") as handle:
            html = handle.read()

        self.assertIn('href="kmeans.html"', html, "Dashboard does not link to kmeans.html")
        self.assertNotIn("Coming in Phase 6", html,
                         "Stale 'Coming in Phase 6' text remains on the K-Means card")

        # The earlier models must still be intact.
        for page in ("regression.html", "decision-tree.html", "knn.html"):
            self.assertIn(f'href="{page}"', html,
                          f"Dashboard no longer links to the {page} page")


class TestAssignmentApi(unittest.TestCase):
    """POST /api/kmeans/predict must assign to a real cluster centre."""

    def setUp(self):
        self.client = app.test_client()

    def test_valid_customer_assignment_success(self):
        """A valid payload returns 200 with a cluster within the model's K."""
        response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)

        selected_k = load_metrics()["selected_k"]
        self.assertIsInstance(data["cluster"], int)
        self.assertGreaterEqual(data["cluster"], 1)
        self.assertLessEqual(data["cluster"], selected_k,
                             "The assigned cluster must be one the model was trained on")

    def test_assignment_response_structure(self):
        """The success response must carry every documented field."""
        data = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()

        for key in ("success", "model", "cluster", "distance", "cluster_label"):
            self.assertIn(key, data, f"Response is missing required key '{key}'")

        for key in ("algorithm", "inputs", "distances", "separation_ratio",
                    "features", "k", "warnings", "trained_at", "scaling_applied",
                    "cluster_centers_original", "projection"):
            self.assertIn(key, data, f"Response is missing provenance key '{key}'")

        # The echoed inputs must match what was sent.
        for name in FEATURES:
            self.assertAlmostEqual(float(data["inputs"][name]),
                                   float(VALID_PAYLOAD[name]), places=6)

        # The response must advertise the same K the model was trained with.
        self.assertEqual(data["k"], load_metrics()["selected_k"])

    def test_assigned_cluster_is_the_nearest_centre(self):
        """
        The reported cluster must genuinely be the closest of the distances the
        response itself reports — otherwise the answer is not a nearest-centre
        assignment at all.
        """
        data = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()
        distances = data["distances"]

        self.assertTrue(distances, "The response must report the distance to every centre")
        self.assertEqual(len(distances), data["k"],
                         "There must be one distance per cluster centre")

        parsed = {}
        for name, value in distances.items():
            self.assertIsInstance(value, (int, float))
            self.assertGreaterEqual(value, 0, "A Euclidean distance cannot be negative")
            parsed[int(str(name).replace("Cluster ", "").strip())] = float(value)

        nearest = min(parsed, key=parsed.get)
        self.assertEqual(nearest, data["cluster"],
                         "The assigned cluster must be the one with the smallest distance")
        self.assertAlmostEqual(data["distance"], parsed[data["cluster"]], places=4)

    def test_distance_matches_the_reported_scaling(self):
        """
        The distance must be computable from the reported scaled inputs and the
        cluster's own centre, proving the scaling was actually applied.
        """
        data = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()
        centres = {entry["cluster"]: entry for entry in load_metrics()["cluster_centers"]}
        centre = centres[data["cluster"]]

        total = 0.0
        for name in FEATURES:
            difference = float(data["scaled_inputs"][name]) - float(centre["center_scaled"][name])
            total += difference * difference

        self.assertAlmostEqual(data["distance"], round(math.sqrt(total), 4), places=4,
                               msg="The distance must equal the scaled Euclidean distance "
                                   "between the customer and the assigned centre")

    def test_assignment_is_reproducible(self):
        """
        The same input must yield the same assignment because predict.R loads a
        fixed saved model rather than retraining on each request.
        """
        first = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()
        second = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()

        self.assertEqual(first["cluster"], second["cluster"],
                         "Identical input produced different clusters — the model is not reused")
        self.assertEqual(first["distance"], second["distance"],
                         "Identical input produced different distances")

    def test_different_customers_can_reach_different_clusters(self):
        """
        Two clearly different customers must be assignable to different
        clusters. A hard-coded response would return the same cluster for both.
        """
        first = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()
        second = self.client.post('/api/kmeans/predict', json=CONTRAST_PAYLOAD).get_json()

        self.assertTrue(first["success"], first)
        self.assertTrue(second["success"], second)
        self.assertNotEqual(
            first["cluster"], second["cluster"],
            "Different customers landed in the same cluster — the assignment looks hard-coded"
        )
        self.assertNotEqual(first["cluster_label"], second["cluster_label"],
                            "Two different clusters must not share a generated label")

    def test_customer_at_a_centre_lands_in_that_cluster(self):
        """
        Feeding the model's own centre back in must return that cluster, with a
        distance of essentially zero. This is the strongest available check that
        the stored centres and the assignment path agree.
        """
        for entry in load_metrics()["cluster_centers"]:
            payload = {name: float(entry["center_original"][name]) for name in FEATURES}
            data = self.client.post('/api/kmeans/predict', json=payload).get_json()

            self.assertTrue(data["success"], data)
            self.assertEqual(data["cluster"], entry["cluster"],
                             "A customer sitting exactly on a centre must land in it")
            self.assertAlmostEqual(data["distance"], 0.0, places=4,
                                   msg="A customer on a centre has distance 0")

    def test_assignment_surfaces_real_training_metrics(self):
        """The response must carry the real metrics from training."""
        trained = load_metrics()["metrics"]
        data = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()

        self.assertIn("model_wss", data)
        self.assertIn("model_silhouette_score", data)
        self.assertIn("model_records_clustered", data)
        self.assertAlmostEqual(data["model_wss"], trained["wss"], places=4)
        self.assertAlmostEqual(data["model_silhouette_score"],
                               trained["silhouette_score"], places=4)
        self.assertEqual(data["model_records_clustered"], trained["records_clustered"])


class TestAssignmentValidation(unittest.TestCase):
    """Invalid and missing input must be rejected with clean 400 errors."""

    def setUp(self):
        self.client = app.test_client()

    def assert_rejected(self, payload, expected_fragment):
        response = self.client.post('/api/kmeans/predict', json=payload)
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

    def test_non_numeric_value_rejected(self):
        """A non-numeric feature value must be rejected."""
        self.assert_rejected(dict(VALID_PAYLOAD, age="middle-aged"), "age")

    def test_empty_string_value_rejected(self):
        """An empty value is not a number."""
        self.assert_rejected(dict(VALID_PAYLOAD, spending_score=""), "spending score")

    def test_nan_rejected(self):
        """NaN and infinity are not usable feature values."""
        nan_payload = dict(VALID_PAYLOAD)
        nan_payload["age"] = float("nan")
        self.assert_rejected(nan_payload, "age")

        inf_payload = dict(VALID_PAYLOAD)
        inf_payload["annual_income"] = float("inf")
        self.assert_rejected(inf_payload, "annual income")

    def test_negative_income_rejected(self):
        """A negative income is not a real customer value."""
        self.assert_rejected(dict(VALID_PAYLOAD, annual_income=-1), "annual income")

    def test_out_of_domain_age_rejected(self):
        """An age beyond the documented 0-120 domain must be rejected."""
        self.assert_rejected(dict(VALID_PAYLOAD, age=150), "age")

    def test_spending_score_above_100_rejected(self):
        """A spending score is a 0-100 percentage."""
        self.assert_rejected(dict(VALID_PAYLOAD, spending_score=150), "spending score")

    def test_negative_purchase_frequency_rejected(self):
        """Purchase frequency cannot be negative."""
        self.assert_rejected(dict(VALID_PAYLOAD, purchase_frequency=-2),
                             "purchase frequency")

    def test_missing_fields_rejected(self):
        """A partial payload must report every missing field."""
        data = self.assert_rejected({"age": 40}, "required")
        self.assertGreaterEqual(len(data["errors"]), 3,
                                "All three missing fields should be reported")

    def test_empty_body_rejected(self):
        """An empty JSON object must be rejected."""
        self.assert_rejected({}, "required")

    def test_unknown_field_rejected(self):
        """A typo'd field name must not be silently ignored."""
        data = self.assert_rejected(dict(VALID_PAYLOAD, income=9), "unrecognised")
        self.assertIn("income", " ".join(data["errors"]))

    def test_identifier_is_not_accepted_as_a_feature(self):
        """customer_id must be rejected as a clustering feature."""
        self.assert_rejected(dict(VALID_PAYLOAD, customer_id="CUST_1001"), "unrecognised")

    def test_non_json_request_rejected(self):
        """A request without a JSON content type must return 415."""
        response = self.client.post('/api/kmeans/predict', data="age=40")
        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.get_json()["success"])

    def test_malformed_json_rejected(self):
        """A body that claims to be JSON but is not must return 400."""
        response = self.client.post(
            '/api/kmeans/predict',
            data="{not valid json",
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_domain_boundaries_accepted(self):
        """Both ends of every documented domain must be valid."""
        cases = [
            {"age": 0}, {"age": 120},
            {"annual_income": 0},
            {"spending_score": 0}, {"spending_score": 100},
            {"purchase_frequency": 0},
        ]
        for override in cases:
            response = self.client.post('/api/kmeans/predict',
                                        json=dict(VALID_PAYLOAD, **override))
            self.assertEqual(
                response.status_code, 200,
                f"{override} was rejected: {response.get_data(as_text=True)}"
            )

    def test_out_of_training_range_warns_but_is_answered(self):
        """
        A logically valid value outside the training range must still be
        assigned, and the extrapolation must be reported rather than hidden.
        """
        response = self.client.post('/api/kmeans/predict', json=dict(
            VALID_PAYLOAD, age=110
        ))
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["cluster"], range(1, load_metrics()["selected_k"] + 1))

        warned_fields = {w["field"] for w in data.get("warnings", [])}
        self.assertIn("age", warned_fields,
                      "An out-of-range age must produce an extrapolation warning")

    def test_in_range_input_has_no_extrapolation_warning(self):
        """A normal customer must not be told the assignment was extrapolated."""
        data = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD).get_json()
        out_of_range = [w for w in data.get("warnings", []) if w["field"] in FEATURES]
        self.assertEqual(out_of_range, [],
                         f"In-range input produced warnings: {out_of_range}")

    def test_validate_customer_input_unit_level(self):
        """The validation helper returns cleaned floats and error messages."""
        cleaned, errors = validate_customer_input(VALID_PAYLOAD)
        self.assertEqual(errors, [])
        self.assertEqual(set(cleaned.keys()), set(FEATURES))
        for value in cleaned.values():
            self.assertIsInstance(value, float)

        _, errors = validate_customer_input({"age": 40})
        self.assertTrue(errors)
        self.assertEqual(len(errors), 3, "One error per missing field")

        _, errors = validate_customer_input("not a dict")
        self.assertEqual(len(errors), 1)

        # Booleans must not be coerced to 1 / 0.
        _, errors = validate_customer_input(dict(VALID_PAYLOAD, age=True))
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
            "error": "Error in kmeans(): object 'centers' not found",
            "return_code": 1
        }
        with patch("backend.services.kmeans_service.r_runner.execute_script",
                   return_value=failure):
            response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("centers' not found", data["error"])

    def test_r_unparseable_output_returns_502(self):
        """Non-JSON stdout from R must produce a clean 502, not a crash."""
        garbage = {
            "success": True,
            "data": None,          # runner could not parse a JSON object
            "stdout": "not json at all",
            "return_code": 0
        }
        with patch("backend.services.kmeans_service.r_runner.execute_script",
                   return_value=garbage):
            response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.get_json()["success"])

    def test_r_returns_out_of_range_cluster_returns_502(self):
        """
        A cluster id beyond the model's K must be rejected rather than passed
        through to the user.
        """
        selected_k = load_metrics()["selected_k"]
        bad = {
            "success": True,
            "data": {"success": True, "cluster": selected_k + 5, "k": selected_k,
                     "model": MODEL_NAME},
            "return_code": 0
        }
        with patch("backend.services.kmeans_service.r_runner.execute_script",
                   return_value=bad):
            response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertIn("trained with K", response.get_json()["error"])

    def test_r_returns_non_integer_cluster_returns_502(self):
        """A cluster id that is not an integer must be rejected."""
        bad = {
            "success": True,
            "data": {"success": True, "cluster": "two", "k": 4, "model": MODEL_NAME},
            "return_code": 0
        }
        with patch("backend.services.kmeans_service.r_runner.execute_script",
                   return_value=bad):
            response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 502)
        self.assertIn("unexpected cluster id", response.get_json()["error"].lower())

    def test_missing_model_file_returns_503(self):
        """An untrained model must return 503 with retraining instructions."""
        # app.py puts backend/ on sys.path, so the routes hold this service as
        # 'services.kmeans_service' — a different module object than the
        # 'backend.services.kmeans_service' alias used above.
        with patch("services.kmeans_service.is_model_trained", return_value=False):
            response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)

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
                "error": "Field 'spending_score' must be between 0 and 100.",
                "errors": ["Field 'spending_score' must be between 0 and 100."]
            }
        }
        with patch("backend.services.kmeans_service.r_runner.execute_script",
                   return_value=r_rejection):
            response = self.client.post('/api/kmeans/predict', json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])


class TestMetricsClustersAndProfilesApi(unittest.TestCase):
    """Metrics, clusters and profiles endpoints must serve the real artifacts."""

    def setUp(self):
        self.client = app.test_client()

    def test_metrics_endpoint(self):
        """GET /api/kmeans/metrics returns the real clustering diagnostics."""
        response = self.client.get('/api/kmeans/metrics')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertEqual(data["learning_type"], "unsupervised")
        self.assertIsInstance(data["selected_k"], int)

        metrics = data["metrics"]
        for key in ("wss", "silhouette_score", "between_cluster_ratio", "cluster_count"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            self.assertGreaterEqual(metrics[key], 0)

        # The endpoint must agree with the artifact on disk.
        on_disk = load_metrics()["metrics"]
        for key in ("wss", "silhouette_score", "between_cluster_ratio"):
            self.assertEqual(metrics[key], on_disk[key])

    def test_metrics_endpoint_reports_dataset_provenance(self):
        """Metrics must describe the real clustered dataset."""
        data = self.client.get('/api/kmeans/metrics').get_json()
        dataset = data["dataset"]

        self.assertEqual(dataset["file"], "datasets/customers.csv")
        self.assertEqual(dataset["identifier_column"], IDENTIFIER_COLUMN)
        self.assertEqual(dataset["features"], list(FEATURES))
        self.assertGreater(dataset["rows_clean"], 0)
        self.assertEqual(
            dataset["rows_clean"] + dataset["rows_dropped"], dataset["rows_loaded"],
            "Kept plus dropped rows must equal the loaded dataset size"
        )
        self.assertIn(IDENTIFIER_COLUMN, dataset["excluded_from_clustering"])

    def test_metrics_endpoint_includes_k_comparison_and_selection(self):
        """The K grid and the selection reasoning must both be exposed."""
        data = self.client.get('/api/kmeans/metrics').get_json()

        self.assertGreaterEqual(len(data["k_comparison"]), 3)
        self.assertIn("selected_k", data["k_selection"])
        self.assertIn("elbow_k", data["k_selection"])
        self.assertIn("silhouette_k", data["k_selection"])
        self.assertTrue(data["k_selection"]["reason"])
        self.assertIn("visualization", data)
        self.assertIn("points", data["visualization"])
        self.assertIn("feature_scaling", data)

    def test_metrics_missing_artifact_returns_503(self):
        """A missing metrics artifact returns 503 with retraining instructions."""
        with patch("services.kmeans_service.get_metrics_path",
                   value="/nonexistent/metrics.json"):
            response = self.client.get('/api/kmeans/metrics')

        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("train.R", data["error"])

    def test_clusters_endpoint(self):
        """GET /api/kmeans/clusters returns the real per-customer assignments."""
        response = self.client.get('/api/kmeans/clusters')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["identifier_column"], IDENTIFIER_COLUMN)
        self.assertEqual(data["features"], list(FEATURES))

        records = data["clusters"]
        self.assertEqual(len(records), data["total_records"])

        selected_k = data["selected_k"]
        sizes = {}
        for record in records:
            self.assertIn("id", record)
            self.assertIn("cluster", record)
            self.assertIn(record["cluster"], range(1, selected_k + 1))
            sizes[record["cluster"]] = sizes.get(record["cluster"], 0) + 1

        self.assertEqual(sizes, {int(k): int(v) for k, v in data["cluster_sizes"].items()},
                         "The reported cluster sizes must match the records returned")
        self.assertEqual(sum(sizes.values()), data["total_records"])

    def test_clusters_endpoint_agrees_with_the_metrics_artifact(self):
        """The two artifacts must describe the same clustering."""
        clusters = self.client.get('/api/kmeans/clusters').get_json()
        metrics = load_metrics()

        self.assertEqual(clusters["selected_k"], metrics["selected_k"])
        self.assertEqual(clusters["total_records"], metrics["dataset"]["rows_clean"])
        self.assertEqual(clusters["silhouette_score"], metrics["metrics"]["silhouette_score"])
        self.assertEqual(clusters["wss"], metrics["metrics"]["wss"])
        self.assertEqual(len(clusters["cluster_centers_projected"]), metrics["selected_k"])

    def test_clusters_missing_artifact_returns_503(self):
        """A missing clusters artifact returns 503 with retraining instructions."""
        with patch("services.kmeans_service.get_clusters_path",
                   value="/nonexistent/clusters.json"):
            response = self.client.get('/api/kmeans/clusters')

        self.assertEqual(response.status_code, 503)
        self.assertIn("train.R", response.get_json()["error"])

    def test_profiles_endpoint(self):
        """GET /api/kmeans/profiles returns the real per-cluster statistics."""
        response = self.client.get('/api/kmeans/profiles')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["features"], list(FEATURES))

        profiles = data["profiles"]
        selected_k = data["selected_k"]
        self.assertEqual(len(profiles), selected_k,
                         "There must be one profile per cluster")

        total = data["total_records"]
        self.assertEqual(sum(p["size"] for p in profiles), total,
                         "The profile sizes must sum to the clustered dataset size")

        for profile in profiles:
            for key in ("cluster", "label", "size", "size_share", "segment",
                        "mean_features", "original_mean_features",
                        "center_scaled", "center_original", "center_vs_population_sd"):
                self.assertIn(key, profile, f"Cluster {profile['cluster']} is missing '{key}'")

            for name in FEATURES:
                self.assertIn(name, profile["center_scaled"])
                self.assertIn(name, profile["center_original"])
                self.assertIn(name, profile["center_vs_population_sd"])

    def test_profiles_response_structure(self):
        """
        The profile centres must be internally consistent: the original-scale
        centre must be the scaled centre with the saved scaling undone.
        """
        data = self.client.get('/api/kmeans/profiles').get_json()
        scaling = load_metrics()["feature_scaling"]

        for profile in data["profiles"]:
            for name in FEATURES:
                scaled = float(profile["center_scaled"][name])
                original = float(profile["center_original"][name])

                expected = scaled * float(scaling["scale"][name]) + float(scaling["center"][name])
                self.assertAlmostEqual(
                    original, round(expected, 3), places=2,
                    msg=(f"Cluster {profile['cluster']}: the original-scale centre for "
                         f"'{name}' must be the scaled centre with the scaling undone")
                )

    def test_profiles_missing_artifact_returns_503(self):
        """A missing artifact returns 503 with retraining instructions."""
        with patch("services.kmeans_service.get_profiles_path",
                   value="/nonexistent/profiles.json"):
            response = self.client.get('/api/kmeans/profiles')

        self.assertEqual(response.status_code, 503)
        self.assertIn("train.R", response.get_json()["error"])

    def test_profiles_come_from_the_profiles_artifact(self):
        """
        The profile data is read from profiles.json, the dedicated artifact
        train.R writes, rather than being parsed out of metrics.json.
        """
        data = self.client.get('/api/kmeans/profiles').get_json()
        self.assertEqual(data["profiles_file"], "r_models/kmeans/profiles.json")

        on_disk = load_json("r_models/kmeans/profiles.json")
        self.assertEqual(
            [p["cluster"] for p in data["profiles"]],
            [p["cluster"] for p in on_disk["profiles"]],
            "The endpoint must serve the profiles exactly as train.R wrote them"
        )
        for served, stored in zip(data["profiles"], on_disk["profiles"]):
            self.assertEqual(served["size"], stored["size"])
            self.assertEqual(served["label"], stored["label"])
            self.assertEqual(served["center_original"], stored["center_original"])

    def test_profiles_artifact_documents_how_names_are_derived(self):
        """
        Cluster labels are generated from measured centres, so the artifact has
        to say so - a reader must not have to guess whether a name is invented.
        """
        data = self.client.get('/api/kmeans/profiles').get_json()
        self.assertTrue(data.get("naming"),
                        "profiles.json must document the label derivation rule")
        self.assertIn("profiles.json", data.get("profiles_file", ""))

    def test_config_endpoint(self):
        """GET /api/kmeans/config returns the real trained configuration."""
        response = self.client.get('/api/kmeans/config')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], "K-Means")
        self.assertEqual(data["selected_k"], load_metrics()["selected_k"])
        self.assertEqual(data["feature_names"], list(FEATURES))

    def test_config_endpoint_reports_real_feature_ranges(self):
        """The config ranges must be the ones the model was actually trained on."""
        data = self.client.get('/api/kmeans/config').get_json()
        trained = load_metrics()["dataset"]["feature_ranges"]

        self.assertEqual(set(data["feature_ranges"].keys()), set(FEATURES))
        for name in FEATURES:
            self.assertEqual(data["feature_ranges"][name]["min"], trained[name]["min"])
            self.assertEqual(data["feature_ranges"][name]["max"], trained[name]["max"])

        for feature in data["features"]:
            self.assertLess(feature["min"], feature["max"],
                            f"'{feature['name']}' needs a real min/max")

    def test_config_endpoint_reports_the_identifier_is_excluded(self):
        """
        customer_id labels customers but must never be a clustering feature, so
        the config has to say which column was held out.
        """
        data = self.client.get('/api/kmeans/config').get_json()
        self.assertEqual(data["identifier_column"], IDENTIFIER_COLUMN)
        self.assertIn(IDENTIFIER_COLUMN, data["excluded_from_clustering"])
        self.assertNotIn(IDENTIFIER_COLUMN, data["feature_names"])

    def test_config_missing_artifact_returns_503(self):
        """A missing metrics artifact returns 503 with retraining instructions."""
        with patch("services.kmeans_service.get_metrics_path",
                   value="/nonexistent/metrics.json"):
            response = self.client.get('/api/kmeans/config')

        self.assertEqual(response.status_code, 503)
        self.assertIn("train.R", response.get_json()["error"])

    def test_schema_endpoint_reports_training_ranges(self):
        """GET /api/kmeans/schema exposes the real dataset ranges."""
        response = self.client.get('/api/kmeans/schema')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["model_trained"])
        self.assertEqual(data["identifier_column"], IDENTIFIER_COLUMN)

        features = {f["name"]: f for f in data["features"]}
        self.assertEqual(set(features.keys()), set(FEATURES))

        for name in FEATURES:
            self.assertLess(features[name]["min"], features[name]["max"],
                            f"'{name}' must have a real min/max from the dataset")

        # The documented domain must sit inside the observed data.
        self.assertGreaterEqual(features["age"]["min"], 0)
        self.assertLessEqual(features["age"]["max"], 120)
        self.assertGreaterEqual(features["spending_score"]["min"], 0)
        self.assertLessEqual(features["spending_score"]["max"], 100)


class TestPhasePreservation(unittest.TestCase):
    """Phase 1-5 endpoints must keep working after Phase 6."""

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

    def test_phase3_regression_still_works(self):
        """The Phase 3 Linear Regression endpoints must be unaffected."""
        response = self.client.post('/api/regression/predict', json={
            "area": 1500, "bedrooms": 3, "bathrooms": 2,
            "location_score": 8, "property_age": 5
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], "Linear Regression")
        self.assertIsInstance(data["prediction"], (int, float))

        self.assertEqual(self.client.get('/api/regression/metrics').status_code, 200)

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

    def test_phase5_knn_still_works(self):
        """The Phase 5 KNN endpoints must be unaffected."""
        response = self.client.post('/api/knn/predict', json={
            "study_hours": 6.5, "attendance": 88, "previous_score": 74,
            "assignments_completed": 9, "practical_score": 81
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], ("LOW", "MEDIUM", "HIGH"))

        for path in ('/api/knn/metrics', '/api/knn/config', '/api/knn/schema'):
            self.assertEqual(self.client.get(path).status_code, 200, f"{path} regressed")

    def test_root_endpoint_lists_all_four_models(self):
        data = self.client.get('/').get_json()
        endpoints = data["endpoints"]

        for key in ("regression_predict", "decision_tree_predict",
                    "knn_predict", "kmeans_predict", "kmeans_metrics",
                    "kmeans_clusters", "kmeans_profiles", "kmeans_schema"):
            self.assertIn(key, endpoints)

        self.assertIn("Phase 3", data["phase"])
        self.assertIn("Phase 4", data["phase"])
        self.assertIn("Phase 5", data["phase"])
        self.assertIn("Phase 6", data["phase"])


if __name__ == '__main__':
    unittest.main()
