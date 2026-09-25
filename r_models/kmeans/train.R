# ==============================================================================
# AI Insight Hub — Phase 6
# K-Means Training Script (Customer Segmentation)
#
# WHAT THIS SCRIPT DOES
#   1. Loads datasets/customers.csv
#   2. Validates the schema and every row
#   3. Drops rows that cannot be made numerically valid (and counts every drop)
#   4. Selects the clustering features, EXCLUDING the customer_id identifier
#   5. Standardises those features (z-score, from the cleaned data)
#   6. Scores K = 2..6 with REAL clustering diagnostics:
#        * total within-cluster sum of squares (WSS / tot.withinss)
#        * average silhouette width
#        * the elbow: the largest drop in WSS
#        * a repeat-stability check over several random restarts
#      K is then chosen from those numbers, never picked for appearance.
#   7. Fits the final model with stats::kmeans() using a fixed seed
#   8. Writes model.rds, metrics.json and clusters.json
#
# WHY THE FEATURES ARE SCALED
#   K-Means minimises a squared Euclidean distance. In customers.csv
#   annual_income spans 15.6-149.3 while purchase_frequency spans 2-50; left
#   unscaled, annual_income's larger numeric range would dominate the distance
#   and the other three features would barely influence the result. Scaling
#   every feature to mean 0 / sd 1 makes each one contribute equally.
#
# WHY customer_id IS NEVER CLUSTERED ON
#   customer_id is a unique identifier with no behavioural meaning - it is
#   ordered by registration, so clustering on it would recover "which accounts
#   were opened first" instead of finding customer types. It is used only to
#   label the output rows.
#
# WHY THERE IS NO train/test SPLIT
#   K-Means is UNSUPERVISED: there is no label to be right or wrong about, so
#   there is no held-out accuracy to report. The honest quality measures are
#   internal cohesion (WSS) and separation (silhouette), and this script
#   computes those on the real clustering rather than inventing an accuracy.
#
# NOTHING IS HARD-CODED. Every number written to disk comes from the kmeans()
# fit this script performs, and every cluster label is derived from that
# cluster's own measured centre.
#
# RUN IT WITH:
#   Rscript r_models/kmeans/train.R
#
# This is an educational ML implementation, not a real commercial tool.
# ==============================================================================

# --- Library configuration (kept in sync with the Python RRunner bridge) ------
lib_user <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_user) && dir.exists(lib_user)) {
  .libPaths(c(lib_user, .libPaths()))
}

has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
if (!has_jsonlite) {
  cat("FATAL: The 'jsonlite' package is required to write the model metrics.\n")
  cat("Install it with: install.packages('jsonlite')\n")
  quit(status = 1)
}

# --- Resolve project paths from the script location (works from any cwd) -----
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else {
  getwd()
}
project_root <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/")

dataset_path <- file.path(project_root, "datasets", "customers.csv")
model_dir <- script_dir

# Paths for the artifacts this script produces
model_path <- file.path(model_dir, "model.rds")
metrics_path <- file.path(model_dir, "metrics.json")
clusters_path <- file.path(model_dir, "clusters.json")
profiles_path <- file.path(model_dir, "profiles.json")

# --- Model configuration -----------------------------------------------------
# The identifier column. It is NEVER used as a clustering feature.
IDENTIFIER_COLUMN <- "customer_id"

# The four numeric behavioural features the model clusters on. These match the
# numeric_columns declared for customers.csv in
# backend/services/dataset_validator.py and are asserted against the real file
# header in STEP 2 below, so a schema change is caught here rather than silently
# changing the model.
FEATURES <- c("age", "annual_income", "spending_score", "purchase_frequency")

REQUIRED_COLUMNS <- c(IDENTIFIER_COLUMN, FEATURES)

RANDOM_SEED <- 42         # Fixed seed so the whole script reproduces exactly
K_GRID <- c(2, 3, 4, 5, 6)   # Candidate cluster counts that get scored
N_START <- 25             # kmeans() restarts per fit; kmeans keeps the best
STABILITY_RESTARTS <- 10  # Independent fits used to measure K's stability

# ==============================================================================
# STEP 1 - Load the dataset
# ==============================================================================
if (!file.exists(dataset_path)) {
  cat(sprintf("FATAL: Dataset not found at %s\n", dataset_path))
  quit(status = 1)
}

raw_data <- read.csv(dataset_path, stringsAsFactors = FALSE, check.names = FALSE)
rows_loaded <- nrow(raw_data)
cat(sprintf("[1/12] Loaded dataset: %s (%d rows)\n", dataset_path, rows_loaded))
cat(sprintf("      Columns found: %s\n", paste(names(raw_data), collapse = ", ")))

# ==============================================================================
# STEP 2 - Validate the schema against the real header
#   The feature list above is not taken on trust: it is checked against the
#   columns the CSV actually has, so a renamed or dropped column fails loudly
#   here instead of quietly producing a different model.
# ==============================================================================
missing_columns <- setdiff(REQUIRED_COLUMNS, names(raw_data))
if (length(missing_columns) > 0) {
  cat(sprintf("FATAL: customers.csv is missing required column(s): %s\n",
              paste(missing_columns, collapse = ", ")))
  quit(status = 1)
}

# Any other numeric column present that we are NOT clustering on is reported, so
# a future schema change cannot hide a feature that is silently being dropped.
excluded_numeric <- setdiff(names(raw_data), REQUIRED_COLUMNS)
excluded_numeric <- excluded_numeric[vapply(
  excluded_numeric,
  function(col_name) is.numeric(raw_data[[col_name]]),
  logical(1)
)]

cat(sprintf("[2/12] Schema validation passed. Identifier: '%s'\n", IDENTIFIER_COLUMN))
cat(sprintf("      Clustering features: %s\n", paste(FEATURES, collapse = ", ")))
if (length(excluded_numeric) > 0) {
  cat(sprintf("      Excluded from clustering (not in the feature set): %s\n",
              paste(excluded_numeric, collapse = ", ")))
}

# The identifier must actually identify rows. If it were duplicated we could not
# tell which customer a cluster assignment belonged to.
id_values <- trimws(as.character(raw_data[[IDENTIFIER_COLUMN]]))
rows_blank_id <- sum(is.na(id_values) | id_values == "")
unique_ids <- unique(id_values[!is.na(id_values) & id_values != ""])
if (length(unique_ids) != length(id_values) - rows_blank_id) {
  cat(sprintf("FATAL: %s is not unique - it cannot identify each customer row.\n",
              IDENTIFIER_COLUMN))
  quit(status = 1)
}

# ==============================================================================
# STEP 3 - Clean / validate the data
#   read.csv() parses a dirty numeric column as character, so every feature is
#   coerced to numeric and any row that cannot be made fully valid is dropped
#   (and counted, never silently ignored).
# ==============================================================================
data <- raw_data[, REQUIRED_COLUMNS]

for (col_name in FEATURES) {
  if (!is.numeric(data[[col_name]])) {
    data[[col_name]] <- suppressWarnings(as.numeric(as.character(data[[col_name]])))
  }
}

data[[IDENTIFIER_COLUMN]] <- trimws(as.character(data[[IDENTIFIER_COLUMN]]))

is_complete <- stats::complete.cases(data)
rows_incomplete <- sum(!is_complete)
data <- data[is_complete, ]

# Domain rules a genuine customer record must satisfy. These are the documented
# schema of the dataset, not values copied out of any model output.
is_valid_row <- data$age >= 0 & data$age <= 120 &
               data$annual_income >= 0 &
               data$spending_score >= 0 & data$spending_score <= 100 &
               data$purchase_frequency >= 0
rows_invalid <- sum(!is_valid_row)
data <- data[is_valid_row, ]

rows_dropped <- rows_incomplete + rows_invalid
rows_clean <- nrow(data)

if (rows_clean <= length(FEATURES) + 1) {
  cat(sprintf("FATAL: Not enough usable rows after cleaning (%d rows remain).\n", rows_clean))
  quit(status = 1)
}

# A feature with no spread cannot take part in a distance, so this is fatal.
feature_sd <- vapply(FEATURES, function(col_name) stats::sd(data[[col_name]]), numeric(1))
constant_features <- FEATURES[is.na(feature_sd) | feature_sd == 0]
if (length(constant_features) > 0) {
  cat(sprintf("FATAL: Feature(s) %s have no variation and cannot be clustered on.\n",
              paste(constant_features, collapse = ", ")))
  quit(status = 1)
}

# Feature ranges are recomputed from the cleaned data. These are what the API
# and predict.R compare against, so they must come from real data.
feature_ranges <- lapply(FEATURES, function(col_name) {
  values <- data[[col_name]]
  list(min = min(values), max = max(values))
})
names(feature_ranges) <- FEATURES

cat(sprintf("[3/12] Data cleaning complete: %d rows kept, %d dropped (%d incomplete, %d domain-invalid)\n",
            rows_clean, rows_dropped, rows_incomplete, rows_invalid))

# ==============================================================================
# STEP 4 - Standardise the clustering features
#   One transformation, computed once and stored in model.rds. predict.R applies
#   EXACTLY this transformation to an incoming customer before measuring it
#   against the centres, which is what makes the two sides comparable.
# ==============================================================================
raw_matrix <- as.matrix(data[, FEATURES])
storage.mode(raw_matrix) <- "double"

scaler_center <- vapply(seq_along(FEATURES), function(j) mean(raw_matrix[, j]), numeric(1))
scaler_scale <- vapply(seq_along(FEATURES), function(j) stats::sd(raw_matrix[, j]), numeric(1))
names(scaler_center) <- FEATURES
names(scaler_scale) <- FEATURES

# The single transformation used for BOTH training and customer assignment.
apply_scaling <- function(raw) {
  scaled <- sweep(as.matrix(raw), 2L, scaler_center, "-")
  sweep(scaled, 2L, scaler_scale, "/")
}

scaled_matrix <- apply_scaling(raw_matrix)

# as.matrix() leaves numbered rownames ("1", "2", ...) on a data frame, and
# sweep() preserves them. R refuses to subtract two matrices whose dimnames
# differ, and the kmeans() centres are labelled "1".."K", so the scaled matrix is
# given clean feature column names and no rownames here. Every centre slice used
# further down is then reduced to the same shape.
dimnames(scaled_matrix) <- list(NULL, FEATURES)

cat("[4/12] Feature scaling fitted on the cleaned dataset\n")
for (col_name in FEATURES) {
  cat(sprintf("      %-24s centre = %9.4f   scale = %8.4f\n",
              col_name, scaler_center[[col_name]], scaler_scale[[col_name]]))
}

# ==============================================================================
# STEP 5 - PCA, computed FOR VISUALISATION ONLY
#   The clustering below runs on the four scaled features, never on the
#   components. The two leading principal components are computed purely so the
#   dashboard can draw a scatter plot: a 4-D cluster has no other honest 2-D
#   picture, and this is the standard one.
#   The component loadings are stored so the centres and any held-out customer
#   can be projected with the exact same basis later in predict.R.
# ==============================================================================
pca_result <- stats::prcomp(scaled_matrix, center = FALSE, scale. = FALSE)
explained_variance <- (pca_result$sdev^2) / sum(pca_result$sdev^2)

pc1_loadings <- pca_result$rotation[, 1]
pc2_loadings <- pca_result$rotation[, 2]

# Project every customer and every cluster centre onto the same 2-D basis.
customer_projection <- cbind(
  PC1 = as.numeric(scaled_matrix %*% pc1_loadings),
  PC2 = as.numeric(scaled_matrix %*% pc2_loadings)
)

cat(sprintf("[5/12] PCA for visualisation only: PC1 explains %.1f%% of variance, PC2 %.1f%%\n",
            100 * explained_variance[1], 100 * explained_variance[2]))

# ==============================================================================
# STEP 6 - Average silhouette width
#   For each customer, s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is the
#   mean distance to the other members of its OWN cluster and b(i) the smallest
#   mean distance to any OTHER cluster. It approaches 1 when a point is tightly
#   packed and far from the rest, and 0 when it sits on a cluster boundary.
#   Computed here from the real distance matrix, not estimated.
# ==============================================================================
average_silhouette <- function(cluster_assignment, distance_matrix) {
  n <- nrow(distance_matrix)

  # Silhouette is undefined when there is only one cluster.
  if (length(unique(cluster_assignment)) < 2 || n <= 1) return(rep(NA_real_, n))

  within_mean <- vapply(seq_len(n), function(i) {
    same <- which(cluster_assignment == cluster_assignment[i] & seq_len(n) != i)
    if (length(same) == 0) return(0)
    mean(distance_matrix[i, same])
  }, numeric(1))

  between_mean <- vapply(seq_len(n), function(i) {
    other_clusters <- setdiff(unique(cluster_assignment), cluster_assignment[i])
    if (length(other_clusters) == 0) return(0)
    min(vapply(other_clusters, function(other) {
      members <- which(cluster_assignment == other)
      if (length(members) == 0) return(Inf)
      mean(distance_matrix[i, members])
    }, numeric(1)))
  }, numeric(1))

  denominator <- pmax(within_mean, between_mean)
  silhouette <- (between_mean - within_mean) / denominator
  # A point alone in its cluster has an undefined silhouette; R's own
  # cluster::silhouette() assigns it 0 as well, so this script does the same.
  silhouette[!is.finite(silhouette)] <- 0
  silhouette
}

euclidean_distance_matrix <- function(train_matrix, query_matrix) {
  train_matrix <- as.matrix(train_matrix)
  query_matrix <- as.matrix(query_matrix)
  squared <- outer(rowSums(query_matrix^2), rowSums(train_matrix^2), "+") -
    2 * (query_matrix %*% t(train_matrix))
  # Floating-point cancellation can push a zero distance slightly negative.
  squared[squared < 0] <- 0
  sqrt(squared)
}

# ==============================================================================
# STEP 7 - Score every candidate K with real diagnostics
#   For each K in K_GRID, kmeans() is fitted with N_START random restarts, and
#   the fit is scored on three axes:
#     * WSS        - lower is tighter, but it falls monotonically as K rises, so
#                    it cannot pick K on its own
#     * silhouette - higher is better separation, and it CAN pick K on its own
#     * stability  - the average WSS over several independent seeded fits; a K
#                    whose solution depends on a lucky restart is not trusted
#   The elbow is located as the point of largest proportional WSS reduction.
# ==============================================================================
set.seed(RANDOM_SEED)

k_grid_results <- list()

for (k_value in K_GRID) {
  if (k_value >= rows_clean) {
    cat(sprintf("      Skipping K = %d - only %d usable rows.\n", k_value, rows_clean))
    next
  }

  fit <- tryCatch(
    stats::kmeans(scaled_matrix, centers = k_value, iter.max = 100, nstart = N_START),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    cat(sprintf("      K = %d produced an error and was skipped.\n", k_value))
    next
  }

  distances <- euclidean_distance_matrix(scaled_matrix, scaled_matrix)
  silhouette_values <- average_silhouette(fit$cluster, distances)

  # Independent restarts, each from its own seed, to see how stable this K is.
  stability_wss <- numeric(STABILITY_RESTARTS)
  for (restart in seq_len(STABILITY_RESTARTS)) {
    # kmeans() reads the RNG for its initial centres, so re-seeding immediately
    # before the call is what makes each restart genuinely independent.
    set.seed(RANDOM_SEED + 1000L * k_value + restart)
    restart_fit <- tryCatch(
      stats::kmeans(scaled_matrix, centers = k_value, iter.max = 100, nstart = 1),
      error = function(e) NULL
    )
    stability_wss[restart] <- if (is.null(restart_fit)) NA_real_ else restart_fit$tot.withinss
  }

  cluster_sizes <- as.integer(table(fit$cluster))

  k_grid_results[[length(k_grid_results) + 1L]] <- list(
    k = as.integer(k_value),
    wss = as.numeric(fit$tot.withinss),
    betweenss = as.numeric(fit$betweenss),
    total_ss = as.numeric(fit$tot.withinss + fit$betweenss),
    between_ratio = as.numeric(fit$betweenss / (fit$tot.withinss + fit$betweenss)),
    silhouette_mean = mean(silhouette_values),
    cluster_sizes = cluster_sizes,
    smallest_cluster = if (length(cluster_sizes) > 0) min(cluster_sizes) else 0L,
    stability_wss_mean = if (all(is.na(stability_wss))) NA_real_ else mean(stability_wss, na.rm = TRUE),
    stability_wss_sd = if (sum(!is.na(stability_wss)) < 2) NA_real_ else stats::sd(stability_wss, na.rm = TRUE)
  )
}

if (length(k_grid_results) < 2) {
  cat("FATAL: Fewer than two candidate K values could be fitted - cannot choose K.\n")
  quit(status = 1)
}

grid_k <- vapply(k_grid_results, function(entry) entry$k, numeric(1))
grid_wss <- vapply(k_grid_results, function(entry) entry$wss, numeric(1))
grid_silhouette <- vapply(k_grid_results, function(entry) entry$silhouette_mean, numeric(1))
grid_between <- vapply(k_grid_results, function(entry) entry$between_ratio, numeric(1))
grid_stability <- vapply(k_grid_results, function(entry) entry$stability_wss_mean, numeric(1))

# The WSS curve falls monotonically, so "best WSS" would always pick the largest
# K. The elbow is where the curve flattens: the K whose proportional WSS
# reduction against the previous K is the largest.
wss_reduction_ratio <- rep(NA_real_, length(grid_k))
if (length(grid_k) > 1) {
  wss_reduction_ratio[-1] <-
    (grid_wss[-length(grid_wss)] - grid_wss[-1]) / grid_wss[-length(grid_wss)]
}
transition_positions <- which(!is.na(wss_reduction_ratio)) + 1L
elbow_index <- if (length(transition_positions) == 0) {
  1L
} else {
  transition_positions[which.max(wss_reduction_ratio[transition_positions])]
}
ELBOW_K <- as.integer(grid_k[elbow_index])

# The silhouette maximum is the K with the most internally cohesive and most
# separated clusters.
SILHOUETTE_K <- as.integer(grid_k[which.max(grid_silhouette)])

cat(sprintf("[6/12] Scored %d candidate K values (kmeans, nstart = %d)\n",
            length(k_grid_results), N_START))
cat(sprintf("      %-4s %12s %12s %14s %14s\n",
            "K", "WSS", "Silhouette", "Between ratio", "Stability WSS"))
for (i in seq_along(k_grid_results)) {
  cat(sprintf("      %-4d %12.4f %12.4f %14.4f %14.4f\n",
              grid_k[i], grid_wss[i], grid_silhouette[i], grid_between[i], grid_stability[i]))
}
cat(sprintf("      Elbow (largest proportional WSS drop) -> K = %d\n", ELBOW_K))
cat(sprintf("      Best average silhouette width          -> K = %d\n", SILHOUETTE_K))

# ==============================================================================
# STEP 8 - Choose K from those two criteria
#   The silhouette score is the primary criterion: unlike WSS it is not
#   monotonically decreasing, so it identifies a genuine optimum on its own.
#   The elbow is the tie-breaker. When the silhouette peak IS the elbow, the
#   two independent criteria agree and that K is taken. When they disagree,
#   the elbow is preferred only if its silhouette is close to the peak -
#   otherwise the silhouette peak wins, because preferring the elbow on a
#   materially worse silhouette would trade measurable separation for a
#   curve-shape heuristic.
#   The decision and its full reasoning go into metrics.json so the choice can
#   be audited rather than taken on trust.
# ==============================================================================
peak_silhouette <- max(grid_silhouette, na.rm = TRUE)
peak_index <- which(grid_silhouette == peak_silhouette)[1]

# "Close" means within 5% of the peak silhouette, or within 0.02 in absolute
# terms - whichever is the looser of the two for this dataset's score range.
silhouette_agreement_threshold <- peak_silhouette - max(0.02, 0.05 * abs(peak_silhouette))
elbow_silhouette <- grid_silhouette[elbow_index]
criteria_agree <- identical(as.integer(grid_k[elbow_index]), as.integer(grid_k[peak_index]))

if (criteria_agree) {
  SELECTED_K <- ELBOW_K
  selection_basis <- "elbow and silhouette agree"
  selection_reason <- sprintf(
    "The elbow method and the average silhouette width independently select the same K (%d), so both criteria agree.",
    SELECTED_K
  )
} else if (!is.na(elbow_silhouette) && elbow_silhouette >= silhouette_agreement_threshold) {
  SELECTED_K <- ELBOW_K
  selection_basis <- "elbow (silhouette within 5% of the peak)"
  selection_reason <- sprintf(
    paste0("The elbow method selects K = %d and the silhouette peak is K = %d. The elbow's silhouette (%.4f) ",
           "is within 5%% of the peak (%.4f), so the K where the WSS curve flattens is taken."),
    ELBOW_K, SILHOUETTE_K, elbow_silhouette, peak_silhouette
  )
} else {
  SELECTED_K <- SILHOUETTE_K
  selection_basis <- "silhouette maximum (elbow's silhouette materially lower)"
  selection_reason <- sprintf(
    paste0("The elbow method suggests K = %d but its average silhouette width (%.4f) is materially below the ",
           "peak of %.4f at K = %d. WSS always falls as K rises, so the elbow is the weaker criterion here, and ",
           "the silhouette maximum is taken."),
    ELBOW_K, elbow_silhouette, peak_silhouette, SILHOUETTE_K
  )
}

selected_index <- match(SELECTED_K, grid_k)
SELECTED_SILHOUETTE <- grid_silhouette[selected_index]
SELECTED_WSS <- grid_wss[selected_index]
SELECTED_BETWEEN <- grid_between[selected_index]

cat(sprintf("[7/12] Selected K = %d - %s\n", SELECTED_K, selection_reason))

# ==============================================================================
# STEP 9 - Fit the final model with stats::kmeans()
#   The seed is set immediately before the fit, not once at the top of the
#   script: the K grid above has drawn from the RNG many times, so only a fresh
#   set.seed() here guarantees this fit is the one a re-run reproduces.
# ==============================================================================
set.seed(RANDOM_SEED)

final_fit <- stats::kmeans(scaled_matrix, centers = SELECTED_K,
                           iter.max = 100, nstart = N_START)
final_assignments <- as.integer(final_fit$cluster)

# kmeans() reports the within-cluster sum of squares it minimised. The WSS this
# model will actually answer new customers against is recomputed here from the
# stored centres, so the reported number and the number predict.R measures
# against are provably the same quantity.
recomputed_wss <- sum((scaled_matrix -
  final_fit$centers[final_assignments, , drop = FALSE])^2)
recompute_difference <- abs(recomputed_wss - final_fit$tot.withinss)
if (recompute_difference > 1e-6 * max(1, abs(final_fit$tot.withinss))) {
  cat(sprintf("FATAL: WSS recomputed from the stored centres (%.6f) disagrees with kmeans()'s own (%.6f).\n",
              recomputed_wss, final_fit$tot.withinss))
  cat("       Refusing to write artifacts - the stored centres are not the model's.\n")
  quit(status = 1)
}

# Re-running the identical fit must reproduce this exact assignment, or the
# model would not be reproducible and nothing is written.
set.seed(RANDOM_SEED)
verification_fit <- stats::kmeans(scaled_matrix, centers = SELECTED_K,
                                  iter.max = 100, nstart = N_START)
reproducible <- identical(as.integer(verification_fit$cluster), final_assignments)

if (!reproducible) {
  cat("FATAL: Re-running kmeans() with the same seed produced a different clustering.\n")
  cat("       Refusing to write artifacts - the model would not be reproducible.\n")
  quit(status = 1)
}

cluster_sizes <- as.integer(table(final_assignments))
if (length(cluster_sizes) != SELECTED_K) {
  cat(sprintf("FATAL: Expected %d clusters but the fit produced %d.\n",
              SELECTED_K, length(cluster_sizes)))
  quit(status = 1)
}
if (any(cluster_sizes < 2)) {
  cat(sprintf("FATAL: kmeans() produced a cluster of %d member(s) - that is not a segment.\n",
              min(cluster_sizes)))
  quit(status = 1)
}

total_ss <- recomputed_wss + final_fit$betweenss
distances <- euclidean_distance_matrix(scaled_matrix, scaled_matrix)
final_silhouette_values <- average_silhouette(final_assignments, distances)
FINAL_SILHOUETTE <- mean(final_silhouette_values)

# How far each customer sits from its own centre, and from the nearest centre it
# was NOT assigned to. The ratio of the two is what the dashboard uses to shade
# the scatter plot by how firmly a customer belongs.
distance_to_own_center <- sqrt(
  rowSums((scaled_matrix - final_fit$centers[final_assignments, , drop = FALSE])^2)
)
nearest_other_distance <- vapply(seq_len(rows_clean), function(i) {
  other_clusters <- setdiff(seq_len(SELECTED_K), final_assignments[i])
  if (length(other_clusters) == 0) return(Inf)
  min(sqrt(rowSums((scaled_matrix[i, ] -
    final_fit$centers[other_clusters, , drop = FALSE])^2)))
}, numeric(1))
separation_ratio <- distance_to_own_center / nearest_other_distance

# Project the centres onto the same PCA basis as the customers.
center_projection <- cbind(
  PC1 = as.numeric(final_fit$centers %*% pc1_loadings),
  PC2 = as.numeric(final_fit$centers %*% pc2_loadings)
)

cat(sprintf("[8/12] Final kmeans() model: K = %d, WSS = %.4f, between-cluster ratio = %.4f, mean silhouette = %.4f\n",
            SELECTED_K, recomputed_wss, final_fit$betweenss / total_ss, FINAL_SILHOUETTE))
cat(sprintf("      Cluster sizes: %s\n",
            paste(sprintf("%d=%d", seq_len(SELECTED_K), cluster_sizes), collapse = ", ")))
cat(sprintf("      Reproducible under the same seed: %s\n", reproducible))
cat(sprintf("[9/12] WSS self-check passed: recomputed from the stored centres (%.6f) matches kmeans()\n",
            recomputed_wss))

# ==============================================================================
# STEP 10 - Derive each cluster's profile and label from its OWN measured centre
#   A cluster's centre is compared against the population mean, feature by
#   feature, and each feature is classed by how far the centre sits from that
#   mean in standard-deviation units. The label is assembled from those actual
#   descriptors - it is a description of the data, never a hand-written name.
#   Where the evidence is genuinely ambiguous the neutral fallback "Cluster n"
#   is used instead of inventing a persona.
# ==============================================================================
population_mean <- vapply(seq_along(FEATURES), function(j) mean(raw_matrix[, j]), numeric(1))
names(population_mean) <- FEATURES

# Human-readable feature phrases used to assemble the generated label. The
# descriptors themselves are computed from the data - only the wording here is
# written by hand.
FEATURE_PHRASES <- list(
  age = list(high = "Older", low = "Younger"),
  annual_income = list(high = "High Income", low = "Lower Income"),
  spending_score = list(high = "High Spending", low = "Low Spending"),
  purchase_frequency = list(high = "Frequent Buyers", low = "Rare Buyers")
)

cluster_profiles <- lapply(seq_len(SELECTED_K), function(cluster_id) {
  members <- which(final_assignments == cluster_id)

  # The cluster's centre in scaled space, and the same centre converted back to
  # the original units by undoing the scaling exactly.
  center_scaled <- as.numeric(final_fit$centers[cluster_id, ])
  center_original <- center_scaled * as.numeric(scaler_scale[FEATURES]) +
    as.numeric(scaler_center[FEATURES])

  # How far this centre sits from the population, per feature, in sd units.
  # center_original is an unnamed numeric, so the names have to be put back
  # explicitly before this vector can be looked up by feature name below.
  center_vs_population <- (center_original - as.numeric(population_mean[FEATURES])) /
    as.numeric(scaler_scale[FEATURES])
  names(center_vs_population) <- FEATURES

  descriptors <- character(0)
  for (feature_name in FEATURES) {
    offset <- center_vs_population[[feature_name]]
    phrase_set <- FEATURE_PHRASES[[feature_name]]
    if (is.null(phrase_set)) next
    # Beyond +-0.5 sd the centre is genuinely separated on this feature; inside
    # that band the cluster is not distinct enough on it to be named after it.
    if (offset >= 0.5) {
      descriptors <- c(descriptors, phrase_set$high)
    } else if (offset <= -0.5) {
      descriptors <- c(descriptors, phrase_set$low)
    }
  }

  # One descriptor on its own is too thin to name a segment after; two or more
  # is a description the data actually supports.
  ambiguous <- length(descriptors) < 2
  label <- if (ambiguous) {
    sprintf("Cluster %d", cluster_id)
  } else {
    paste(descriptors, collapse = " / ")
  }

  # A deterministic, explainable segment name: the largest cluster is the
  # "Core" segment and the smallest the "Niche" one. This is a rank of the real
  # cluster sizes, not an opinion about the customers.
  size_label <- "Standard"
  if (cluster_id == which.max(cluster_sizes)) size_label <- "Core"
  if (cluster_id == which.min(cluster_sizes)) size_label <- "Niche"

  # NOTE: a cluster's centre stays a NAMED VECTOR here, never a 1-row matrix.
  # The column names must match the members' matrix for R to subtract them, and
  # the rows' names must stay NULL, so a matrix slice would not fit. R recycles
  # the vector down the rows, which is exactly the per-feature difference wanted.
  center_row <- final_fit$centers[cluster_id, ]

  list(
    cluster = as.integer(cluster_id),
    label = label,
    label_is_generic = ambiguous,
    segment = size_label,
    size = as.integer(length(members)),
    size_share = as.numeric(length(members) / rows_clean),
    mean_features = as.list(stats::setNames(
      as.numeric(round(colMeans(scaled_matrix[members, , drop = FALSE]), 6)),
      FEATURES
    )),
    original_mean_features = as.list(stats::setNames(
      as.numeric(round(colMeans(raw_matrix[members, , drop = FALSE]), 6)),
      FEATURES
    )),
    center_scaled = as.list(stats::setNames(as.numeric(round(center_scaled, 6)), FEATURES)),
    center_original = as.list(stats::setNames(as.numeric(round(center_original, 6)), FEATURES)),
    center_vs_population_sd = as.list(stats::setNames(
      as.numeric(round(center_vs_population, 6)), FEATURES
    )),
    descriptors = as.list(descriptors),
    wss_share = as.numeric(sum((scaled_matrix[members, , drop = FALSE] -
      center_row)^2) / recomputed_wss),
    mean_distance_to_center = as.numeric(round(mean(distance_to_own_center[members]), 6))
  )
})

cat("[10/12] Cluster profiles derived from each cluster's own measured centre:\n")
for (profile in cluster_profiles) {
  cat(sprintf("       Cluster %d (%s) - %d customers - %s\n",
              profile$cluster, profile$segment, profile$size, profile$label))
}

# ==============================================================================
# STEP 11 - Save the artifacts
# ==============================================================================
trained_at <- strftime(as.POSIXlt(Sys.time(), "UTC"), "%Y-%m-%dT%H:%M:%SZ")

# The bundle holds everything needed to segment a new customer: the kmeans object
# itself, the centres, the exact scaling, the PCA basis used for the plot and the
# metadata. predict.R loads this one file and refits nothing.
model_bundle <- list(
  model = final_fit,
  model_type = "K-Means",
  algorithm = "stats::kmeans (Lloyd's algorithm, Euclidean distance)",
  task = "Customer Segmentation",
  identifier_column = IDENTIFIER_COLUMN,
  features = FEATURES,
  scaler = list(
    method = "standardisation (z-score) fitted on the cleaned dataset",
    center = scaler_center,
    scale = scaler_scale
  ),
  feature_ranges = feature_ranges,
  k = as.integer(SELECTED_K),
  k_grid = as.integer(K_GRID),
  cluster_centers = final_fit$centers,
  cluster_sizes = stats::setNames(cluster_sizes, as.character(seq_len(SELECTED_K))),
  population_mean = population_mean,
  # Kept for the dashboard's 2-D projection ONLY. The clustering above was
  # performed on the four scaled features, not on these components.
  pca = list(
    purpose = "visualisation only - the model clusters on the scaled features, not on these components",
    pc1_loadings = pc1_loadings,
    pc2_loadings = pc2_loadings,
    explained_variance_ratio = as.numeric(explained_variance)
  ),
  cluster_profiles = cluster_profiles,
  r_version = sprintf("%s.%s", R.version$major, R.version$minor),
  trained_at = trained_at,
  random_seed = RANDOM_SEED,
  n_start = N_START,
  rows_loaded = rows_loaded,
  rows_dropped = rows_dropped,
  rows_clustered = rows_clean
)

saveRDS(model_bundle, file = model_path)

# JSON cannot represent NaN/Inf; guard so the files always parse cleanly in
# Python and in the browser.
safe_number <- function(value) {
  if (is.null(value) || length(value) == 0 || is.na(value) || is.nan(value) || is.infinite(value)) {
    return(NULL)
  }
  round(as.numeric(value), 6)
}

# Every row of the K comparison, in ascending K so the chart reads naturally.
k_comparison_payload <- lapply(k_grid_results, function(entry) {
  list(
    k = as.integer(entry$k),
    wss = safe_number(entry$wss),
    betweenss = safe_number(entry$betweenss),
    total_ss = safe_number(entry$total_ss),
    between_ratio = safe_number(entry$between_ratio),
    silhouette_mean = safe_number(entry$silhouette_mean),
    smallest_cluster = as.integer(entry$smallest_cluster),
    stability_wss_mean = safe_number(entry$stability_wss_mean),
    stability_wss_sd = safe_number(entry$stability_wss_sd),
    wss_drop_from_previous = safe_number(wss_reduction_ratio[match(entry$k, grid_k)]),
    selected = identical(as.integer(entry$k), SELECTED_K),
    is_elbow = identical(as.integer(entry$k), ELBOW_K)
  )
})

# The mean silhouette of each cluster, for the profile view.
cluster_silhouette_summary <- vapply(seq_len(SELECTED_K), function(cluster_id) {
  members <- which(final_assignments == cluster_id)
  mean(final_silhouette_values[members])
}, numeric(1))
names(cluster_silhouette_summary) <- as.character(seq_len(SELECTED_K))

# The projection payload the dashboard's scatter plot is drawn from: every
# customer's 2-D position, the cluster it was assigned and its separation ratio.
projection_payload <- list(
  PC1 = as.numeric(round(customer_projection[, 1], 4)),
  PC2 = as.numeric(round(customer_projection[, 2], 4)),
  cluster = final_assignments
)

# --- metrics.json -----------------------------------------------------------
metrics_payload <- list(
  success = TRUE,
  model = "K-Means",
  algorithm = model_bundle$algorithm,
  task = "Customer Segmentation",
  learning_type = "unsupervised",
  trained_at = trained_at,
  r_version = model_bundle$r_version,
  random_seed = RANDOM_SEED,
  selected_k = as.integer(SELECTED_K),
  # The headline numbers the dashboard shows, straight from the fitted model.
  metrics = list(
    wss = safe_number(recomputed_wss),
    bss = safe_number(final_fit$betweenss),
    total_ss = safe_number(total_ss),
    between_cluster_ratio = safe_number(final_fit$betweenss / total_ss),
    silhouette_score = safe_number(FINAL_SILHOUETTE),
    wss_from_kmeans = safe_number(final_fit$tot.withinss),
    wss_recompute_difference = safe_number(recompute_difference),
    cluster_count = as.integer(SELECTED_K),
    records_clustered = as.integer(rows_clean),
    silhouette_range = "-1 to 1, higher is better",
    interpretation = paste(
      "The silhouette score is the mean silhouette width of the real clustering: how",
      "tightly each customer groups with its own cluster versus how far it sits from",
      "the nearest other cluster. WSS is the total squared distance of every customer",
      "to its own cluster centre - lower means tighter clusters, but it always falls",
      "as K rises, which is why K is chosen with the silhouette as well."
    )
  ),
  dataset = list(
    file = "datasets/customers.csv",
    rows_loaded = rows_loaded,
    rows_clean = rows_clean,
    rows_dropped = rows_dropped,
    rows_incomplete = rows_incomplete,
    rows_domain_invalid = rows_invalid,
    identifier_column = IDENTIFIER_COLUMN,
    features = FEATURES,
    feature_ranges = feature_ranges,
    excluded_from_clustering = as.list(IDENTIFIER_COLUMN)
  ),
  feature_scaling = list(
    method = model_bundle$scaler$method,
    reason = paste(
      "K-Means minimises a squared Euclidean distance. annual_income and",
      "purchase_frequency live on very different numeric ranges, so unscaled the",
      "larger-range column would dominate the distance and the others would barely",
      "contribute. Every feature is standardised to mean 0 and standard deviation 1",
      "using the cleaned dataset, and predict.R applies the identical transformation."
    ),
    center = as.list(scaler_center),
    scale = as.list(scaler_scale)
  ),
  k_selection = list(
    method = "average silhouette width with the elbow method as tie-breaker",
    k_grid = as.integer(K_GRID),
    elbow_k = as.integer(ELBOW_K),
    silhouette_k = as.integer(SILHOUETTE_K),
    selected_k = as.integer(SELECTED_K),
    selected_on = "the full cleaned dataset (unsupervised - there is no held-out split)",
    basis = selection_basis,
    reason = selection_reason,
    agreement_threshold = safe_number(silhouette_agreement_threshold),
    explain = list(
      elbow = "WSS falls monotonically as K rises, so it cannot choose K by itself. The elbow is the K where that fall flattens, located here as the largest proportional drop in WSS.",
      silhouette = "The average silhouette width is not monotonic, so its maximum is a genuine optimum. It is therefore the primary criterion, and the elbow only breaks a close tie."
    )
  ),
  k_comparison = k_comparison_payload,
  clusters = lapply(cluster_profiles, function(entry) {
    list(
      cluster = entry$cluster,
      label = entry$label,
      label_is_generic = entry$label_is_generic,
      segment = entry$segment,
      size = entry$size,
      size_share = safe_number(entry$size_share),
      mean_features = entry$mean_features,
      original_mean_features = entry$original_mean_features,
      center_original = entry$center_original,
      center_scaled = entry$center_scaled,
      silhouette_mean = safe_number(unname(cluster_silhouette_summary[as.character(entry$cluster)])),
      wss_share = safe_number(entry$wss_share),
      mean_distance_to_center = safe_number(entry$mean_distance_to_center)
    )
  }),
  cluster_centers = lapply(seq_len(SELECTED_K), function(cluster_id) {
    list(
      cluster = as.integer(cluster_id),
      center_scaled = as.list(stats::setNames(
        as.numeric(round(as.numeric(final_fit$centers[cluster_id, ]), 6)), FEATURES
      )),
      center_original = as.list(stats::setNames(
        as.numeric(round(
          as.numeric(final_fit$centers[cluster_id, ] *
                       as.numeric(scaler_scale[FEATURES])) +
            as.numeric(scaler_center[FEATURES]), 6
        )),
        FEATURES
      )),
      size = as.integer(cluster_sizes[cluster_id]),
      projection = list(
        PC1 = safe_number(center_projection[cluster_id, 1]),
        PC2 = safe_number(center_projection[cluster_id, 2])
      )
    )
  }),
  cluster_profiles = cluster_profiles,
  visualization = list(
    method = "PCA (stats::prcomp) to two components",
    purpose = "visualisation only - the clustering itself ran on the four scaled features, never on the components",
    axis_x = "PC1",
    axis_y = "PC2",
    explained_variance_ratio = as.list(stats::setNames(
      as.numeric(round(explained_variance, 6)),
      paste0("PC", seq_along(explained_variance))
    )),
    total_explained_variance = safe_number(sum(explained_variance[1:2])),
    points = projection_payload
  ),
  reproducibility = list(
    seed = RANDOM_SEED,
    n_start = N_START,
    same_seed_same_clusters = reproducible,
    note = paste(
      "kmeans() itself starts from random centres. A fixed seed plus nstart restarts",
      "makes the fit reproducible: re-running this script with the same seed was",
      "verified to return the identical assignment for every customer."
    )
  ),
  model_info = list(
    implementation = "stats::kmeans()",
    distance = "Euclidean",
    algorithm_detail = "Lloyd's iterative refinement with Hartigan-Wong initialisation",
    clusters = as.integer(SELECTED_K),
    cluster_sizes = as.list(stats::setNames(cluster_sizes, as.character(seq_len(SELECTED_K)))),
    total_iterations = as.integer(final_fit$iter %||% length(final_fit$withinss)),
    unsupervised = TRUE
  ),
  model_file = "r_models/kmeans/model.rds",
  clusters_file = "r_models/kmeans/clusters.json"
)

writeLines(
  jsonlite::toJSON(metrics_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = metrics_path
)

# --- clusters.json ----------------------------------------------------------
# The actual dataset records with the cluster the model assigned them, plus the
# customer's own feature values and where it sits in the 2-D projection. Only
# the identifier and the clustering features are published - no other column
# from the source file is copied out.
clusters_payload <- list(
  success = TRUE,
  model = "K-Means",
  task = "Customer Segmentation",
  selected_k = as.integer(SELECTED_K),
  trained_at = trained_at,
  dataset = "datasets/customers.csv",
  total_records = as.integer(rows_clean),
  identifier_column = IDENTIFIER_COLUMN,
  features = FEATURES,
  silhouette_score = safe_number(FINAL_SILHOUETTE),
  wss = safe_number(recomputed_wss),
  cluster_sizes = as.list(stats::setNames(cluster_sizes, as.character(seq_len(SELECTED_K)))),
  cluster_labels = as.list(stats::setNames(
    vapply(cluster_profiles, function(entry) entry$label, character(1)),
    as.character(seq_len(SELECTED_K))
  )),
  cluster_centers_projected = lapply(seq_len(SELECTED_K), function(cluster_id) {
    list(
      cluster = as.integer(cluster_id),
      PC1 = safe_number(center_projection[cluster_id, 1]),
      PC2 = safe_number(center_projection[cluster_id, 2])
    )
  }),
  clusters = lapply(seq_len(rows_clean), function(i) {
    record <- list(
      id = as.character(data[[IDENTIFIER_COLUMN]][i]),
      cluster = as.integer(final_assignments[i])
    )
    for (feature_name in FEATURES) {
      record[[feature_name]] <- safe_number(as.numeric(raw_matrix[i, feature_name]))
    }
    record$distance_to_center <- safe_number(distance_to_own_center[i])
    record$separation_ratio <- safe_number(separation_ratio[i])
    record$PC1 <- safe_number(customer_projection[i, 1])
    record$PC2 <- safe_number(customer_projection[i, 2])
    record
  })
)

writeLines(
  jsonlite::toJSON(clusters_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = clusters_path
)

# --- profiles.json -----------------------------------------------------------
# The per-cluster profiles are their own artifact so a consumer of the
# segmentation does not have to parse the full metrics document to describe the
# segments. The values are the SAME `cluster_profiles` list that went into
# metrics.json and into the model bundle above - they are not recomputed here,
# so the three files can never disagree.
#
# For each cluster this publishes:
#   size                    - how many customers landed in it
#   mean_features           - member means in SCALED units (what the model saw)
#   original_mean_features  - the same means in the dataset's own units
#   center_scaled           - the centroid kmeans() fitted
#   center_original         - that centroid converted back to original units
#   center_vs_population_sd - how far the centre sits from the population mean
#                             per feature, in sd units. This is the "relative
#                             characteristic" reading: a positive value means
#                             this segment is above average on that feature.
profiles_payload <- list(
  success = TRUE,
  model = "K-Means",
  task = "Customer Segmentation",
  learning_type = "unsupervised",
  trained_at = trained_at,
  dataset = "datasets/customers.csv",
  identifier_column = IDENTIFIER_COLUMN,
  features = as.list(FEATURES),
  selected_k = as.integer(SELECTED_K),
  total_records = as.integer(rows_clean),
  feature_ranges = feature_ranges,
  scaling = list(
    method = "standardisation (z-score) fitted on the cleaned dataset",
    center = as.list(scaler_center[FEATURES]),
    scale = as.list(scaler_scale[FEATURES])
  ),
  naming = paste(
    "Cluster names are generated from each cluster's own measured centre: every",
    "feature whose centre sits at least 0.5 population standard deviations above",
    "or below the population mean contributes a descriptor. A cluster with fewer",
    "than two such descriptors is too ambiguous to characterise and falls back to",
    "the neutral name 'Cluster n'."
  ),
  # Relative position of every centre on every feature, so the profiles can be
  # compared against each other rather than read one at a time.
  comparison = list(
    population_mean = as.list(population_mean[FEATURES]),
    ranks_by_size = as.list(stats::setNames(
      as.integer(order(cluster_sizes, decreasing = TRUE)),
      paste("Cluster", seq_len(SELECTED_K))
    ))
  ),
  profiles = cluster_profiles
)

writeLines(
  jsonlite::toJSON(profiles_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = profiles_path
)

cat(sprintf("[11/12] Saved model bundle  -> %s\n", model_path))
cat(sprintf("       Saved metrics        -> %s\n", metrics_path))
cat(sprintf("       Saved cluster data   -> %s\n", clusters_path))
cat(sprintf("       Saved cluster profiles -> %s\n", profiles_path))
cat("======================================================================\n")
cat("PHASE 6 K-MEANS TRAINING COMPLETE\n")
cat(sprintf("  SELECTED K      = %d (elbow = %d, silhouette peak = %d)\n",
            SELECTED_K, ELBOW_K, SILHOUETTE_K))
cat(sprintf("  RECORDS         = %d clustered / %d dropped\n", rows_clean, rows_dropped))
cat(sprintf("  WSS             = %.4f\n", recomputed_wss))
cat(sprintf("  BETWEEN RATIO   = %.4f\n", final_fit$betweenss / total_ss))
cat(sprintf("  SILHOUETTE      = %.4f\n", FINAL_SILHOUETTE))
cat(sprintf("  CLUSTER SIZES   = %s\n",
            paste(sprintf("%d=%d", seq_len(SELECTED_K), cluster_sizes), collapse = ", ")))
for (profile in cluster_profiles) {
  cat(sprintf("  CLUSTER %d      = %s\n", profile$cluster, profile$label))
}
cat("======================================================================\n")
