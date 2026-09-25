# ==============================================================================
# AI Insight Hub — Phase 5
# K-Nearest Neighbors Training Script (Student Performance Prediction)
#
# WHAT THIS SCRIPT DOES
#   1. Loads datasets/students.csv
#   2. Validates that all required columns are present
#   3. Cleans invalid / missing data (non-numeric, NA, out-of-domain values)
#   4. Splits the cleaned data into a STRATIFIED training set and test set
#   5. Standardises the features using TRAINING-set centre and scale only
#   6. Chooses K by repeated stratified cross-validation on the TRAINING data
#   7. Fits the real classifier with class::knn()
#   8. Predicts on the held-out test set
#   9. Calculates real metrics: accuracy, confusion matrix, macro precision,
#      recall and F1 score
#  10. Verifies that the nearest-neighbour extraction it reports really is the
#      one class::knn() used, then saves the artifacts
#
# WHY FEATURES ARE SCALED
#   KNN measures a straight-line (Euclidean) distance between records. In
#   students.csv `assignments_completed` ranges 1-10 while `previous_score`
#   ranges 36-96, and after centring `previous_score` varies far more than
#   `assignments_completed` does. Left unscaled, the two large-variance columns
#   would silently dominate the distance and the small ones would contribute
#   almost nothing. Standardising every column to mean 0 / sd 1 puts all five
#   features on the same footing, so each one contributes what it should.
#
# NOTHING IS HARD-CODED. Every number in metrics.json comes from the model this
# script actually fits, and every class in the confusion matrix comes from
# class::knn()'s own predictions.
#
# A NOTE ON THIS DATASET'S CLASS IMBALANCE
#   students.csv contains 120 rows: MEDIUM=78, HIGH=41 and LOW=1. A classifier
#   needs at least one training row per class, so that single LOW row is
#   necessarily used for training and the held-out test set contains NO LOW rows
#   at all. This script therefore reports TWO macro averages:
#     * metrics.precision / recall / f1_score average over all THREE declared
#       classes, which is the conservative headline number (LOW scores 0);
#     * metrics.macro_supported_only averages over just the classes that
#       actually have test support, so the MEDIUM/HIGH quality is visible too.
#   Both are real, computed from the same predictions — nothing is invented, and
#   the class distribution and per-class support are reported alongside them.
#
# RUN IT WITH:
#   Rscript r_models/knn/train.R
#
# This is an educational ML implementation, not a real assessment tool.
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

has_class <- requireNamespace("class", quietly = TRUE)
if (!has_class) {
  cat("FATAL: The 'class' package is required to train the KNN classifier.\n")
  cat("Install it with: install.packages('class')\n")
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

dataset_path <- file.path(project_root, "datasets", "students.csv")
model_dir <- script_dir

# Paths for the artifacts this script produces
model_path <- file.path(model_dir, "model.rds")
metrics_path <- file.path(model_dir, "metrics.json")

# --- Model configuration -----------------------------------------------------
TARGET <- "performance"
FEATURES <- c("study_hours", "attendance", "previous_score",
              "assignments_completed", "practical_score")
REQUIRED_COLUMNS <- c(FEATURES, TARGET)
# Fixed class order, so every class index means the same thing everywhere.
CLASS_LEVELS <- c("LOW", "MEDIUM", "HIGH")
TRAIN_FRACTION <- 0.8
RANDOM_SEED <- 42   # Fixed seed so the split (and therefore the metrics) reproduce

# K values actually evaluated. K is NOT guessed: each one is scored by repeated
# stratified cross-validation on the TRAINING data only, below.
K_GRID <- c(3, 5, 7, 9)
CV_FOLDS <- 5
CV_REPEATS <- 10

# ==============================================================================
# STEP 1 — Load the dataset
# ==============================================================================
if (!file.exists(dataset_path)) {
  cat(sprintf("FATAL: Dataset not found at %s\n", dataset_path))
  quit(status = 1)
}

raw_data <- read.csv(dataset_path, stringsAsFactors = FALSE, check.names = FALSE)
rows_loaded <- nrow(raw_data)
cat(sprintf("[1/11] Loaded dataset: %s (%d rows)\n", dataset_path, rows_loaded))

# ==============================================================================
# STEP 2 — Validate required columns
# ==============================================================================
missing_columns <- setdiff(REQUIRED_COLUMNS, names(raw_data))
if (length(missing_columns) > 0) {
  cat(sprintf("FATAL: Dataset is missing required column(s): %s\n",
              paste(missing_columns, collapse = ", ")))
  quit(status = 1)
}
cat(sprintf("[2/11] Column validation passed. Required columns present: %s\n",
            paste(REQUIRED_COLUMNS, collapse = ", ")))

# ==============================================================================
# STEP 3 — Clean / validate the data
#   read.csv() may parse a dirty numeric column as character, so every feature
#   is coerced to numeric and any row that cannot be made fully valid is
#   dropped (and counted, never silently ignored).
# ==============================================================================
data <- raw_data[, REQUIRED_COLUMNS]

for (col_name in FEATURES) {
  if (!is.numeric(data[[col_name]])) {
    data[[col_name]] <- suppressWarnings(as.numeric(as.character(data[[col_name]])))
  }
}

# Normalise the target to upper case and trim whitespace so that "low" and
# "LOW" cannot silently become two different classes.
data[[TARGET]] <- toupper(trimws(as.character(data[[TARGET]])))

is_complete <- stats::complete.cases(data)
rows_incomplete <- sum(!is_complete)
data <- data[is_complete, ]

# Domain rules that a genuine student record must satisfy. These are the
# documented schema of the dataset, not values copied from any model output.
is_valid_row <- data$study_hours >= 0 &
               data$attendance >= 0 & data$attendance <= 100 &
               data$previous_score >= 0 & data$previous_score <= 100 &
               data$assignments_completed >= 0 &
               data$practical_score >= 0 & data$practical_score <= 100
rows_invalid <- sum(!is_valid_row)
data <- data[is_valid_row, ]

# Any target outside the documented class set makes the row unusable.
is_known_class <- data[[TARGET]] %in% CLASS_LEVELS
rows_unknown_class <- sum(!is_known_class)
data <- data[is_known_class, ]

rows_dropped <- rows_incomplete + rows_invalid + rows_unknown_class
rows_clean <- nrow(data)

if (rows_clean <= length(FEATURES) + 1) {
  cat(sprintf("FATAL: Not enough usable rows after cleaning (%d rows remain).\n", rows_clean))
  quit(status = 1)
}

# Convert the target to a factor with the fixed class order. class::knn() reads
# the class CODE off this factor, so the order must be deterministic.
data[[TARGET]] <- factor(data[[TARGET]], levels = CLASS_LEVELS)

# Every class must be present for a confusion matrix to be meaningful.
class_counts <- table(data[[TARGET]])
missing_classes <- setdiff(CLASS_LEVELS, names(class_counts))
if (length(missing_classes) > 0) {
  cat(sprintf("FATAL: Dataset contains no rows for class(es): %s\n",
              paste(missing_classes, collapse = ", ")))
  quit(status = 1)
}

# Recompute the feature ranges from the cleaned data. These ranges are what the
# API and the R prediction script use to flag out-of-range input, so they must
# come from real data and not from a hand-written constant.
feature_ranges <- lapply(FEATURES, function(col_name) {
  values <- data[[col_name]]
  list(min = min(values), max = max(values))
})
names(feature_ranges) <- FEATURES

cat(sprintf("[3/11] Data cleaning complete: %d rows kept, %d dropped (%d incomplete, %d domain-invalid, %d unknown class)\n",
            rows_clean, rows_dropped, rows_incomplete, rows_invalid, rows_unknown_class))
cat(sprintf("      Class distribution: %s\n",
            paste(sprintf("%s=%d", CLASS_LEVELS, as.integer(class_counts[CLASS_LEVELS])),
                  collapse = ", ")))

# ==============================================================================
# STEP 4 — Stratified train / test split
#   Splitting inside each class guarantees every class appears in the training
#   half. NOTE the sampling idiom: `class_rows[sample.int(length(class_rows))]`
#   shuffles the INDEX positions. Calling sample(class_rows) directly would be
#   wrong here, because R's sample() treats a length-1 vector as the sequence
#   1:n — for the single LOW row that silently manufactures a completely
#   different index and corrupts the split.
# ==============================================================================
set.seed(RANDOM_SEED)

train_index <- integer(0)
test_index <- integer(0)

for (class_name in CLASS_LEVELS) {
  class_rows <- which(data[[TARGET]] == class_name)
  shuffled <- class_rows[sample.int(length(class_rows))]

  # A class with a single row can supply a training row but no test row, so the
  # test share is only taken when at least one row can be spared.
  n_test <- if (length(shuffled) < 2L) {
    0L
  } else {
    max(1L, min(length(shuffled) - 1L, floor((1 - TRAIN_FRACTION) * length(shuffled))))
  }

  test_index <- c(test_index, shuffled[seq_len(n_test)])
  train_index <- c(train_index, shuffled[(n_test + 1L):length(shuffled)])
}

# Guard against the split silently overlapping or losing rows.
if (length(intersect(train_index, test_index)) > 0) {
  cat("FATAL: The stratified split put the same row in both the training and test sets.\n")
  quit(status = 1)
}
if (length(unique(c(train_index, test_index))) != rows_clean) {
  cat("FATAL: The stratified split did not use every cleaned row exactly once.\n")
  quit(status = 1)
}

train_data <- data[train_index, ]
test_data <- data[test_index, ]

# Every class must be represented in the TRAINING set, otherwise it can never
# be predicted at all.
train_classes <- unique(as.character(train_data[[TARGET]]))
absent_from_train <- setdiff(CLASS_LEVELS, train_classes)
if (length(absent_from_train) > 0) {
  cat(sprintf("FATAL: Stratified split failed — class(es) %s are missing from the training set.\n",
              paste(absent_from_train, collapse = ", ")))
  quit(status = 1)
}

train_class_table <- table(train_data[[TARGET]])
test_class_table <- table(test_data[[TARGET]])

cat(sprintf("[4/11] Stratified split: %d training rows / %d test rows (%d%% / %d%%)\n",
            nrow(train_data), nrow(test_data),
            round(100 * nrow(train_data) / rows_clean),
            round(100 * nrow(test_data) / rows_clean)))
cat(sprintf("      Training class counts: %s\n",
            paste(sprintf("%s=%d", CLASS_LEVELS, as.integer(train_class_table[CLASS_LEVELS])),
                  collapse = ", ")))
cat(sprintf("      Test class counts:     %s\n",
            paste(sprintf("%s=%d", CLASS_LEVELS, as.integer(test_class_table[CLASS_LEVELS])),
                  collapse = ", ")))

# ==============================================================================
# STEP 5 — Standardise the features (TRAINING SET ONLY)
#   The centre and scale are computed from the training half and then applied
#   unchanged to the test half. Fitting them on the full dataset would leak test
#   information into training, so they are stored and reused verbatim by
#   predict.R — the prediction path applies EXACTLY this transformation.
# ==============================================================================
train_raw <- as.matrix(train_data[, FEATURES])
test_raw <- as.matrix(test_data[, FEATURES])

scaler_center <- vapply(seq_along(FEATURES), function(j) mean(train_raw[, j]), numeric(1))
scaler_scale <- vapply(seq_along(FEATURES), function(j) {
  value <- stats::sd(train_raw[, j])
  # A constant column has no spread; scaling by 0 would divide by zero.
  if (is.na(value) || value == 0) 1 else value
}, numeric(1))
names(scaler_center) <- FEATURES
names(scaler_scale) <- FEATURES

# The single transformation used for BOTH training and prediction.
apply_scaling <- function(raw_matrix) {
  scaled <- sweep(as.matrix(raw_matrix), 2L, scaler_center, "-")
  sweep(scaled, 2L, scaler_scale, "/")
}

train_x <- apply_scaling(train_raw)
test_x <- apply_scaling(test_raw)
train_y <- train_data[[TARGET]]

cat(sprintf("[5/11] Feature scaling fitted on the training set only (%d rows)\n", nrow(train_data)))
for (col_name in FEATURES) {
  cat(sprintf("      %-24s centre = %9.4f   scale = %8.4f\n",
              col_name, scaler_center[[col_name]], scaler_scale[[col_name]]))
}

# ==============================================================================
# STEP 6 — Choose K by repeated stratified cross-validation (TRAINING DATA ONLY)
#   The held-out test set is NOT consulted when choosing K, so the test score
#   below stays an honest estimate of how the model generalises.
#
#   10 repeats of 5 stratified folds are used because a single 5-fold split on
#   this dataset is noisy. K is chosen with the standard one-standard-error
#   rule: among the K values that are statistically indistinguishable from the
#   best, the SMALLEST K is taken.
# ==============================================================================
confusion_from_predictions <- function(actual, predicted) {
  table(
    factor(as.character(actual), levels = CLASS_LEVELS),
    factor(as.character(predicted), levels = CLASS_LEVELS)
  )
}

macro_f1_from_confusion <- function(cm) {
  class_index <- seq_along(CLASS_LEVELS)
  f1_values <- vapply(class_index, function(i) {
    true_positive <- cm[i, i]
    predicted_total <- sum(cm[, i])
    actual_total <- sum(cm[i, ])
    class_precision <- if (predicted_total > 0) true_positive / predicted_total else 0
    class_recall <- if (actual_total > 0) true_positive / actual_total else 0
    if ((class_precision + class_recall) > 0) {
      2 * class_precision * class_recall / (class_precision + class_recall)
    } else {
      0
    }
  }, numeric(1))
  mean(f1_values)
}

# Assigns each training row a fold id, stratified by class.
make_cv_folds <- function(frame_data, folds, seed) {
  set.seed(seed)
  fold_id <- integer(nrow(frame_data))
  for (class_name in CLASS_LEVELS) {
    class_rows <- which(frame_data[[TARGET]] == class_name)
    if (length(class_rows) == 0) next
    shuffled <- class_rows[sample.int(length(class_rows))]
    fold_id[shuffled] <- rep(seq_len(folds), length.out = length(shuffled))
  }
  fold_id
}

# The same scaling routine, applied to a fold's own training portion so that no
# information from the fold being scored leaks into that fold's own scaling.
apply_scaling_with <- function(raw_matrix, center, scale) {
  sweep(sweep(as.matrix(raw_matrix), 2L, center, "-"), 2L, scale, "/")
}

grid_results <- list()

for (k_value in K_GRID) {
  fold_scores <- numeric(0)

  for (repeat_id in seq_len(CV_REPEATS)) {
    # A different seed per repeat re-shuffles the folds, so the score averages
    # over many possible splits rather than one lucky one.
    cv_folds <- make_cv_folds(train_data, CV_FOLDS, seed = RANDOM_SEED + repeat_id)

    for (fold in seq_len(CV_FOLDS)) {
      in_fold <- which(cv_folds == fold)
      out_fold <- which(cv_folds != fold)

      fold_train_raw <- train_raw[out_fold, , drop = FALSE]
      fold_valid_raw <- train_raw[in_fold, , drop = FALSE]
      fold_train_y <- train_y[out_fold]

      if (nrow(fold_valid_raw) == 0) next
      # A fold that cannot exercise more than one class carries no signal.
      if (length(unique(fold_train_y)) < 2) next

      fold_center <- vapply(seq_along(FEATURES), function(j) mean(fold_train_raw[, j]), numeric(1))
      fold_scale <- vapply(seq_along(FEATURES), function(j) {
        value <- stats::sd(fold_train_raw[, j])
        if (is.na(value) || value == 0) 1 else value
      }, numeric(1))

      fold_train_x <- apply_scaling_with(fold_train_raw, fold_center, fold_scale)
      fold_valid_x <- apply_scaling_with(fold_valid_raw, fold_center, fold_scale)

      fold_predictions <- class::knn(
        fold_train_x, fold_valid_x, fold_train_y,
        k = k_value, prob = TRUE
      )

      fold_scores <- c(
        fold_scores,
        macro_f1_from_confusion(
          confusion_from_predictions(train_y[in_fold], as.character(fold_predictions))
        )
      )
    }
  }

  if (length(fold_scores) == 0) next
  grid_results[[length(grid_results) + 1L]] <- list(
    k = as.integer(k_value),
    mean_f1 = mean(fold_scores),
    sd_f1 = stats::sd(fold_scores),
    se_f1 = stats::sd(fold_scores) / sqrt(length(fold_scores)),
    folds_evaluated = length(fold_scores)
  )
}

if (length(grid_results) == 0) {
  cat("FATAL: Cross-validation produced no usable folds — cannot choose K.\n")
  quit(status = 1)
}

grid_k <- vapply(grid_results, function(entry) entry$k, numeric(1))
grid_scores <- vapply(grid_results, function(entry) entry$mean_f1, numeric(1))
grid_se <- vapply(grid_results, function(entry) entry$se_f1, numeric(1))

# Best mean F1 defines the one-standard-error threshold.
best_index <- which.max(grid_scores)
se_threshold <- grid_scores[best_index] - grid_se[best_index]

# Among the K values statistically indistinguishable from the best, take the
# smallest K.
within_one_se <- which(grid_scores >= se_threshold)
selected_index <- within_one_se[which.min(grid_k[within_one_se])]

SELECTED_K <- as.integer(grid_results[[selected_index]]$k)

# Report the whole grid, best-scoring first.
ordering <- order(-grid_scores, grid_k)
cv_summary_payload <- lapply(grid_results[ordering], function(entry) {
  list(
    k = as.integer(entry$k),
    cv_f1_mean = round(entry$mean_f1, 6),
    cv_f1_sd = round(entry$sd_f1, 6),
    cv_f1_se = round(entry$se_f1, 6),
    folds_evaluated = as.integer(entry$folds_evaluated),
    selected = identical(as.integer(entry$k), SELECTED_K)
  )
})

cat(sprintf("[6/11] Cross-validated K in {%s} over %d repeats of %d stratified folds (training data only)\n",
            paste(K_GRID, collapse = ", "), CV_REPEATS, CV_FOLDS))
for (entry in grid_results[ordering]) {
  cat(sprintf("      K = %2d  mean CV macro F1 = %.4f  (sd %.4f, se %.4f, %d folds)%s\n",
              entry$k, entry$mean_f1, entry$sd_f1, entry$se_f1, entry$folds_evaluated,
              if (identical(as.integer(entry$k), SELECTED_K)) "   <- SELECTED" else ""))
}
cat(sprintf("      Best mean CV F1 = %.4f, 1-SE threshold = %.4f\n",
            grid_scores[best_index], se_threshold))
cat(sprintf("      Selected K = %d by the one-standard-error rule\n", SELECTED_K))

# ==============================================================================
# STEP 7 — Train the real KNN model with class::knn()
#   class::knn() is a lazy, instance-based learner: the "model" is the training
#   matrix plus its labels. Nothing is fitted here beyond storing the scaled
#   training set, its labels and K — the whole of which goes into model.rds, so
#   predict.R never has to retrain.
# ==============================================================================
knn_model <- list(
  train_x = train_x,
  train_y = train_y,
  k = SELECTED_K
)

# Predict on the held-out test set.
test_predictions <- class::knn(train_x, test_x, train_y, k = SELECTED_K, prob = TRUE)
test_predicted_classes <- as.character(test_predictions)
test_probabilities <- attr(test_predictions, "prob")
actual_classes <- as.character(test_data[[TARGET]])

cat(sprintf("[7/11] Generated %d test-set predictions with class::knn() (K = %d)\n",
            length(test_predicted_classes), SELECTED_K))

# ==============================================================================
# STEP 8 — Nearest-neighbour extraction
#   class::knn() returns only the predicted class and an optional proportion, so
#   the neighbours behind a prediction have to be recovered explicitly.
#
#   They are recovered with the SAME metric class::knn() uses internally —
#   Euclidean distance — computed as:
#       ||a - b||^2 = |a|^2 - 2*a.b + |b|^2
#   which is arithmetically identical to summing squared differences but avoids
#   an n x p loop. STEP 9 then PROVES, against class::knn()'s own output, that
#   this reproduces its predictions and its probabilities exactly. Until that
#   check passes, the neighbour data is treated as unverified.
# ==============================================================================
euclidean_distance_matrix <- function(train_matrix, query_matrix) {
  train_matrix <- as.matrix(train_matrix)
  query_matrix <- as.matrix(query_matrix)
  squared <- outer(rowSums(query_matrix^2), rowSums(train_matrix^2), "+") -
             2 * (query_matrix %*% t(train_matrix))
  # Floating-point cancellation can push a zero distance slightly negative.
  squared[squared < 0] <- 0
  sqrt(squared)
}

# Returns a k x n_query matrix of training-row indices, nearest first.
nearest_indices <- function(distance_matrix, k) {
  query_count <- nrow(distance_matrix)
  indices <- matrix(0L, nrow = k, ncol = query_count)
  for (i in seq_len(query_count)) {
    indices[, i] <- order(distance_matrix[i, ])[seq_len(k)]
  }
  indices
}

test_distances <- euclidean_distance_matrix(train_x, test_x)
test_neighbor_indices <- nearest_indices(test_distances, SELECTED_K)

cat(sprintf("[8/11] Extracted the %d nearest training observations for every test record\n",
            SELECTED_K))

# ==============================================================================
# STEP 9 — Self-check: the extracted neighbours must BE the ones class::knn()
#   used. If the majority class of the extracted neighbours disagrees with
#   class::knn()'s prediction, or the winning-class proportion disagrees with the
#   `prob` attribute it returns, then the neighbour data is not the model's and
#   nothing is written to disk.
# ==============================================================================
mismatches <- 0L
for (i in seq_len(nrow(test_x))) {
  neighbor_rows <- test_neighbor_indices[, i]
  neighbor_class_counts <- table(as.character(train_y[neighbor_rows]))
  majority <- as.character(names(sort(neighbor_class_counts, decreasing = TRUE))[1])
  derived_probability <- unname(max(neighbor_class_counts) / SELECTED_K)

  if (!identical(majority, test_predicted_classes[i])) {
    mismatches <- mismatches + 1L
  } else if (!is.null(test_probabilities) &&
             abs(derived_probability - test_probabilities[i]) > 1e-9) {
    mismatches <- mismatches + 1L
  }
}

if (mismatches > 0L) {
  cat(sprintf("FATAL: Extracted neighbours disagree with class::knn() on %d of %d test records.\n",
              mismatches, nrow(test_x)))
  cat("       Refusing to write metrics.json — the reported neighbours would not\n")
  cat("       be the neighbours the model actually used.\n")
  quit(status = 1)
}
cat("[9/11] Neighbour self-check passed: extracted neighbours reproduce\n")
cat(sprintf("       class::knn()'s predictions AND probabilities on %d/%d test records\n",
            nrow(test_x), nrow(test_x)))

# ==============================================================================
# STEP 10 — Calculate real evaluation metrics
#   Every metric below is derived from the confusion matrix built out of the
#   test-set predictions; none of them is a literal.
# ==============================================================================
# table(actual, predicted) with a fixed level order keeps the matrix square so
# that a class the model never predicted still contributes a zero row/column.
confusion <- table(
  factor(actual_classes, levels = CLASS_LEVELS),
  factor(test_predicted_classes, levels = CLASS_LEVELS)
)

total_records <- sum(confusion)
correct_predictions <- sum(diag(confusion))
accuracy <- correct_predictions / total_records

# Per-class precision / recall / F1. A class the model never predicted has a
# precision denominator of zero; it is reported as 0 rather than NaN so the
# metrics file always parses cleanly.
per_class <- list()
precision_values <- numeric(0)
recall_values <- numeric(0)
f1_values <- numeric(0)
supported_precision <- numeric(0)
supported_recall <- numeric(0)
supported_f1 <- numeric(0)

for (class_name in CLASS_LEVELS) {
  class_index <- match(class_name, CLASS_LEVELS)

  true_positive <- confusion[class_index, class_index]
  predicted_total <- sum(confusion[, class_index])   # column total
  actual_total <- sum(confusion[class_index, ])      # row total
  false_positive <- predicted_total - true_positive
  false_negative <- actual_total - true_positive

  class_precision <- if (predicted_total > 0) true_positive / predicted_total else 0
  class_recall <- if (actual_total > 0) true_positive / actual_total else 0
  class_f1 <- if ((class_precision + class_recall) > 0) {
    2 * class_precision * class_recall / (class_precision + class_recall)
  } else {
    0
  }

  per_class[[class_name]] <- list(
    precision = round(class_precision, 6),
    recall = round(class_recall, 6),
    f1_score = round(class_f1, 6),
    support = as.integer(actual_total),
    predicted = as.integer(predicted_total),
    true_positives = as.integer(true_positive),
    false_positives = as.integer(false_positive),
    false_negatives = as.integer(false_negative)
  )

  precision_values <- c(precision_values, class_precision)
  recall_values <- c(recall_values, class_recall)
  f1_values <- c(f1_values, class_f1)

  # The same three numbers restricted to classes that actually have test rows.
  if (actual_total > 0) {
    supported_precision <- c(supported_precision, class_precision)
    supported_recall <- c(supported_recall, class_recall)
    supported_f1 <- c(supported_f1, class_f1)
  }
}

# Student performance is a multi-class problem with no single "positive" class,
# so the headline precision / recall / F1 are the macro averages across all
# THREE declared classes. The per-class numbers are kept in the payload as well.
macro_precision <- mean(precision_values)
macro_recall <- mean(recall_values)
macro_f1 <- mean(f1_values)

# The same averages over only the classes that have held-out support. This is
# reported ALONGSIDE the headline, never instead of it: a class with no test rows
# contributes 0 to the macro average, and saying so is more honest than quietly
# dropping it.
supported_macro_precision <- if (length(supported_precision) > 0) mean(supported_precision) else NULL
supported_macro_recall <- if (length(supported_recall) > 0) mean(supported_recall) else NULL
supported_macro_f1 <- if (length(supported_f1) > 0) mean(supported_f1) else NULL

classes_without_test_support <- CLASS_LEVELS[
  as.integer(test_class_table[CLASS_LEVELS]) == 0L
]

# ==============================================================================
#   Class REACHABILITY analysis
#   A KNN majority vote only returns a class when at least ceil(K/2) of the K
#   nearest training rows carry that class. So a class whose training set holds
#   fewer than ceil(K/2) rows can NEVER be predicted, no matter what input
#   arrives.
#
#   students.csv has exactly ONE LOW row, so at the selected K=5 a LOW answer
#   would need 3 LOW neighbours and is arithmetically impossible. The model is
#   therefore genuinely a MEDIUM/HIGH classifier. This is computed, not assumed,
#   and reported so the limitation is visible rather than discovered later as a
#   mysterious "why did it never say LOW?".
# ==============================================================================
# Guards against ties: a tie is broken by the neighbour that is nearest first,
# so reaching ceil(K/2) is the practical condition.
class_reachability <- lapply(CLASS_LEVELS, function(class_name) {
  class_index <- match(class_name, CLASS_LEVELS)
  # train_class_table is already ordered by CLASS_LEVELS, so the class index is
  # also the table position. Indexing by NAME here (train_class_table[class_name])
  # would silently return NA and mark every class unreachable.
  available <- as.integer(train_class_table[class_index])
  votes_needed <- ceiling(SELECTED_K / 2)
  list(
    class = class_name,
    training_rows = available,
    test_support = as.integer(test_class_table[class_index]),
    votes_needed_to_win = as.integer(votes_needed),
    reachable = as.integer(available >= votes_needed),
    note = if (available >= votes_needed) {
      sprintf("%s needs %d of the K nearest training rows to be %s and %d such rows exist.",
              class_name, votes_needed, class_name, available)
    } else {
      sprintf(paste0("%s cannot be predicted at K=%d: a majority vote needs %d of the K nearest ",
                     "training rows to be %s, but the dataset holds only %d."),
              class_name, SELECTED_K, votes_needed, class_name, available)
    }
  )
})
names(class_reachability) <- CLASS_LEVELS

unreachable_classes <- CLASS_LEVELS[
  vapply(class_reachability, function(entry) !isTRUE(as.logical(entry$reachable)), logical(1))
]

cat(sprintf("      Class reachability at K = %d (a majority vote needs ceil(K/2) = %d neighbours):\n",
            SELECTED_K, ceiling(SELECTED_K / 2)))
for (entry in class_reachability) {
  cat(sprintf("        %-8s training rows = %3d  ->  %s\n",
              entry$class, entry$training_rows,
              if (isTRUE(as.logical(entry$reachable))) "reachable" else "UNREACHABLE"))
}

# Balanced accuracy = mean of the per-class recalls; it is reported because a
# headline accuracy alone is misleading on an imbalanced dataset.
balanced_accuracy <- mean(recall_values)
error_rate <- 1 - accuracy

# A model that always predicted the most common class would also look accurate
# on an imbalanced test set, so that baseline is recorded alongside the real
# score to make the accuracy interpretable rather than misleading.
majority_class <- names(sort(test_class_table, decreasing = TRUE))[1]
majority_accuracy <- as.numeric(test_class_table[[majority_class]]) / total_records

cat(sprintf("[10/11] Test metrics -> Accuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f\n",
            accuracy, macro_precision, macro_recall, macro_f1))
cat(sprintf("       Macro over classes with test support -> Precision = %.4f | Recall = %.4f | F1 = %.4f\n",
            supported_macro_precision, supported_macro_recall, supported_macro_f1))
cat(sprintf("       Correct = %d / %d | Majority-class baseline ('always %s') = %.4f\n",
            correct_predictions, total_records, majority_class, majority_accuracy))

cat("      Confusion matrix (rows = actual, columns = predicted):\n")
cat(sprintf("        %-10s %8s %8s %8s %8s\n", "", CLASS_LEVELS[1], CLASS_LEVELS[2],
            CLASS_LEVELS[3], "Total"))
for (class_name in CLASS_LEVELS) {
  class_index <- match(class_name, CLASS_LEVELS)
  cat(sprintf("        %-10s %8d %8d %8d %8d\n",
              class_name,
              confusion[class_index, 1], confusion[class_index, 2], confusion[class_index, 3],
              sum(confusion[class_index, ])))
}

# ==============================================================================
# STEP 11 — Save the artifacts
# ==============================================================================
trained_at <- strftime(as.POSIXlt(Sys.time(), "UTC"), "%Y-%m-%dT%H:%M:%SZ")

# The bundle keeps the scaled training matrix, its labels, the exact scaling
# parameters, the selected K and the metadata the prediction script needs, so
# predict.R can answer a request by loading exactly one file and running
# class::knn() against it. It never refits anything.
model_bundle <- list(
  model = knn_model,
  model_type = "K-Nearest Neighbors",
  algorithm = "class::knn (Euclidean distance, majority vote)",
  target = TARGET,
  features = FEATURES,
  class_levels = as.character(CLASS_LEVELS),
  feature_ranges = feature_ranges,
  scaler = list(
    method = "standardisation (z-score) fitted on the training set",
    center = scaler_center,
    scale = scaler_scale
  ),
  k = SELECTED_K,
  k_grid = as.integer(K_GRID),
  r_version = sprintf("%s.%s", R.version$major, R.version$minor),
  class_package_version = as.character(utils::packageVersion("class")),
  trained_at = trained_at,
  random_seed = RANDOM_SEED,
  train_fraction = TRAIN_FRACTION,
  rows_total_loaded = rows_loaded,
  rows_dropped = rows_dropped,
  rows_train = nrow(train_data),
  rows_test = nrow(test_data)
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

confusion_matrix_payload <- lapply(CLASS_LEVELS, function(actual_class) {
  actual_index <- match(actual_class, CLASS_LEVELS)
  lapply(CLASS_LEVELS, function(predicted_class) {
    as.integer(confusion[actual_index, match(predicted_class, CLASS_LEVELS)])
  })
})

# The held-out test rows, with the model's real prediction for each one.
test_predictions_payload <- lapply(seq_len(nrow(test_data)), function(i) {
  list(
    actual_class = actual_classes[i],
    predicted_class = test_predicted_classes[i],
    correct = identical(actual_classes[i], test_predicted_classes[i]),
    neighbor_classes = as.list(as.character(train_y[test_neighbor_indices[, i]])),
    neighbor_distances = lapply(seq_len(SELECTED_K), function(j) {
      safe_number(test_distances[i, test_neighbor_indices[j, i]])
    })
  )
})

metrics_payload <- list(
  success = TRUE,
  model = "K-Nearest Neighbors",
  algorithm = model_bundle$algorithm,
  task = "Student Performance Prediction",
  target = TARGET,
  classes = as.character(CLASS_LEVELS),
  selected_k = as.integer(SELECTED_K),
  trained_at = trained_at,
  r_version = model_bundle$r_version,
  class_package_version = model_bundle$class_package_version,
  random_seed = RANDOM_SEED,
  metrics = list(
    accuracy = safe_number(accuracy),
    precision = safe_number(macro_precision),
    recall = safe_number(macro_recall),
    f1_score = safe_number(macro_f1),
    macro_supported_only = list(
      precision = safe_number(supported_macro_precision),
      recall = safe_number(supported_macro_recall),
      f1_score = safe_number(supported_macro_f1),
      classes = as.list(CLASS_LEVELS[as.integer(test_class_table[CLASS_LEVELS]) > 0L])
    ),
    balanced_accuracy = safe_number(balanced_accuracy),
    error_rate = safe_number(error_rate),
    correct_predictions = as.integer(correct_predictions),
    total_predictions = as.integer(total_records),
    majority_class_baseline = list(
      class = as.character(majority_class),
      accuracy = safe_number(majority_accuracy)
    ),
    averaging = "macro over all declared classes"
  ),
  class_distribution_note = list(
    full_dataset = as.list(stats::setNames(as.integer(class_counts[CLASS_LEVELS]), CLASS_LEVELS)),
    test_support = as.list(stats::setNames(as.integer(test_class_table[CLASS_LEVELS]), CLASS_LEVELS)),
    classes_without_test_support = as.list(classes_without_test_support),
    classes_unreachable_at_selected_k = as.list(unreachable_classes),
    class_reachability = class_reachability,
    message = sprintf(
      paste0("students.csv is heavily imbalanced and holds only %d LOW row, which must be used ",
             "for training, so the test set has no LOW rows. A KNN majority vote at K=%d also needs ",
             "%d of the K nearest rows to be LOW, which one LOW row can never satisfy, so LOW is ",
             "structurally unreachable and predictions are limited to %s. The headline macro averages ",
             "span all three declared classes (LOW scores 0); macro_supported_only spans just the ",
             "classes with test support."),
      as.integer(class_counts[["LOW"]]), SELECTED_K, ceiling(SELECTED_K / 2),
      paste(setdiff(CLASS_LEVELS, unreachable_classes), collapse = " and ")
    )
  ),
  per_class_metrics = per_class,
  confusion_matrix = list(
    labels = CLASS_LEVELS,
    rows = "actual",
    columns = "predicted",
    matrix = confusion_matrix_payload
  ),
  dataset = list(
    file = "datasets/students.csv",
    rows_loaded = rows_loaded,
    rows_clean = rows_clean,
    rows_dropped = rows_dropped,
    rows_incomplete = rows_incomplete,
    rows_domain_invalid = rows_invalid,
    rows_unknown_class = rows_unknown_class,
    rows_train = nrow(train_data),
    rows_test = nrow(test_data),
    target = TARGET,
    features = FEATURES,
    feature_ranges = feature_ranges,
    class_distribution = as.list(stats::setNames(as.integer(class_counts[CLASS_LEVELS]), CLASS_LEVELS))
  ),
  feature_scaling = list(
    method = model_bundle$scaler$method,
    reason = paste(
      "KNN measures Euclidean distance, so unscaled columns with larger spread would",
      "dominate the distance while smaller-spread columns would barely contribute.",
      "Every feature is standardised to mean 0 and standard deviation 1 using the",
      "training set, and predict.R applies the identical transformation."
    ),
    center = as.list(scaler_center),
    scale = as.list(scaler_scale)
  ),
  model_info = list(
    implementation = "class::knn()",
    distance = "Euclidean",
    vote = "majority vote over the K nearest training rows",
    k = as.integer(SELECTED_K),
    k_grid = as.integer(K_GRID),
    k_selection = list(
      method = "one-standard-error rule on repeated stratified cross-validation",
      folds = CV_FOLDS,
      repeats = CV_REPEATS,
      selected_on = "training data only (macro F1 over all declared classes)",
      best_mean_cv_f1 = safe_number(grid_scores[best_index]),
      one_se_threshold = safe_number(se_threshold)
    ),
    train_rows = as.integer(nrow(train_data)),
    test_rows = as.integer(nrow(test_data))
  ),
  k_comparison = cv_summary_payload,
  test_predictions = test_predictions_payload,
  model_file = "r_models/knn/model.rds"
)

writeLines(
  jsonlite::toJSON(metrics_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = metrics_path
)

cat(sprintf("[11/11] Saved model bundle  -> %s\n", model_path))
cat(sprintf("       Saved metrics        -> %s\n", metrics_path))
cat("======================================================================\n")
cat("PHASE 5 KNN TRAINING COMPLETE\n")
cat(sprintf("  SELECTED K = %d (chosen on training data only)\n", SELECTED_K))
cat(sprintf("  ACCURACY  = %.4f (%d / %d)\n", accuracy, correct_predictions, total_records))
cat(sprintf("  PRECISION = %.4f (macro, all 3 classes)\n", macro_precision))
cat(sprintf("  RECALL    = %.4f (macro, all 3 classes)\n", macro_recall))
cat(sprintf("  F1 SCORE  = %.4f (macro, all 3 classes)\n", macro_f1))
cat(sprintf("  F1 SCORE  = %.4f (macro, classes with test support)\n", supported_macro_f1))
if (length(unreachable_classes) > 0) {
  cat(sprintf("  NOTE: class(es) %s are unreachable at K=%d (too few training rows to win a majority vote).\n",
              paste(unreachable_classes, collapse = ", "), SELECTED_K))
}
cat("======================================================================\n")
