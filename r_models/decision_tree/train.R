# ==============================================================================
# AI Insight Hub — Phase 4
# Decision Tree Training Script (Financial Risk Classification)
#
# WHAT THIS SCRIPT DOES
#   1. Loads datasets/risk.csv
#   2. Validates that all required columns are present
#   3. Cleans invalid / missing data (non-numeric, NA, out-of-domain values)
#   4. Splits the cleaned data into a STRATIFIED training set and test set so
#      that every risk class is represented in both halves (the dataset is
#      heavily imbalanced, so a naive random split could lose the rare class)
#   5. Trains a real CART Decision Tree classifier with rpart::rpart()
#   6. Predicts on the held-out test set
#   7. Calculates real evaluation metrics: accuracy, confusion matrix,
#      precision, recall and F1 score
#   8. Extracts the REAL fitted tree structure out of the rpart object
#   9. Saves the trained model bundle to r_models/decision_tree/model.rds
#  10. Saves the metrics to r_models/decision_tree/metrics.json
#  11. Saves the tree structure to r_models/decision_tree/tree.json
#
# NOTHING IS HARD-CODED. Every number in metrics.json and every node in
# tree.json is computed from the model that this script actually fits.
#
# RUN IT WITH:
#   Rscript r_models/decision_tree/train.R
#
# This is an educational ML implementation, not a production credit-scoring
# system, and it does not constitute financial advice.
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

has_rpart <- requireNamespace("rpart", quietly = TRUE)
if (!has_rpart) {
  cat("FATAL: The 'rpart' package is required to train the Decision Tree.\n")
  cat("Install it with: install.packages('rpart')\n")
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

dataset_path <- file.path(project_root, "datasets", "risk.csv")
model_dir <- script_dir

# Paths for the artifacts this script produces
model_path <- file.path(model_dir, "model.rds")
metrics_path <- file.path(model_dir, "metrics.json")
tree_path <- file.path(model_dir, "tree.json")

# --- Model configuration -----------------------------------------------------
TARGET <- "risk"
FEATURES <- c("age", "income", "credit_score", "existing_loans", "employment_years")
REQUIRED_COLUMNS <- c(FEATURES, TARGET)
# The class order is fixed so that the class-probability columns of the rpart
# object (yval2.V1 .. yval2.V3) always mean the same thing.
CLASS_LEVELS <- c("LOW", "MEDIUM", "HIGH")
TRAIN_FRACTION <- 0.8
RANDOM_SEED <- 42   # Fixed seed so the split (and therefore the metrics) are reproducible

# Tree growth controls. The final (maxdepth, minsplit) pair is not chosen by
# hand: it is selected below by repeated stratified cross-validation on the
# TRAINING data only, so the held-out test set stays an honest estimate.
MAX_DEPTH_GRID <- c(3, 4, 5, 6, 7)
MIN_SPLIT_GRID <- c(2, 3, 5, 8, 10, 15)
CV_FOLDS <- 5
CV_REPEATS <- 10
CPARAM <- 0.0001

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

# Domain rules that a genuine applicant record must satisfy. These are the
# documented schema of the dataset, not values copied from any model output.
is_valid_row <- data$age > 0 &
               data$income > 0 &
               data$credit_score >= 300 & data$credit_score <= 900 &
               data$existing_loans >= 0 &
               data$employment_years >= 0
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

# Convert the target to a factor with the fixed class order. rpart learns the
# class index (1/2/3) from this factor, so the order must be deterministic.
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
# API and the R prediction script use to validate incoming applicant input, so
# they must come from real data and not from a hand-written constant.
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
#   risk.csv is strongly imbalanced (the HIGH class dominates), so a plain
#   random split can leave a class out of the test set entirely. Splitting
#   inside each class guarantees every class appears in both halves.
# ==============================================================================
set.seed(RANDOM_SEED)

train_index <- integer(0)
test_index <- integer(0)

for (class_name in CLASS_LEVELS) {
  class_rows <- which(data[[TARGET]] == class_name)
  shuffled <- sample(class_rows)

  # Give every class at least one training row and one test row.
  n_test <- max(1L, min(length(shuffled) - 1L, floor((1 - TRAIN_FRACTION) * length(shuffled))))
  test_index <- c(test_index, shuffled[seq_len(n_test)])
  train_index <- c(train_index, shuffled[(n_test + 1L):length(shuffled)])
}

train_data <- data[train_index, ]
test_data <- data[test_index, ]

# Every class must be represented in the test set, otherwise precision / recall
# for that class cannot be computed at all.
test_classes <- unique(as.character(test_data[[TARGET]]))
absent_from_test <- setdiff(CLASS_LEVELS, test_classes)
if (length(absent_from_test) > 0) {
  cat(sprintf("FATAL: Stratified split failed — class(es) %s are missing from the test set.\n",
              paste(absent_from_test, collapse = ", ")))
  quit(status = 1)
}

cat(sprintf("[4/11] Stratified split: %d training rows / %d test rows (%d%% / %d%%)\n",
            nrow(train_data), nrow(test_data),
            round(TRAIN_FRACTION * 100), round((1 - TRAIN_FRACTION) * 100)))

# ==============================================================================
# STEP 5 — Choose tree depth by repeated stratified cross-validation
#   (TRAINING DATA ONLY)
#   The dataset is strongly imbalanced and the LOW class has very few rows, so
#   a single 5-fold split is a noisy way to pick hyper-parameters — the
#   resulting F1 estimate has a standard error of roughly 0.14. Repeating the
#   stratified split 10 times drops that error to about 0.02, which is what
#   makes the comparison between grid points meaningful.
#
#   Selection uses the 1-SE rule: among all settings whose mean CV F1 is within
#   one standard error of the best, the SIMPLEST tree (lowest depth, then
#   highest minsplit) is taken. That favours a model that generalises over one
#   that chases the last fraction of a percent on a 97-row training set.
# ==============================================================================
make_cv_folds <- function(frame_data, k, seed) {
  set.seed(seed)
  fold_id <- integer(nrow(frame_data))
  for (class_name in CLASS_LEVELS) {
    class_rows <- which(frame_data[[TARGET]] == class_name)
    if (length(class_rows) == 0) next
    shuffled <- sample(class_rows)
    fold_id[shuffled] <- rep(seq_len(k), length.out = length(shuffled))
  }
  fold_id
}

confusion_from_predictions <- function(actual, predicted) {
  table(
    factor(actual, levels = CLASS_LEVELS),
    factor(predicted, levels = CLASS_LEVELS)
  )
}

macro_f1_from_confusion <- function(cm) {
  class_index <- seq_along(CLASS_LEVELS)
  precision_vec <- vapply(class_index, function(i) {
    column_total <- sum(cm[, i])
    if (column_total > 0) cm[i, i] / column_total else 0
  }, numeric(1))
  recall_vec <- vapply(class_index, function(i) {
    row_total <- sum(cm[i, ])
    if (row_total > 0) cm[i, i] / row_total else 0
  }, numeric(1))
  f1_vec <- (2 * precision_vec * recall_vec) / (precision_vec + recall_vec)
  f1_vec[is.na(f1_vec)] <- 0
  mean(f1_vec)
}

grid_results <- list()
for (depth in MAX_DEPTH_GRID) {
  for (min_split in MIN_SPLIT_GRID) {
    fold_scores <- numeric(0)

    for (repeat_id in seq_len(CV_REPEATS)) {
      # A different seed per repeat re-shuffles the folds, so the score
      # averages over many possible splits rather than one lucky one.
      cv_folds <- make_cv_folds(train_data, CV_FOLDS, seed = RANDOM_SEED + repeat_id)

      for (fold in seq_len(CV_FOLDS)) {
        fold_train <- train_data[cv_folds != fold, , drop = FALSE]
        fold_valid <- train_data[cv_folds == fold, , drop = FALSE]
        if (nrow(fold_valid) == 0 || length(unique(fold_valid[[TARGET]])) < 2) next

        fold_model <- rpart::rpart(
          as.formula(paste(TARGET, "~", paste(FEATURES, collapse = " + "))),
          data = fold_train,
          method = "class",
          control = rpart::rpart.control(maxdepth = depth, minsplit = min_split, cp = CPARAM)
        )
        fold_predictions <- predict(fold_model, newdata = fold_valid, type = "class")
        fold_scores <- c(
          fold_scores,
          macro_f1_from_confusion(
            confusion_from_predictions(as.character(fold_valid[[TARGET]]),
                                       as.character(fold_predictions))
          )
        )
      }
    }

    if (length(fold_scores) == 0) next
    grid_results[[length(grid_results) + 1L]] <- list(
      max_depth = depth,
      min_split = min_split,
      mean_f1 = mean(fold_scores),
      sd_f1 = stats::sd(fold_scores),
      se_f1 = stats::sd(fold_scores) / sqrt(length(fold_scores)),
      folds_evaluated = length(fold_scores)
    )
  }
}

grid_scores <- vapply(grid_results, function(entry) entry$mean_f1, numeric(1))
grid_depths <- vapply(grid_results, function(entry) entry$max_depth, numeric(1))
grid_minsplits <- vapply(grid_results, function(entry) entry$min_split, numeric(1))
grid_se <- vapply(grid_results, function(entry) entry$se_f1, numeric(1))

# Best mean F1 defines the one-standard-error threshold.
best_index <- which.max(grid_scores)
se_threshold <- grid_scores[best_index] - grid_se[best_index]

# Among the settings statistically indistinguishable from the best, take the
# simplest tree: lowest depth first, then the highest minsplit.
within_one_se <- which(grid_scores >= se_threshold)
simplicity <- order(grid_depths[within_one_se], -grid_minsplits[within_one_se])
selected_index <- within_one_se[simplicity[1]]

MAX_DEPTH <- grid_results[[selected_index]]$max_depth
MIN_SPLIT <- grid_results[[selected_index]]$min_split

# Report the whole grid, best-scoring first.
ordering <- order(-grid_scores, grid_depths, -grid_minsplits)
cv_summary_payload <- lapply(grid_results[ordering], function(entry) {
  list(
    max_depth = as.integer(entry$max_depth),
    min_split = as.integer(entry$min_split),
    cv_f1_mean = round(entry$mean_f1, 6),
    cv_f1_sd = round(entry$sd_f1, 6),
    cv_f1_se = round(entry$se_f1, 6)
  )
})

cat(sprintf("[5/11] Cross-validated %d (maxdepth, minsplit) combinations over %d repeats of %d stratified folds\n",
            length(grid_results), CV_REPEATS, CV_FOLDS))
cat(sprintf("      Best mean CV F1 = %.4f (maxdepth %d, minsplit %d), 1-SE threshold = %.4f\n",
            grid_scores[best_index],
            grid_results[[best_index]]$max_depth,
            grid_results[[best_index]]$min_split,
            se_threshold))
cat(sprintf("      Selected by 1-SE rule: maxdepth = %d, minsplit = %d (CV F1 = %.4f +/- %.4f)\n",
            MAX_DEPTH, MIN_SPLIT, grid_scores[selected_index], grid_se[selected_index]))

# ==============================================================================
# STEP 6 — Train the real Decision Tree model with the selected settings
# ==============================================================================
# Formula built from the FEATURES vector so the model always matches the
# dataset schema that the API validates against.
model_formula <- as.formula(paste(TARGET, "~", paste(FEATURES, collapse = " + ")))

# method = "class" makes rpart grow a CART classification tree; the default
# Gini impurity is used as the split criterion.
tree_model <- rpart::rpart(
  model_formula,
  data = train_data,
  method = "class",
  control = rpart::rpart.control(maxdepth = MAX_DEPTH, minsplit = MIN_SPLIT, cp = CPARAM)
)

class_levels <- attr(tree_model, "ylevels")
n_classes <- length(class_levels)

cat("[6/11] Trained Decision Tree with rpart::rpart()\n")
cat(sprintf("      Method: %s | Nodes: %d | Leaves: %d | Variables used: %d\n",
            tree_model$method, nrow(tree_model$frame),
            sum(tree_model$frame$var == "<leaf>"),
            length(tree_model$variable.importance)))

importance_table <- sort(tree_model$variable.importance, decreasing = TRUE)
if (length(importance_table) > 0) {
  cat("      Variable importance (Gini gain):\n")
  for (feature_name in names(importance_table)) {
    cat(sprintf("        %-18s %12.4f\n", feature_name, importance_table[[feature_name]]))
  }
}

# ==============================================================================
# STEP 7 — Predict on the held-out test set
# ==============================================================================
test_predictions <- predict(tree_model, newdata = test_data, type = "class")
actual_classes <- as.character(test_data[[TARGET]])

# ==============================================================================
# STEP 8 — Calculate real evaluation metrics from the predictions
#   Every metric below is derived from the confusion matrix built out of the
#   test-set predictions; none of them is a literal.
# ==============================================================================
# table(actual, predicted) with a fixed level order keeps the matrix square so
# that a class the model never predicted still contributes a zero row/column.
confusion <- table(
  factor(actual_classes, levels = CLASS_LEVELS),
  factor(as.character(test_predictions), levels = CLASS_LEVELS)
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
}

# Risk classification is a multi-class problem with no single "positive" class,
# so the headline precision / recall / F1 are the macro averages across the
# three risk tiers. The per-class numbers are kept in the payload as well.
macro_precision <- mean(precision_values)
macro_recall <- mean(recall_values)
macro_f1 <- mean(f1_values)

# Balanced accuracy = mean of the per-class recalls; it is reported because a
# headline accuracy alone is misleading on a heavily imbalanced dataset.
balanced_accuracy <- mean(recall_values)
error_rate <- 1 - accuracy

# A model that always predicted the most common class would also look accurate
# on an imbalanced test set, so that baseline is recorded alongside the real
# score to make the accuracy interpretable rather than misleading.
test_class_table <- table(actual_classes)
majority_class <- names(sort(test_class_table, decreasing = TRUE))[1]
majority_accuracy <- as.numeric(test_class_table[[majority_class]]) / total_records

cat(sprintf("[7/11] Generated %d test-set predictions with predict()\n", length(test_predictions)))
cat(sprintf("[8/11] Test metrics -> Accuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f\n",
            accuracy, macro_precision, macro_recall, macro_f1))
cat(sprintf("      Balanced accuracy = %.4f | Correct = %d / %d\n",
            balanced_accuracy, correct_predictions, total_records))
cat(sprintf("      Majority-class baseline ('always %s') = %.4f\n", majority_class, majority_accuracy))

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
# STEP 9 — Extract the REAL tree structure out of the fitted rpart object
#
# rpart stores its splits in `splits`, a matrix whose ROWS are the candidate
# splits considered at each node. Only one row per node is the split the node
# actually used, and rpart's own labels.rpart() identifies it with:
#
#     irow <- cumsum(c(1, ncompete + nsurrogate + !is.leaf))[c(whichrow, FALSE)]
#
# Two details matter and are easy to get wrong:
#   * `frame` rows are indexed by NODE ID (the row name), not by position.
#   * A node's children are always 2*n and 2*n+1.
# `ncat` (column 2 of `splits`) says which way a node splits:
#     ncat <  0  ->  ordered:  left branch is "< cut",  right branch is ">= cut"
#     ncat == 1  ->  left branch is ">= cut",         right branch is "< cut"
#     ncat == 2  ->  left branch is "< cut",          right branch is ">= cut"
#
# These are exactly the rules rpart's own labels()/print() use, so the tree
# written to tree.json is the model that was actually fitted.
# ==============================================================================
frame <- tree_model$frame
node_ids <- as.numeric(rownames(frame))
is_leaf <- (frame$var == "<leaf>")
internal_mask <- !is_leaf

split_row_index <- cumsum(c(1, frame$ncompete + frame$nsurrogate + !is_leaf))[
  c(internal_mask, FALSE)
]

node_splits <- list()
if (any(internal_mask)) {
  internal_ids <- node_ids[internal_mask]
  internal_vars <- as.character(frame$var[internal_mask])
  split_ncat <- tree_model$splits[split_row_index, 2]
  split_cut <- tree_model$splits[split_row_index, 4]

  for (k in seq_along(internal_ids)) {
    ncat_value <- as.numeric(split_ncat[k])
    node_splits[[as.character(internal_ids[k])]] <- list(
      variable = internal_vars[k],
      cut = as.numeric(split_cut[k]),
      # TRUE means "values below the cut travel down the left branch".
      left_is_below = if (ncat_value < 0) TRUE else (ncat_value == 2)
    )
  }
}

# Formats a split threshold for display without trailing zeros.
format_cut <- function(value) {
  formatted <- formatC(as.numeric(value), format = "f", digits = 2)
  sub("\\.?0+$", "", formatted)
}

# Builds the condition text rpart itself would print for a branch.
branch_label <- function(split_info, go_left) {
  operator <- if (go_left == split_info$left_is_below) "<" else ">="
  sprintf("%s %s %s", split_info$variable, operator, format_cut(split_info$cut))
}

# `frame` keeps the class counts and probabilities in a NESTED MATRIX COLUMN
# called `yval2`, with one row per node in the same order as `frame` itself.
# Printing the frame gives those columns the cosmetic names yval2.V1, yval2.V2,
# ... but they are NOT real column names, so they are read POSITIONALLY. The
# actual layout is:
#
#     column 1        predicted class index (an integer code for ylevels)
#     columns 2..n+1  class counts      (in ylevels order; they sum to frame$n)
#     columns n+2..2n+1  class probabilities (each row sums to 1)
#     column 2n+2     node probability (the one named "nodeprob")
#
# The mapping is verified numerically before it is trusted: the counts must sum
# to the node's sample count and the probabilities must sum to 1.
yval2_matrix <- frame$yval2
frame_row_index <- as.integer(rownames(frame))

class_count_columns <- 2L:(n_classes + 1L)
class_probability_columns <- (n_classes + 2L):(2L * n_classes + 1L)
node_probability_column <- 2L * n_classes + 2L

if (node_probability_column > ncol(yval2_matrix)) {
  cat("FATAL: The fitted tree's yval2 matrix does not have the expected layout.\n")
  quit(status = 1)
}

layout_ok <- all(
  abs(rowSums(yval2_matrix[, class_count_columns, drop = FALSE]) - frame$n) < 1e-6
) && all(
  abs(rowSums(yval2_matrix[, class_probability_columns, drop = FALSE]) - 1) < 1e-6
)
if (!isTRUE(layout_ok)) {
  cat("FATAL: Could not locate the class count / probability columns in the fitted tree.\n")
  quit(status = 1)
}

build_tree_node <- function(node_id) {
  key <- as.character(node_id)
  row_index <- match(node_id, frame_row_index)
  if (is.na(row_index)) {
    cat(sprintf("FATAL: Node %s is not present in the fitted tree.\n", key))
    quit(status = 1)
  }

  node_class_index <- as.integer(frame[key, "yval"])

  node <- list(
    id = as.integer(node_id),
    is_leaf = identical(as.character(frame[key, "var"]), "<leaf>"),
    samples = as.integer(frame[key, "n"]),
    weighted_samples = round(as.numeric(frame[key, "wt"]), 6),
    deviance = round(as.numeric(frame[key, "dev"]), 6),
    node_probability = round(as.numeric(yval2_matrix[row_index, node_probability_column]), 6),
    predicted_class = as.character(class_levels[node_class_index]),
    left_is_below = NULL,
    class_counts = list(),
    class_probabilities = list()
  )

  for (k in seq_len(n_classes)) {
    class_name <- class_levels[k]
    node$class_counts[[class_name]] <- as.integer(yval2_matrix[row_index, class_count_columns[k]])
    node$class_probabilities[[class_name]] <- round(
      as.numeric(yval2_matrix[row_index, class_probability_columns[k]]), 6
    )
  }

  if (isTRUE(node$is_leaf)) {
    node$variable <- NULL
    node$cut <- NULL
    node$left_condition <- NULL
    node$right_condition <- NULL
    node$left <- NULL
    node$right <- NULL
    return(node)
  }

  split_info <- node_splits[[key]]
  if (is.null(split_info)) {
    cat(sprintf("FATAL: No split information found for node %s.\n", key))
    quit(status = 1)
  }

  node$variable <- split_info$variable
  node$cut <- split_info$cut
  node$left_is_below <- isTRUE(split_info$left_is_below)
  node$left_condition <- branch_label(split_info, go_left = TRUE)
  node$right_condition <- branch_label(split_info, go_left = FALSE)
  node$left <- build_tree_node(2L * as.integer(node_id))
  node$right <- build_tree_node(2L * as.integer(node_id) + 1L)

  node
}

tree_root <- build_tree_node(1L)
tree_nodes_total <- nrow(frame)
tree_leaves_total <- sum(is_leaf)
tree_depth_total <- max(nchar(rownames(frame))) - 1L

cat(sprintf("[9/11] Extracted fitted tree: %d nodes (%d internal, %d leaves)\n",
            tree_nodes_total, tree_nodes_total - tree_leaves_total, tree_leaves_total))

# ==============================================================================
# STEP 10 — Self-check: the extracted tree must route a record exactly the way
#   rpart's own predict() does. If the hand-built routing disagreed with the
#   model, tree.json would be a decorative drawing rather than the real model,
#   so this is verified before anything is written to disk.
# ==============================================================================
walk_tree <- function(feature_row) {
  current <- tree_root
  steps <- 0L
  while (!isTRUE(current$is_leaf)) {
    value <- as.numeric(feature_row[[current$variable]])
    if (is.na(value)) {
      return(NULL)
    }
    goes_left <- if (isTRUE(current$left_is_below)) value < current$cut else value >= current$cut
    current <- if (goes_left) current$left else current$right
    steps <- steps + 1L
    if (steps > 64L) break
  }
  current
}

routing_matrix <- as.matrix(test_data[, FEATURES])
routed_classes <- vapply(seq_len(nrow(routing_matrix)), function(i) {
  walked <- walk_tree(as.list(routing_matrix[i, ]))
  if (is.null(walked)) NA_character_ else walked$predicted_class
}, character(1))

routing_agreement <- mean(routed_classes == as.character(test_predictions))
if (!isTRUE(all.equal(as.numeric(routing_agreement), 1.0))) {
  cat(sprintf("FATAL: Extracted tree routes only %.2f%% of test records the same way predict() does.\n",
              routing_agreement * 100))
  cat("       Refusing to write tree.json — it would not represent the fitted model.\n")
  quit(status = 1)
}
cat(sprintf("[10/11] Tree routing self-check passed: extracted tree matches predict() on %d/%d test records\n",
            round(routing_agreement * nrow(test_data)), nrow(test_data)))

# The leaf probabilities recorded in the tree must also be the real ones, so a
# second check compares the extracted leaf distribution against rpart's own
# predict(type = "prob") for the same records.
leaf_prob_matrix <- as.matrix(test_data[, FEATURES])
# vcollect() stacks one row per record, so the result is transposed to match the
# record-per-row layout of predict()'s own probability matrix.
routed_probabilities <- t(vapply(seq_len(nrow(leaf_prob_matrix)), function(i) {
  walked <- walk_tree(as.list(leaf_prob_matrix[i, ]))
  if (is.null(walked)) return(rep(NA_real_, n_classes))
  as.numeric(unlist(walked$class_probabilities[as.character(class_levels)]))
}, numeric(n_classes)))

model_probabilities <- as.matrix(predict(tree_model, newdata = test_data, type = "prob"))
colnames(model_probabilities) <- as.character(class_levels)
probability_agreement <- mean(
  apply(abs(routed_probabilities - model_probabilities), 1, max) < 1e-6
)
if (!isTRUE(all.equal(as.numeric(probability_agreement), 1.0))) {
  cat(sprintf("FATAL: Extracted leaf probabilities differ from predict(type='prob') on %.2f%% of test records.\n",
              (1 - probability_agreement) * 100))
  cat("       Refusing to write tree.json — it would not represent the fitted model.\n")
  quit(status = 1)
}
cat(sprintf("       Leaf probability self-check passed on %d/%d test records\n",
            nrow(test_data), nrow(test_data)))

# ==============================================================================
# STEP 11 — Save the artifacts
# ==============================================================================
trained_at <- strftime(as.POSIXlt(Sys.time(), "UTC"), "%Y-%m-%dT%H:%M:%SZ")

# The bundle keeps the fitted model together with the metadata the prediction
# script needs (feature order, valid ranges, class order) and the extracted
# tree, so that predict.R can report a genuine per-request decision path by
# loading exactly one file.
model_bundle <- list(
  model = tree_model,
  model_type = "Decision Tree",
  algorithm = "rpart::rpart (CART, Gini impurity)",
  target = TARGET,
  features = FEATURES,
  class_levels = as.character(class_levels),
  feature_ranges = feature_ranges,
  formula = paste(deparse(model_formula), collapse = " "),
  tree = tree_root,
  r_version = sprintf("%s.%s", R.version$major, R.version$minor),
  rpart_version = as.character(utils::packageVersion("rpart")),
  trained_at = trained_at,
  random_seed = RANDOM_SEED,
  train_fraction = TRAIN_FRACTION,
  max_depth = MAX_DEPTH,
  min_split = MIN_SPLIT,
  rows_total_loaded = rows_loaded,
  rows_dropped = rows_dropped,
  rows_train = nrow(train_data),
  rows_test = nrow(test_data)
)

saveRDS(model_bundle, file = model_path)

# JSON cannot represent NaN/Inf; guard against a degenerate split so the files
# always parse cleanly in Python and in the browser.
safe_number <- function(value) {
  if (is.null(value) || length(value) == 0 || is.na(value) || is.nan(value) || is.infinite(value)) {
    return(NULL)
  }
  round(as.numeric(value), 6)
}

importance_payload <- lapply(names(importance_table), function(feature_name) {
  list(feature = feature_name, importance = safe_number(importance_table[[feature_name]]))
})

confusion_matrix_payload <- lapply(CLASS_LEVELS, function(actual_class) {
  actual_index <- match(actual_class, CLASS_LEVELS)
  lapply(CLASS_LEVELS, function(predicted_class) {
    as.integer(confusion[actual_index, match(predicted_class, CLASS_LEVELS)])
  })
})

metrics_payload <- list(
  success = TRUE,
  model = "Decision Tree",
  algorithm = model_bundle$algorithm,
  task = "Financial Risk Classification",
  target = TARGET,
  classes = as.character(class_levels),
  trained_at = trained_at,
  r_version = model_bundle$r_version,
  rpart_version = model_bundle$rpart_version,
  random_seed = RANDOM_SEED,
  metrics = list(
    accuracy = safe_number(accuracy),
    precision = safe_number(macro_precision),
    recall = safe_number(macro_recall),
    f1_score = safe_number(macro_f1),
    balanced_accuracy = safe_number(balanced_accuracy),
    error_rate = safe_number(error_rate),
    correct_predictions = as.integer(correct_predictions),
    total_predictions = as.integer(total_records),
    majority_class_baseline = list(
      class = as.character(majority_class),
      accuracy = safe_number(majority_accuracy)
    ),
    averaging = "macro"
  ),
  per_class_metrics = per_class,
  confusion_matrix = list(
    labels = CLASS_LEVELS,
    rows = "actual",
    columns = "predicted",
    matrix = confusion_matrix_payload
  ),
  dataset = list(
    file = "datasets/risk.csv",
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
    class_distribution = as.list(stats::setNames(
      as.integer(class_counts[CLASS_LEVELS]), CLASS_LEVELS
    ))
  ),
  model_info = list(
    control = list(
      maxdepth = as.integer(MAX_DEPTH),
      minsplit = as.integer(MIN_SPLIT),
      cp = CPARAM,
      cv_folds = CV_FOLDS,
      cv_selected_on = "training data only (macro F1)"
    ),
    total_nodes = as.integer(tree_nodes_total),
    internal_nodes = as.integer(tree_nodes_total - tree_leaves_total),
    leaf_nodes = as.integer(tree_leaves_total),
    variable_importance = importance_payload
  ),
  cross_validation = cv_summary_payload,
  model_file = "r_models/decision_tree/model.rds"
)

tree_payload <- list(
  success = TRUE,
  model = "Decision Tree",
  algorithm = model_bundle$algorithm,
  target = TARGET,
  classes = as.character(class_levels),
  features = FEATURES,
  feature_labels = list(
    age = "Age",
    income = "Annual Income (INR)",
    credit_score = "Credit Score",
    existing_loans = "Existing Loans",
    employment_years = "Employment Years"
  ),
  trained_at = trained_at,
  total_nodes = as.integer(tree_nodes_total),
  internal_nodes = as.integer(tree_nodes_total - tree_leaves_total),
  leaf_nodes = as.integer(tree_leaves_total),
  control = list(
    maxdepth = as.integer(MAX_DEPTH),
    minsplit = as.integer(MIN_SPLIT),
    cp = CPARAM
  ),
  variable_importance = importance_payload,
  root = tree_root
)

writeLines(
  jsonlite::toJSON(metrics_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = metrics_path
)
writeLines(
  jsonlite::toJSON(tree_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = tree_path
)

cat(sprintf("[11/11] Saved model bundle  -> %s\n", model_path))
cat(sprintf("       Saved metrics        -> %s\n", metrics_path))
cat(sprintf("       Saved tree structure -> %s (%d nodes)\n", tree_path, tree_nodes_total))
cat("======================================================================\n")
cat("PHASE 4 DECISION TREE TRAINING COMPLETE\n")
cat(sprintf("  ACCURACY  = %.4f (%d / %d)\n", accuracy, correct_predictions, total_records))
cat(sprintf("  PRECISION = %.4f (macro)\n", macro_precision))
cat(sprintf("  RECALL    = %.4f (macro)\n", macro_recall))
cat(sprintf("  F1 SCORE  = %.4f (macro)\n", macro_f1))
cat("======================================================================\n")
