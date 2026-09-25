# ==============================================================================
# AI Insight Hub — Phase 3
# Linear Regression Training Script (Property Price Prediction)
#
# WHAT THIS SCRIPT DOES
#   1. Loads datasets/housing.csv
#   2. Validates that all required columns are present
#   3. Cleans invalid / missing data (non-numeric, NA, non-positive area or price)
#   4. Splits the cleaned data into a training set and a test set
#   5. Trains a real multiple linear regression model with stats::lm()
#   6. Predicts on the held-out test set
#   7. Calculates real evaluation metrics: R², RMSE, MAE
#   8. Saves the trained model bundle to r_models/regression/model.rds
#   9. Saves the metrics to r_models/regression/metrics.json
#  10. Saves the test-set actual vs predicted pairs to
#      r_models/regression/evaluation.json (used by the frontend chart)
#
# NOTHING IS HARD-CODED. Every number in metrics.json is computed from the
# model that this script actually fits.
#
# RUN IT WITH:
#   Rscript r_models/regression/train.R
#
# This is an educational ML implementation, not a production valuation tool.
# ==============================================================================

# --- Library configuration (kept in sync with the Python RRunner bridge) ------
lib_user <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_user) && dir.exists(lib_user)) {
  .libPaths(c(lib_user, .libPaths()))
}

suppressPackageStartupMessages({
  has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
})
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

dataset_path <- file.path(project_root, "datasets", "housing.csv")
model_dir <- script_dir

# Paths for the artifacts this script produces
model_path <- file.path(model_dir, "model.rds")
metrics_path <- file.path(model_dir, "metrics.json")
evaluation_path <- file.path(model_dir, "evaluation.json")

# --- Model configuration -----------------------------------------------------
TARGET <- "price"
FEATURES <- c("area", "bedrooms", "bathrooms", "location_score", "property_age")
REQUIRED_COLUMNS <- c(FEATURES, TARGET)
TRAIN_FRACTION <- 0.8
RANDOM_SEED <- 42   # Fixed seed so the split (and therefore the metrics) are reproducible

# ==============================================================================
# STEP 1 — Load the dataset
# ==============================================================================
if (!file.exists(dataset_path)) {
  cat(sprintf("FATAL: Dataset not found at %s\n", dataset_path))
  quit(status = 1)
}

raw_data <- read.csv(dataset_path, stringsAsFactors = FALSE, check.names = FALSE)
rows_loaded <- nrow(raw_data)
cat(sprintf("[1/8] Loaded dataset: %s (%d rows)\n", dataset_path, rows_loaded))

# ==============================================================================
# STEP 2 — Validate required columns
# ==============================================================================
missing_columns <- setdiff(REQUIRED_COLUMNS, names(raw_data))
if (length(missing_columns) > 0) {
  cat(sprintf("FATAL: Dataset is missing required column(s): %s\n",
              paste(missing_columns, collapse = ", ")))
  quit(status = 1)
}
cat(sprintf("[2/8] Column validation passed. Required columns present: %s\n",
            paste(REQUIRED_COLUMNS, collapse = ", ")))

# ==============================================================================
# STEP 3 — Clean invalid / missing data
#   read.csv() may parse a dirty numeric column as character, so every feature
#   is coerced to numeric and any row that cannot be made fully valid is
#   dropped (and counted, never silently ignored).
# ==============================================================================
data <- raw_data[, REQUIRED_COLUMNS]

for (col_name in REQUIRED_COLUMNS) {
  if (!is.numeric(data[[col_name]])) {
    data[[col_name]] <- suppressWarnings(as.numeric(as.character(data[[col_name]])))
  }
}

is_complete <- stats::complete.cases(data)
rows_incomplete <- sum(!is_complete)
data <- data[is_complete, ]

# A property must have a positive area and a positive target price to be usable.
is_positive <- data$area > 0 & data$price > 0
rows_non_positive <- sum(!is_positive)
data <- data[is_positive, ]

rows_dropped <- rows_incomplete + rows_non_positive
rows_clean <- nrow(data)

if (rows_clean <= length(FEATURES) + 1) {
  cat(sprintf("FATAL: Not enough usable rows after cleaning (%d rows remain).\n", rows_clean))
  quit(status = 1)
}

# Recompute the feature ranges from the cleaned data. These ranges are what the
# API and the R prediction script use to validate incoming property input, so
# they must come from real data and not from a hand-written constant.
feature_ranges <- lapply(FEATURES, function(col_name) {
  values <- data[[col_name]]
  list(min = min(values), max = max(values))
})
names(feature_ranges) <- FEATURES

cat(sprintf("[3/8] Data cleaning complete: %d rows kept, %d dropped (%d incomplete, %d non-positive)\n",
            rows_clean, rows_dropped, rows_incomplete, rows_non_positive))

# ==============================================================================
# STEP 4 — Train / test split
# ==============================================================================
set.seed(RANDOM_SEED)
row_indices <- sample(seq_len(rows_clean))
split_at <- floor(TRAIN_FRACTION * rows_clean)

train_data <- data[row_indices[1:split_at], ]
test_data <- data[row_indices[(split_at + 1):rows_clean], ]

cat(sprintf("[4/8] Split dataset: %d training rows / %d test rows (%d%% / %d%%)\n",
            nrow(train_data), nrow(test_data),
            round(TRAIN_FRACTION * 100), round((1 - TRAIN_FRACTION) * 100)))

# ==============================================================================
# STEP 5 — Train the real Linear Regression model
# ==============================================================================
# Formula built from the FEATURES vector so the model always matches the
# dataset schema that the API validates against.
model_formula <- as.formula(paste(TARGET, "~", paste(FEATURES, collapse = " + ")))

regression_model <- stats::lm(model_formula, data = train_data)

coefficients_table <- summary(regression_model)$coefficients
cat("[5/8] Trained Linear Regression model with stats::lm()\n")
cat("      Coefficients (term / estimate / std.error / t.value / p.value):\n")
for (coef_name in rownames(coefficients_table)) {
  row <- coefficients_table[coef_name, ]
  cat(sprintf("        %-16s %14.2f %14.2f %10.2f %12.6f\n",
              coef_name, row[["Estimate"]], row[["Std. Error"]],
              row[["t value"]], row[["Pr(>|t|)"]]))
}

# ==============================================================================
# STEP 6 — Predict on the held-out test set
# ==============================================================================
test_predictions <- stats::predict(regression_model, newdata = test_data)
actual_prices <- test_data$price

# ==============================================================================
# STEP 7 — Calculate real evaluation metrics from the trained model
# ==============================================================================
residuals_vec <- actual_prices - test_predictions

# Coefficient of determination: 1 - SSres / SStot
r2 <- 1 - sum(residuals_vec^2) / sum((actual_prices - mean(actual_prices))^2)

# Root mean squared error
rmse <- sqrt(mean(residuals_vec^2))

# Mean absolute error
mae <- mean(abs(residuals_vec))

# Training-set metrics are recorded too, so the train/test gap is visible.
train_predictions <- stats::predict(regression_model, newdata = train_data)
train_residuals <- train_data$price - train_predictions
train_r2 <- 1 - sum(train_residuals^2) / sum((train_data$price - mean(train_data$price))^2)
train_rmse <- sqrt(mean(train_residuals^2))
train_mae <- mean(abs(train_residuals))

# Adjusted R-squared straight from the model summary (uses the model's own
# degrees-of-freedom accounting rather than a re-derived formula).
adjusted_r2 <- summary(regression_model)$adj.r.squared

cat(sprintf("[6/8] Generated %d test-set predictions with predict()\n", length(test_predictions)))
cat(sprintf("[7/8] Test metrics  -> R2 = %.4f | RMSE = %.2f | MAE = %.2f\n", r2, rmse, mae))
cat(sprintf("      Train metrics -> R2 = %.4f | RMSE = %.2f | MAE = %.2f\n", train_r2, train_rmse, train_mae))

# Per-record actual vs predicted rows for the frontend scatter chart.
evaluation_rows <- lapply(seq_len(nrow(test_data)), function(i) {
  list(
    record = i,
    actual = round(as.numeric(actual_prices[i]), 2),
    predicted = round(as.numeric(test_predictions[i]), 2),
    residual = round(as.numeric(residuals_vec[i]), 2)
  )
})

# ==============================================================================
# STEP 8 — Save the artifacts
# ==============================================================================
# The bundle keeps the fitted model together with the metadata the prediction
# script needs (feature order, valid ranges, provenance) so that predict.R has
# to load exactly one file.
model_bundle <- list(
  model = regression_model,
  model_type = "Linear Regression",
  algorithm = "stats::lm (Ordinary Least Squares)",
  target = TARGET,
  features = FEATURES,
  feature_ranges = feature_ranges,
  formula = paste(deparse(model_formula), collapse = " "),
  r_version = sprintf("%s.%s", R.version$major, R.version$minor),
  trained_at = strftime(as.POSIXlt(Sys.time(), "UTC"), "%Y-%m-%dT%H:%M:%SZ"),
  random_seed = RANDOM_SEED,
  train_fraction = TRAIN_FRACTION,
  rows_total_loaded = rows_loaded,
  rows_dropped = rows_dropped,
  rows_train = nrow(train_data),
  rows_test = nrow(test_data)
)

saveRDS(model_bundle, file = model_path)

# JSON cannot represent NaN/Inf; guard against a degenerate split so the file
# always parses cleanly in Python and in the browser.
safe_number <- function(value) {
  if (is.null(value) || length(value) == 0 || is.na(value) || is.nan(value) || is.infinite(value)) {
    return(NULL)
  }
  round(as.numeric(value), 6)
}

metrics_payload <- list(
  success = TRUE,
  model = "Linear Regression",
  algorithm = model_bundle$algorithm,
  trained_at = model_bundle$trained_at,
  metrics = list(
    r2 = safe_number(r2),
    rmse = safe_number(rmse),
    mae = safe_number(mae),
    adjusted_r2 = safe_number(adjusted_r2),
    train_r2 = safe_number(train_r2),
    train_rmse = safe_number(train_rmse),
    train_mae = safe_number(train_mae)
  ),
  dataset = list(
    file = "datasets/housing.csv",
    rows_loaded = rows_loaded,
    rows_clean = rows_clean,
    rows_dropped = rows_dropped,
    rows_train = nrow(train_data),
    rows_test = nrow(test_data),
    target = TARGET,
    features = FEATURES,
    feature_ranges = feature_ranges
  ),
  coefficients = lapply(seq_len(nrow(coefficients_table)), function(i) {
    list(
      term = rownames(coefficients_table)[i],
      estimate = safe_number(coefficients_table[i, "Estimate"]),
      std_error = safe_number(coefficients_table[i, "Std. Error"]),
      t_value = safe_number(coefficients_table[i, "t value"]),
      p_value = safe_number(coefficients_table[i, "Pr(>|t|)"])
    )
  })
)

evaluation_payload <- list(
  success = TRUE,
  model = "Linear Regression",
  unit = "INR",
  trained_at = model_bundle$trained_at,
  count = length(evaluation_rows),
  points = evaluation_rows
)

writeLines(
  jsonlite::toJSON(metrics_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = metrics_path
)
writeLines(
  jsonlite::toJSON(evaluation_payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
  con = evaluation_path
)

cat(sprintf("[8/8] Saved model bundle  -> %s\n", model_path))
cat(sprintf("      Saved metrics        -> %s\n", metrics_path))
cat(sprintf("      Saved evaluation data-> %s (%d test points)\n", evaluation_path, length(evaluation_rows)))
cat("======================================================================\n")
cat("PHASE 3 LINEAR REGRESSION TRAINING COMPLETE\n")
cat(sprintf("  R2   = %.4f\n", r2))
cat(sprintf("  RMSE = %.2f INR\n", rmse))
cat(sprintf("  MAE  = %.2f INR\n", mae))
cat("======================================================================\n")
