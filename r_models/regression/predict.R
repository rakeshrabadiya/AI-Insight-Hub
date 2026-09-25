# ==============================================================================
# AI Insight Hub — Phase 3
# Linear Regression Prediction Script (Property Price Prediction)
#
# CONTRACT
#   Input  — a JSON object passed as the first command-line argument, e.g.
#              {"area":1500,"bedrooms":3,"bathrooms":2,
#               "location_score":8,"property_age":5}
#   Output — a single JSON object printed to stdout:
#              {"success":true,"prediction":...,"unit":"INR",
#               "model":"Linear Regression", ...}
#
# HOW IT WORKS
#   Loads the SAVED model bundle (model.rds) that train.R produced and calls
#   predict() on the already-fitted stats::lm object. It does NOT retrain
#   anything, so a prediction costs the same whether the model is one day old
#   or one year old.
#
# CALLED BY
#   backend/routes/regression.py via services.r_runner.execute_script()
#
# This is an educational ML implementation, not a production valuation tool.
# ==============================================================================

# --- Library configuration (kept in sync with the Python RRunner bridge) ------
lib_user <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_user) && dir.exists(lib_user)) {
  .libPaths(c(lib_user, .libPaths()))
}

has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)

# --- Resolve paths from the script location (works from any cwd) --------------
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else {
  getwd()
}

model_path <- file.path(script_dir, "model.rds")
metrics_path <- file.path(script_dir, "metrics.json")

# --- JSON output helper ------------------------------------------------------
# Works with or without jsonlite; every failure path goes through here so the
# Flask bridge always receives parseable JSON on stdout.
emit_json <- function(payload) {
  if (has_jsonlite) {
    cat(jsonlite::toJSON(payload, auto_unbox = TRUE, pretty = TRUE, null = "null"))
  } else {
    # Minimal base-R fallback for the happy path only.
    cat(sprintf('{"success": %s, "prediction": %s, "unit": "INR", "model": "Linear Regression"}',
                if (isTRUE(payload$success)) "true" else "false",
                if (is.null(payload$prediction)) "null" else format(payload$prediction, digits = 10)))
  }
  cat("\n")
  flush(stdout())
}

fail <- function(message, exit_code = 1) {
  emit_json(list(success = FALSE, error = message, model = "Linear Regression"))
  quit(status = exit_code)
}

# ==============================================================================
# STEP 1 — Parse the incoming JSON payload
# ==============================================================================
raw_args <- commandArgs(trailingOnly = TRUE)

if (length(raw_args) == 0) {
  fail("No input received. Expected a JSON object with the property features.")
}

raw_input <- paste(raw_args, collapse = " ")

if (!has_jsonlite) {
  fail("The 'jsonlite' package is required to read the prediction input. Install it with: install.packages('jsonlite')")
}

input_data <- tryCatch(
  jsonlite::fromJSON(raw_input, simplifyVector = FALSE),
  error = function(e) {
    fail(paste0("Input is not valid JSON: ", conditionMessage(e)))
  }
)

if (!is.list(input_data) || is.null(names(input_data))) {
  fail("Input must be a JSON object containing the property features.")
}

# ==============================================================================
# STEP 2 — Load the SAVED model bundle (no retraining happens here)
# ==============================================================================
if (!file.exists(model_path)) {
  fail(paste0("Trained model not found at: ", model_path,
              ". Run 'Rscript r_models/regression/train.R' to train the model first."))
}

model_bundle <- tryCatch(
  readRDS(model_path),
  error = function(e) {
    fail(paste0("Failed to load the trained model: ", conditionMessage(e)))
  }
)

if (is.null(model_bundle$model) || is.null(model_bundle$features)) {
  fail("The saved model bundle is malformed or was produced by an incompatible training script.")
}

regression_model <- model_bundle$model
feature_names <- model_bundle$features
feature_ranges <- model_bundle$feature_ranges

# ==============================================================================
# STEP 3 — Validate the incoming feature values
#   A value is rejected when it is missing, non-numeric, non-finite, or outside
#   the range the model was actually trained on. Range checks come from
#   model.rds (i.e. from the data), never from hard-coded limits.
# ==============================================================================
coerce_numeric <- function(value) {
  if (is.null(value) || length(value) == 0) return(NA_real_)
  if (is.list(value)) value <- unlist(value, use.names = FALSE)
  as.numeric(value)
}

validation_errors <- character(0)
property_input <- list()

for (feature_name in feature_names) {
  raw_value <- input_data[[feature_name]]

  if (is.null(raw_value)) {
    validation_errors <- c(validation_errors,
                           sprintf("Missing required field: '%s'.", feature_name))
    next
  }

  numeric_value <- coerce_numeric(raw_value)

  if (length(numeric_value) == 0 || is.na(numeric_value) || is.nan(numeric_value)) {
    validation_errors <- c(validation_errors,
                           sprintf("Field '%s' must be a number.", feature_name))
    next
  }
  if (is.infinite(numeric_value)) {
    validation_errors <- c(validation_errors,
                           sprintf("Field '%s' must be a finite number.", feature_name))
    next
  }

  allowed <- feature_ranges[[feature_name]]
  if (!is.null(allowed)) {
    if (numeric_value < allowed$min || numeric_value > allowed$max) {
      validation_errors <- c(validation_errors,
                             sprintf("Field '%s' must be between %s and %s (model training range).",
                                     feature_name,
                                     format(allowed$min, trim = TRUE),
                                     format(allowed$max, trim = TRUE)))
      next
    }
  }

  property_input[[feature_name]] <- numeric_value
}

if (length(validation_errors) > 0) {
  emit_json(list(
    success = FALSE,
    model = "Linear Regression",
    error = paste(validation_errors, collapse = " "),
    errors = as.list(validation_errors)
  ))
  quit(status = 0)
}

# ==============================================================================
# STEP 4 — Run predict() on the saved, already-fitted model
# ==============================================================================
new_data <- as.data.frame(property_input, stringsAsFactors = FALSE, check.names = FALSE)

prediction <- tryCatch(
  as.numeric(stats::predict(regression_model, newdata = new_data))[1],
  error = function(e) {
    fail(paste0("Prediction failed: ", conditionMessage(e)))
  }
)

if (length(prediction) == 0 || is.na(prediction) || is.nan(prediction) || is.infinite(prediction)) {
  fail("The model produced a non-finite prediction for the supplied input.")
}

# A property price is never negative; a small negative value would only come
# from extrapolating far outside the training region, which the range checks
# above already reject.
if (prediction < 0) {
  fail(sprintf("The model produced a negative price (%.2f) for the supplied input, which is not a valid property valuation.", prediction))
}

# Round to 2 decimals (paise) for a clean monetary value in the UI.
prediction <- round(prediction, 2)

# Attach the model's real performance metrics so the client always displays
# numbers that came from training, never from a literal in the response.
model_metrics <- NULL
if (file.exists(metrics_path)) {
  model_metrics <- tryCatch({
    parsed_metrics <- jsonlite::fromJSON(metrics_path, simplifyVector = TRUE)
    if (!is.null(parsed_metrics$metrics)) parsed_metrics$metrics else NULL
  }, error = function(e) NULL)
}

response <- list(
  success = TRUE,
  prediction = prediction,
  unit = "INR",
  model = "Linear Regression",
  algorithm = if (!is.null(model_bundle$algorithm)) model_bundle$algorithm else "stats::lm (Ordinary Least Squares)",
  target = if (!is.null(model_bundle$target)) model_bundle$target else "price",
  inputs = as.list(property_input),
  model_r2 = if (!is.null(model_metrics$r2)) round(as.numeric(model_metrics$r2), 4) else NULL,
  model_rmse = if (!is.null(model_metrics$rmse)) round(as.numeric(model_metrics$rmse), 2) else NULL,
  model_mae = if (!is.null(model_metrics$mae)) round(as.numeric(model_metrics$mae), 2) else NULL,
  trained_at = if (!is.null(model_bundle$trained_at)) model_bundle$trained_at else NULL,
  r_version = if (!is.null(model_bundle$r_version)) model_bundle$r_version else NULL
)

emit_json(response)
