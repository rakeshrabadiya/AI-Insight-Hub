# ==============================================================================
# AI Insight Hub — Phase 4
# Decision Tree Prediction Script (Financial Risk Classification)
#
# CONTRACT
#   Input  — a JSON object passed as the first command-line argument, e.g.
#              {"age":32,"income":750000,"credit_score":735,
#               "existing_loans":1,"employment_years":5}
#   Output — a single JSON object printed to stdout:
#              {"success":true,"model":"Decision Tree","prediction":"MEDIUM",
#               "confidence":0.8,"decision_path":[...],"class_probabilities":{...}}
#
# HOW IT WORKS
#   Loads the SAVED model bundle (model.rds) that train.R produced and calls
#   predict() on the already-fitted rpart object. It does NOT retrain anything,
#   so a prediction costs the same whether the model is one day old or one year
#   old.
#
#   The decision path comes from walking the SAME extracted tree that train.R
#   wrote to tree.json — the tree train.R verified routes records identically
#   to rpart's own predict(). Each step reports the rule that was actually
#   taken, the threshold in it, and the applicant's real value, so the path is
#   a genuine trace of this model rather than a hand-written summary.
#
#   The confidence is the predicted class's probability taken from rpart's own
#   predict(type = "prob") output. It is never invented.
#
# CALLED BY
#   backend/routes/decision_tree.py via services.r_runner.execute_script()
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

MODEL_LABEL <- "Decision Tree"

# --- JSON output helper ------------------------------------------------------
# Every failure path goes through here so the Flask bridge always receives
# parseable JSON on stdout, even when jsonlite is unavailable.
emit_json <- function(payload) {
  if (has_jsonlite) {
    cat(jsonlite::toJSON(payload, auto_unbox = TRUE, pretty = TRUE, null = "null"))
  } else {
    cat(sprintf('{"success": %s, "model": "%s", "error": "%s"}',
                if (isTRUE(payload$success)) "true" else "false",
                MODEL_LABEL,
                if (is.null(payload$error)) "" else gsub('"', "'", payload$error)))
  }
  cat("\n")
  flush(stdout())
}

fail <- function(message, exit_code = 1) {
  emit_json(list(success = FALSE, error = message, model = MODEL_LABEL))
  quit(status = exit_code)
}

# ==============================================================================
# STEP 1 — Parse the incoming JSON payload
# ==============================================================================
raw_args <- commandArgs(trailingOnly = TRUE)

if (length(raw_args) == 0) {
  fail("No input received. Expected a JSON object with the applicant features.")
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
  fail("Input must be a JSON object containing the applicant features.")
}

# ==============================================================================
# STEP 2 — Load the SAVED model bundle (no retraining happens here)
# ==============================================================================
if (!file.exists(model_path)) {
  fail(paste0("Trained model not found at: ", model_path,
              ". Run 'Rscript r_models/decision_tree/train.R' to train the model first."))
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

tree_model <- model_bundle$model
feature_names <- model_bundle$features
feature_ranges <- model_bundle$feature_ranges
tree_structure <- model_bundle$tree

if (is.null(tree_structure)) {
  fail("The saved model bundle does not contain the extracted tree structure. Retrain with train.R.")
}

class_levels <- if (!is.null(model_bundle$class_levels)) {
  as.character(model_bundle$class_levels)
} else {
  as.character(attr(tree_model, "ylevels"))
}

# ==============================================================================
# STEP 3 — Validate the incoming feature values
#   A value is rejected when it is missing, non-numeric, non-finite, outside the
#   range the model was actually trained on, or breaks a domain rule the dataset
#   schema defines. Range checks come from model.rds (i.e. from the data), never
#   from hard-coded limits.
# ==============================================================================
coerce_numeric <- function(value) {
  if (is.null(value) || length(value) == 0) return(NA_real_)
  if (is.list(value)) value <- unlist(value, use.names = FALSE)
  as.numeric(value)
}

validation_errors <- character(0)
applicant_input <- list()

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

  # A value outside the trained range is still classifiable — a CART tree always
  # routes a record down some branch — so it is accepted here and reported to
  # the caller as an extrapolation warning rather than rejected.
  applicant_input[[feature_name]] <- numeric_value
}

# Collect (but do not reject) any feature that falls outside the training range.
extrapolation_warnings <- character(0)
for (feature_name in feature_names) {
  allowed <- feature_ranges[[feature_name]]
  value <- applicant_input[[feature_name]]
  if (is.null(allowed) || is.null(value)) next
  if (value < allowed$min || value > allowed$max) {
    extrapolation_warnings <- c(
      extrapolation_warnings,
      sprintf("%s (%s) is outside the %s - %s range the model was trained on, so this prediction is extrapolated.",
              feature_name, format(value, trim = TRUE),
              format(allowed$min, trim = TRUE), format(allowed$max, trim = TRUE))
    )
  }
}

# Domain rules that match the dataset's own documented constraints.
if (!is.null(applicant_input$age) && applicant_input$age <= 0) {
  validation_errors <- c(validation_errors, "Field 'age' must be greater than 0.")
}
if (!is.null(applicant_input$income) && applicant_input$income <= 0) {
  validation_errors <- c(validation_errors, "Field 'income' must be greater than 0.")
}
if (!is.null(applicant_input$credit_score)) {
  if (applicant_input$credit_score < 300 || applicant_input$credit_score > 900) {
    validation_errors <- c(validation_errors,
                           "Field 'credit_score' must be between 300 and 900.")
  }
}
if (!is.null(applicant_input$existing_loans) && applicant_input$existing_loans < 0) {
  validation_errors <- c(validation_errors, "Field 'existing_loans' must be 0 or greater.")
}
if (!is.null(applicant_input$employment_years) && applicant_input$employment_years < 0) {
  validation_errors <- c(validation_errors, "Field 'employment_years' must be 0 or greater.")
}

if (length(validation_errors) > 0) {
  emit_json(list(
    success = FALSE,
    model = MODEL_LABEL,
    error = paste(validation_errors, collapse = " "),
    errors = as.list(validation_errors)
  ))
  quit(status = 0)
}

# ==============================================================================
# STEP 4 — Run predict() on the saved, already-fitted model
# ==============================================================================
new_data <- as.data.frame(applicant_input, stringsAsFactors = FALSE, check.names = FALSE)

prediction <- tryCatch(
  as.character(predict(tree_model, newdata = new_data, type = "class"))[1],
  error = function(e) {
    fail(paste0("Prediction failed: ", conditionMessage(e)))
  }
)

if (length(prediction) == 0 || is.na(prediction) || prediction == "") {
  fail("The model produced an empty prediction for the supplied input.")
}

# The model can only ever answer with a class it was trained on. Anything else
# means the bundle and the model disagree, which is an error rather than a
# result to pass on to the user.
if (!(prediction %in% class_levels)) {
  fail(sprintf("The model returned the unknown risk class '%s'. Expected one of: %s.",
               prediction, paste(class_levels, collapse = ", ")))
}

# Real per-class probabilities straight from the fitted model.
probabilities <- tryCatch(
  predict(tree_model, newdata = new_data, type = "prob"),
  error = function(e) {
    fail(paste0("Could not compute class probabilities: ", conditionMessage(e)))
  }
)

probability_row <- as.numeric(probabilities[1, as.character(class_levels)])
names(probability_row) <- class_levels

if (any(is.na(probability_row))) {
  fail("The model returned incomplete class probabilities for the supplied input.")
}

# The confidence is the model's own probability for the class it predicted. It
# is reported only when it is a real number, never as a placeholder.
predicted_confidence <- probability_row[[prediction]]
confidence_value <- if (is.na(predicted_confidence)) {
  NULL
} else {
  round(as.numeric(predicted_confidence), 4)
}

# ==============================================================================
# STEP 5 — Walk the extracted tree to produce the REAL decision path
#   This is a trace of the fitted model, not a description written by hand:
#   each step is one node the record actually passed through, with the rule at
#   that node, the threshold in that rule, and the applicant's real value.
# ==============================================================================
walk_decision_path <- function(values) {
  node <- tree_structure
  steps <- list()
  guard <- 0L

  repeat {
    if (isTRUE(node$is_leaf) || is.null(node$variable)) break

    value <- as.numeric(values[[node$variable]])
    if (is.na(value)) break

    goes_left <- if (isTRUE(node$left_is_below)) value < node$cut else value >= node$cut
    taken_condition <- if (goes_left) node$left_condition else node$right_condition

    steps[[length(steps) + 1L]] <- list(
      step = length(steps) + 1L,
      node = as.integer(node$id),
      variable = node$variable,
      condition = taken_condition,
      threshold = as.numeric(node$cut),
      applicant_value = as.numeric(value),
      branch = if (goes_left) "yes" else "no",
      samples_at_node = as.integer(node$samples)
    )

    node <- if (goes_left) node$left else node$right
    guard <- guard + 1L
    if (guard > 64L) break
  }

  if (!is.null(node$predicted_class)) {
    steps[[length(steps) + 1L]] <- list(
      step = length(steps) + 1L,
      node = as.integer(node$id),
      variable = NULL,
      condition = sprintf("Reached leaf node %d -> %s RISK", node$id, node$predicted_class),
      threshold = NULL,
      applicant_value = NULL,
      branch = "leaf",
      samples_at_node = as.integer(node$samples)
    )
  }

  steps
}

decision_path <- walk_decision_path(applicant_input)

# The live predict() output is the authoritative probability set that is
# returned; the leaf probabilities stored in the bundle were already checked
# against it by train.R.
probability_payload <- list()
for (class_name in class_levels) {
  probability_payload[[class_name]] <- round(as.numeric(probability_row[[class_name]]), 4)
}

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
  model = MODEL_LABEL,
  algorithm = if (!is.null(model_bundle$algorithm)) model_bundle$algorithm else "rpart::rpart (CART)",
  prediction = prediction,
  confidence = confidence_value,
  class_probabilities = probability_payload,
  warnings = as.list(extrapolation_warnings),
  decision_path = decision_path,
  decision_depth = length(decision_path) - 1L,
  inputs = as.list(applicant_input),
  target = if (!is.null(model_bundle$target)) model_bundle$target else "risk",
  classes = class_levels,
  model_accuracy = if (!is.null(model_metrics$accuracy)) round(as.numeric(model_metrics$accuracy), 4) else NULL,
  model_precision = if (!is.null(model_metrics$precision)) round(as.numeric(model_metrics$precision), 4) else NULL,
  model_recall = if (!is.null(model_metrics$recall)) round(as.numeric(model_metrics$recall), 4) else NULL,
  model_f1_score = if (!is.null(model_metrics$f1_score)) round(as.numeric(model_metrics$f1_score), 4) else NULL,
  trained_at = if (!is.null(model_bundle$trained_at)) model_bundle$trained_at else NULL,
  r_version = if (!is.null(model_bundle$r_version)) model_bundle$r_version else NULL,
  rpart_version = if (!is.null(model_bundle$rpart_version)) model_bundle$rpart_version else NULL
)

emit_json(response)
