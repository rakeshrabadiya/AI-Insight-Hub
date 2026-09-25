# ==============================================================================
# AI Insight Hub — Phase 5
# K-Nearest Neighbors Prediction Script (Student Performance Prediction)
#
# CONTRACT
#   Input  — a JSON object passed as the first command-line argument, e.g.
#              {"study_hours":6.5,"attendance":88,"previous_score":74,
#               "assignments_completed":9,"practical_score":81}
#   Output — a single JSON object printed to stdout:
#              {"success":true,"model":"K-Nearest Neighbors",
#               "prediction":"HIGH","k":5,"neighbors":[...]}
#
# HOW IT WORKS
#   Loads the SAVED model bundle (model.rds) that train.R produced and calls
#   class::knn() on the already-stored scaled training matrix. It does NOT
#   retrain anything, so a prediction costs the same whether the model is one
#   day old or one year old.
#
#   SCALING: the centre and scale saved in model.rds by train.R are applied to
#   the incoming record BEFORE class::knn() runs. This is the identical
#   transformation used to build the training matrix, which is what makes the
#   distance meaningful — the two sides of the comparison must be on the same
#   scale or the nearest neighbours would be meaningless.
#
#   NEIGHBOURS: class::knn() returns only the predicted class and an optional
#   proportion, so the K nearest training observations behind the answer are
#   recovered here with the same Euclidean metric class::knn() uses
#   internally. The reported `confidence` is NOT a probability invented by this
#   script: it is class::knn()'s own `prob` attribute, which the package defines
#   as the share of the K nearest neighbours belonging to the winning class. The
#   script re-derives that same share from the neighbour list and refuses to
#   answer if the two disagree, so the number is always traceable to an actual
#   count of actual neighbours.
#
# CALLED BY
#   backend/routes/knn.py via services.r_runner.execute_script()
#
# This is an educational ML implementation, not a real assessment tool.
# ==============================================================================

# --- Library configuration (kept in sync with the Python RRunner bridge) ------
lib_user <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_user) && dir.exists(lib_user)) {
  .libPaths(c(lib_user, .libPaths()))
}

has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
has_class <- requireNamespace("class", quietly = TRUE)

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

MODEL_LABEL <- "K-Nearest Neighbors"

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
  fail("No input received. Expected a JSON object with the student features.")
}

raw_input <- paste(raw_args, collapse = " ")

if (!has_jsonlite) {
  fail("The 'jsonlite' package is required to read the prediction input. Install it with: install.packages('jsonlite')")
}

if (!has_class) {
  fail("The 'class' package is required for prediction. Install it with: install.packages('class')")
}

input_data <- tryCatch(
  jsonlite::fromJSON(raw_input, simplifyVector = FALSE),
  error = function(e) {
    fail(paste0("Input is not valid JSON: ", conditionMessage(e)))
  }
)

if (!is.list(input_data) || is.null(names(input_data))) {
  fail("Input must be a JSON object containing the student features.")
}

# ==============================================================================
# STEP 2 — Load the SAVED model bundle (no retraining happens here)
# ==============================================================================
if (!file.exists(model_path)) {
  fail(paste0("Trained model not found at: ", model_path,
              ". Run 'Rscript r_models/knn/train.R' to train the model first."))
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

if (is.null(model_bundle$scaler) || is.null(model_bundle$scaler$center) ||
    is.null(model_bundle$scaler$scale)) {
  fail("The saved model bundle does not contain the feature scaling it was trained with. Retrain with train.R.")
}

train_x <- model_bundle$model$train_x
train_y <- model_bundle$model$train_y
SELECTED_K <- as.integer(model_bundle$k)
scaler_center <- model_bundle$scaler$center
scaler_scale <- model_bundle$scaler$scale
feature_names <- model_bundle$features
feature_ranges <- model_bundle$feature_ranges
class_levels <- as.character(model_bundle$class_levels)

if (is.null(train_x) || is.null(train_y)) {
  fail("The saved model bundle does not contain the training data. Retrain with train.R.")
}

# The stored labels must line up with the stored matrix row for row, otherwise
# the neighbours below would be attributed to the wrong students.
if (nrow(as.matrix(train_x)) != length(train_y)) {
  fail("The saved model's training matrix and labels disagree in length. Retrain with train.R.")
}

# ==============================================================================
# STEP 3 — Validate the incoming feature values
#   A value is rejected when it is missing, non-numeric, non-finite, or breaks a
#   domain rule the dataset schema defines. Range checks against the trained
#   data come from model.rds (i.e. from the data), never from hard-coded limits,
#   and produce a WARNING rather than a rejection — see below.
# ==============================================================================
coerce_numeric <- function(value) {
  if (is.null(value) || length(value) == 0) return(NA_real_)
  if (is.list(value)) value <- unlist(value, use.names = FALSE)
  as.numeric(value)
}

validation_errors <- character(0)
student_input <- list()

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

  student_input[[feature_name]] <- numeric_value
}

# Collect (but do not reject) any feature that falls outside the training range.
# A KNN model can still answer for an out-of-range point — the nearest training
# rows are simply further away than usual — so the record is classified and the
# caller is told the result is an extrapolation.
extrapolation_warnings <- character(0)
for (feature_name in feature_names) {
  allowed <- feature_ranges[[feature_name]]
  value <- student_input[[feature_name]]
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
if (!is.null(student_input$study_hours) && student_input$study_hours < 0) {
  validation_errors <- c(validation_errors, "Field 'study_hours' must be 0 or greater.")
}
if (!is.null(student_input$attendance)) {
  if (student_input$attendance < 0 || student_input$attendance > 100) {
    validation_errors <- c(validation_errors, "Field 'attendance' must be between 0 and 100.")
  }
}
if (!is.null(student_input$previous_score)) {
  if (student_input$previous_score < 0 || student_input$previous_score > 100) {
    validation_errors <- c(validation_errors, "Field 'previous_score' must be between 0 and 100.")
  }
}
if (!is.null(student_input$assignments_completed) && student_input$assignments_completed < 0) {
  validation_errors <- c(validation_errors, "Field 'assignments_completed' must be 0 or greater.")
}
if (!is.null(student_input$practical_score)) {
  if (student_input$practical_score < 0 || student_input$practical_score > 100) {
    validation_errors <- c(validation_errors, "Field 'practical_score' must be between 0 and 100.")
  }
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
# STEP 4 — Apply the SAVED scaling, then run class::knn()
#   The centre and scale come from model.rds, so the incoming record is put on
#   exactly the same scale as the training matrix it will be compared against.
# ==============================================================================
raw_vector <- vapply(feature_names, function(feature_name) {
  as.numeric(student_input[[feature_name]])
}, numeric(1))

scaled_vector <- (raw_vector - as.numeric(scaler_center[feature_names])) /
                 as.numeric(scaler_scale[feature_names])

# A record with any missing scaled value cannot be compared to the training
# matrix, so it is an error rather than a silent prediction.
if (any(is.na(scaled_vector)) || any(is.infinite(scaled_vector))) {
  fail("The supplied values could not be scaled with the model's saved scaling parameters. Retrain with train.R.")
}

query_matrix <- matrix(scaled_vector, nrow = 1, ncol = length(feature_names),
                       dimnames = list(NULL, feature_names))

prediction_result <- tryCatch(
  class::knn(as.matrix(train_x), query_matrix, train_y, k = SELECTED_K, prob = TRUE),
  error = function(e) {
    fail(paste0("Prediction failed: ", conditionMessage(e)))
  }
)

prediction <- as.character(prediction_result)[1]

if (length(prediction) == 0 || is.na(prediction) || prediction == "") {
  fail("The model produced an empty prediction for the supplied input.")
}

# The model can only ever answer with a class it was trained on. Anything else
# means the bundle and the model disagree, which is an error rather than a
# result to pass on to the user.
if (!(prediction %in% class_levels)) {
  fail(sprintf("The model returned the unknown performance class '%s'. Expected one of: %s.",
               prediction, paste(class_levels, collapse = ", ")))
}

# ==============================================================================
# STEP 5 — Recover the K nearest training observations
#   Same Euclidean metric class::knn() uses internally, via
#       ||a - b||^2 = |a|^2 - 2*a.b + |b|^2
# ==============================================================================
euclidean_distance_matrix <- function(train_matrix, query_matrix) {
  train_matrix <- as.matrix(train_matrix)
  query_matrix <- as.matrix(query_matrix)
  squared <- outer(rowSums(query_matrix^2), rowSums(train_matrix^2), "+") -
             2 * (query_matrix %*% t(train_matrix))
  squared[squared < 0] <- 0
  sqrt(squared)
}

distance_matrix <- euclidean_distance_matrix(as.matrix(train_x), query_matrix)
distances <- distance_matrix[1, ]

if (length(distances) < SELECTED_K) {
  fail(sprintf("The saved model has only %d training rows but K is %d. Retrain with train.R.",
               length(distances), SELECTED_K))
}

neighbor_positions <- order(distances)[seq_len(SELECTED_K)]
neighbor_distances <- as.numeric(distances[neighbor_positions])
neighbor_classes <- as.character(train_y[neighbor_positions])

# The true (un-standardised) values of each neighbour's features, so the UI can
# show what the nearest training students actually looked like. These are real
# rows from the dataset, recovered by undoing the saved scaling.
neighbor_features <- lapply(neighbor_positions, function(position) {
  row <- as.numeric(as.matrix(train_x)[position, ])
  original <- row * as.numeric(scaler_scale[feature_names]) +
              as.numeric(scaler_center[feature_names])
  values <- list()
  for (j in seq_along(feature_names)) {
    values[[feature_names[j]]] <- round(original[j], 4)
  }
  values
})

# ==============================================================================
# STEP 6 — The confidence, and a self-check that it is real
#   class::knn()'s `prob` attribute is the share of the K nearest neighbours
#   that belong to the winning class. This script re-derives that same share
#   from the neighbour list it just built. If the two do not agree, the two
#   halves of the computation have diverged and the answer is refused rather
#   than reported.
# ==============================================================================
neighbor_class_table <- table(neighbor_classes)
majority_from_neighbors <- as.character(names(sort(neighbor_class_table, decreasing = TRUE))[1])
derived_confidence <- unname(max(neighbor_class_table) / SELECTED_K)

# The neighbour vote must reproduce the class the model actually predicted.
if (!identical(majority_from_neighbors, prediction)) {
  fail(sprintf(paste0("The recovered neighbours vote for '%s' but the model predicted '%s'. ",
                      "The reported neighbours would not be the ones the model used."),
               majority_from_neighbors, prediction))
}

knn_confidence <- attr(prediction_result, "prob")
if (!is.null(knn_confidence)) {
  if (abs(derived_confidence - as.numeric(knn_confidence)) > 1e-9) {
    fail("The confidence derived from the neighbour list disagrees with class::knn()'s own value.")
  }
}

confidence_value <- round(derived_confidence, 4)

# Class distribution across the K neighbours — a real count, not a probability
# invented by this script.
neighbor_distribution <- list()
for (class_name in class_levels) {
  count_value <- if (class_name %in% names(neighbor_class_table)) {
    as.integer(neighbor_class_table[[class_name]])
  } else {
    0L
  }
  neighbor_distribution[[class_name]] <- count_value
}

neighbor_payload <- lapply(seq_len(SELECTED_K), function(i) {
  list(
    neighbor = as.integer(i),
    training_row = as.integer(neighbor_positions[i]),
    neighbor_class = neighbor_classes[i],
    distance = round(neighbor_distances[i], 6),
    scaled_features = neighbor_features[[i]]
  )
})

# ==============================================================================
# STEP 7 — Attach the model's real training metrics
#   So the client always displays numbers that came from training, never from a
#   literal in the response.
# ==============================================================================
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
  algorithm = if (!is.null(model_bundle$algorithm)) model_bundle$algorithm else "class::knn (Euclidean distance)",
  prediction = prediction,
  k = SELECTED_K,
  confidence = confidence_value,
  confidence_basis = paste(
    sprintf("Share of the %d nearest training neighbours belonging to the predicted class,",
            SELECTED_K),
    "as reported by class::knn() and re-derived from the neighbour list."
  ),
  neighbor_class_distribution = neighbor_distribution,
  neighbors = neighbor_payload,
  warnings = as.list(extrapolation_warnings),
  inputs = as.list(student_input),
  scaled_inputs = as.list(setNames(
    as.list(round(as.numeric(scaled_vector), 6)), feature_names
  )),
  target = if (!is.null(model_bundle$target)) model_bundle$target else "performance",
  classes = class_levels,
  features = feature_names,
  model_accuracy = if (!is.null(model_metrics$accuracy)) round(as.numeric(model_metrics$accuracy), 4) else NULL,
  model_precision = if (!is.null(model_metrics$precision)) round(as.numeric(model_metrics$precision), 4) else NULL,
  model_recall = if (!is.null(model_metrics$recall)) round(as.numeric(model_metrics$recall), 4) else NULL,
  model_f1_score = if (!is.null(model_metrics$f1_score)) round(as.numeric(model_metrics$f1_score), 4) else NULL,
  trained_at = if (!is.null(model_bundle$trained_at)) model_bundle$trained_at else NULL,
  r_version = if (!is.null(model_bundle$r_version)) model_bundle$r_version else NULL,
  class_package_version = if (!is.null(model_bundle$class_package_version)) model_bundle$class_package_version else NULL,
  scaling_applied = if (!is.null(model_bundle$scaler$method)) model_bundle$scaler$method else NULL
)

emit_json(response)
