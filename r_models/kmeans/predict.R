# ==============================================================================
# AI Insight Hub — Phase 6
# K-Means Customer Assignment Script (Customer Segmentation)
#
# CONTRACT
#   Input  — a JSON object passed as the first command-line argument, e.g.
#              {"age":42,"annual_income":85,"spending_score":55,
#               "purchase_frequency":28}
#   Output — a single JSON object printed to stdout:
#              {"success":true,"model":"K-Means","cluster":4,"distance":0.61}
#
# WHY THIS IS NOT A "PREDICTION"
#   K-Means is unsupervised, so there is no label being predicted. What this
#   script does is take the four SAVED cluster centres, scale the incoming
#   customer with the SAME scaling the centres were built from, and return the
#   centre that lies nearest in Euclidean distance. That nearest-centre rule IS
#   what stats::kmeans() itself does to assign its own training rows, so this
#   script and the trained model cannot disagree.
#
# HOW IT STAYS HONEST
#   * It loads model.rds and refits NOTHING. A request costs the same whether
#     the model is one day old or one year old.
#   * It applies the saved scaler, not a freshly computed one. The two sides of
#     the distance must be on the same scale or the answer is meaningless.
#   * It reports the distance to EVERY centre, so the caller can see the full
#     ranking rather than only the winner.
#   * The clustering used the four scaled features, so the distance returned is
#     in standardised feature space. The same distance is also reported on the
#     original units so the UI can show a human-readable figure.
#
# CALLED BY
#   backend/routes/kmeans.py via services.r_runner.execute_script()
#
# This is an educational ML implementation, not a real commercial tool.
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

MODEL_LABEL <- "K-Means"

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
  fail("No input received. Expected a JSON object with the customer features.")
}

raw_input <- paste(raw_args, collapse = " ")

if (!has_jsonlite) {
  fail("The 'jsonlite' package is required to read the assignment input. Install it with: install.packages('jsonlite')")
}

input_data <- tryCatch(
  jsonlite::fromJSON(raw_input, simplifyVector = FALSE),
  error = function(e) {
    fail(paste0("Input is not valid JSON: ", conditionMessage(e)))
  }
)

if (!is.list(input_data) || is.null(names(input_data))) {
  fail("Input must be a JSON object containing the customer features.")
}

# ==============================================================================
# STEP 2 — Load the SAVED model bundle (no retraining happens here)
# ==============================================================================
if (!file.exists(model_path)) {
  fail(paste0("Trained model not found at: ", model_path,
              ". Run 'Rscript r_models/kmeans/train.R' to train the model first."))
}

model_bundle <- tryCatch(
  readRDS(model_path),
  error = function(e) {
    fail(paste0("Failed to load the trained model: ", conditionMessage(e)))
  }
)

if (is.null(model_bundle$cluster_centers) || is.null(model_bundle$features)) {
  fail("The saved model bundle is malformed or was produced by an incompatible training script.")
}

if (is.null(model_bundle$scaler) || is.null(model_bundle$scaler$center) ||
    is.null(model_bundle$scaler$scale)) {
  fail("The saved model bundle does not contain the feature scaling it was trained with. Retrain with train.R.")
}

centers <- as.matrix(model_bundle$cluster_centers)
feature_names <- model_bundle$features
scaler_center <- model_bundle$scaler$center
scaler_scale <- model_bundle$scaler$scale
feature_ranges <- model_bundle$feature_ranges
SELECTED_K <- as.integer(model_bundle$k)
population_mean <- model_bundle$population_mean
cluster_profiles <- model_bundle$cluster_profiles
pca_loadings <- model_bundle$pca

if (is.null(SELECTED_K) || is.na(SELECTED_K) || SELECTED_K < 1) {
  fail("The saved model bundle does not record a valid K. Retrain with train.R.")
}

# The centres and the feature list must describe the same space, otherwise the
# distance below would be comparing different quantities.
if (ncol(centers) != length(feature_names)) {
  fail("The saved cluster centres do not match the model's feature list. Retrain with train.R.")
}

if (nrow(centers) != SELECTED_K) {
  fail(sprintf("The saved model has %d cluster centres but records K = %d. Retrain with train.R.",
               nrow(centers), SELECTED_K))
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
  # A non-numeric value coerces to NA with a warning; the warning is suppressed
  # here because the NA is detected and reported as a proper JSON error below,
  # so stderr stays clean for the caller.
  suppressWarnings(as.numeric(value))
}

validation_errors <- character(0)
customer_input <- list()

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

  customer_input[[feature_name]] <- numeric_value
}

# Reject an unrecognised field so a typo like "income" is not silently ignored.
unknown_fields <- setdiff(names(input_data), feature_names)
if (length(unknown_fields) > 0) {
  validation_errors <- c(validation_errors,
                         sprintf("Unrecognised field(s): %s. Expected: %s.",
                                 paste(unknown_fields, collapse = ", "),
                                 paste(feature_names, collapse = ", ")))
}

# Collect (but do not reject) any feature that falls outside the training range.
# A centroid model can still place such a customer - the nearest centre is simply
# further away than usual - so the assignment is made and the caller is told the
# result is an extrapolation beyond the data the model was fitted on.
extrapolation_warnings <- character(0)
for (feature_name in feature_names) {
  allowed <- feature_ranges[[feature_name]]
  value <- customer_input[[feature_name]]
  if (is.null(allowed) || is.null(value)) next
  if (value < allowed$min || value > allowed$max) {
    extrapolation_warnings <- c(
      extrapolation_warnings,
      sprintf("%s (%s) is outside the %s - %s range the model was trained on, so this assignment extrapolates beyond the dataset.",
              feature_name, format(value, trim = TRUE),
              format(allowed$min, trim = TRUE), format(allowed$max, trim = TRUE))
    )
  }
}

# Domain rules that match the dataset's own documented constraints.
if (!is.null(customer_input$age) && (customer_input$age < 0 || customer_input$age > 120)) {
  validation_errors <- c(validation_errors, "Field 'age' must be between 0 and 120.")
}
if (!is.null(customer_input$annual_income) && customer_input$annual_income < 0) {
  validation_errors <- c(validation_errors, "Field 'annual_income' must be 0 or greater.")
}
if (!is.null(customer_input$spending_score) &&
    (customer_input$spending_score < 0 || customer_input$spending_score > 100)) {
  validation_errors <- c(validation_errors, "Field 'spending_score' must be between 0 and 100.")
}
if (!is.null(customer_input$purchase_frequency) && customer_input$purchase_frequency < 0) {
  validation_errors <- c(validation_errors, "Field 'purchase_frequency' must be 0 or greater.")
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
# STEP 4 — Apply the SAVED scaling, then measure against every centre
#   The centre and scale come from model.rds, so the incoming customer is put on
#   exactly the same scale as the centres it will be compared against.
# ==============================================================================
raw_vector <- vapply(feature_names, function(feature_name) {
  as.numeric(customer_input[[feature_name]])
}, numeric(1))
names(raw_vector) <- feature_names

scaler_center_values <- as.numeric(scaler_center[feature_names])
scaler_scale_values <- as.numeric(scaler_scale[feature_names])

scaled_vector <- (raw_vector - scaler_center_values) / scaler_scale_values
names(scaled_vector) <- feature_names

# A record with any missing scaled value cannot be compared to the centres, so
# this is an error rather than a silent assignment.
if (any(is.na(scaled_vector)) || any(is.infinite(scaled_vector))) {
  fail("The supplied values could not be scaled with the model's saved scaling parameters. Retrain with train.R.")
}

# Distance from the scaled customer to every cluster centre, in the same
# standardised feature space kmeans() itself minimised.
center_distances <- vapply(seq_len(SELECTED_K), function(cluster_id) {
  center_vector <- as.numeric(centers[cluster_id, feature_names])
  sqrt(sum((scaled_vector - center_vector)^2))
}, numeric(1))

nearest_position <- which.min(center_distances)
nearest_distance <- as.numeric(center_distances[[nearest_position]])
runner_up_distance <- if (SELECTED_K > 1) {
  as.numeric(sort(center_distances)[[2]])
} else {
  NA_real_
}

# The same distance expressed in the ORIGINAL units, by undoing the scaling on
# the customer's side and comparing against the un-scaled centres. Rounded
# heavily because a 4-dimension sum of squares is not meaningful to many digits.
original_centers <- sweep(
  sweep(as.matrix(centers), 2L, scaler_scale_values, "*"),
  2L, scaler_center_values, "+"
)
original_distance <- sqrt(
  sum((raw_vector - as.numeric(original_centers[nearest_position, feature_names]))^2)
)

# How far this customer sits from its centre compared with the next-closest one.
# Below 1 means the nearest centre really is meaningfully nearer than the runner-up.
separation_ratio <- if (!is.na(runner_up_distance) && runner_up_distance > 0) {
  nearest_distance / runner_up_distance
} else {
  NA_real_
}

# ==============================================================================
# STEP 5 — Attach the profile of the assigned cluster
#   Read straight from the profiles train.R derived from that cluster's own
#   measured centre. If the bundle has none, the assignment is still returned
#   with a neutral name rather than refusing to answer.
# ==============================================================================
profile <- NULL
if (!is.null(cluster_profiles)) {
  for (candidate in cluster_profiles) {
    if (identical(as.integer(candidate$cluster), as.integer(nearest_position))) {
      profile <- candidate
      break
    }
  }
}

cluster_label <- if (is.null(profile)) {
  sprintf("Cluster %d", nearest_position)
} else {
  as.character(profile$label)
}
cluster_size <- if (is.null(profile)) NULL else as.integer(profile$size)
segment_label <- if (is.null(profile)) NULL else as.character(profile$segment)

# The assigned cluster's centre in the original units, next to the customer's
# own values, so the UI can show the gap between them per feature.
center_original_payload <- as.list(stats::setNames(
  as.numeric(round(as.numeric(original_centers[nearest_position, feature_names]), 4)),
  feature_names
))

# The customer's position on the same 2-D PCA basis the dashboard scatter uses.
# The clustering did NOT run on these components - this is display only, and it
# reuses the exact loadings stored at training time.
projection <- NULL
if (!is.null(pca_loadings) && !is.null(pca_loadings$pc1_loadings) &&
    !is.null(pca_loadings$pc2_loadings)) {
  pc1 <- as.numeric(pca_loadings$pc1_loadings)[match(feature_names, names(pca_loadings$pc1_loadings))]
  pc2 <- as.numeric(pca_loadings$pc2_loadings)[match(feature_names, names(pca_loadings$pc2_loadings))]
  if (all(is.finite(pc1)) && all(is.finite(pc2))) {
    projection <- list(
      PC1 = round(as.numeric(scaled_vector %*% pc1), 4),
      PC2 = round(as.numeric(scaled_vector %*% pc2), 4)
    )
  }
}

# The distance to every centre, so the UI can show the full ranking and make
# clear how decisively the nearest one won.
distance_to_all <- stats::setNames(
  as.list(round(center_distances, 6)),
  paste0("Cluster ", seq_len(SELECTED_K))
)

# ==============================================================================
# STEP 6 — Attach the model's real training metrics
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

safe_number <- function(value) {
  if (is.null(value) || length(value) == 0 || is.na(value) || is.nan(value) || is.infinite(value)) {
    return(NULL)
  }
  round(as.numeric(value), 6)
}

response <- list(
  success = TRUE,
  model = MODEL_LABEL,
  algorithm = if (!is.null(model_bundle$algorithm)) model_bundle$algorithm else "stats::kmeans (Euclidean distance)",
  task = if (!is.null(model_bundle$task)) model_bundle$task else "Customer Segmentation",
  learning_type = "unsupervised",
  # The nearest-centre result, and the distance that decided it.
  cluster = as.integer(nearest_position),
  distance = round(nearest_distance, 6),
  distance_original_units = round(as.numeric(original_distance), 6),
  runner_up_distance = safe_number(runner_up_distance),
  separation_ratio = safe_number(separation_ratio),
  distance_basis = "Euclidean distance in standardised feature space, the same space kmeans() clustered in",
  distances = distance_to_all,
  cluster_label = cluster_label,
  cluster_size = cluster_size,
  segment = segment_label,
  cluster_centers_original = center_original_payload,
  inputs = as.list(stats::setNames(
    as.numeric(customer_input[feature_names]), feature_names
  )),
  scaled_inputs = as.list(stats::setNames(as.numeric(round(scaled_vector, 6)), feature_names)),
  features = feature_names,
  k = as.integer(SELECTED_K),
  projection = projection,
  warnings = as.list(extrapolation_warnings),
  scaling_applied = if (!is.null(model_bundle$scaler$method)) model_bundle$scaler$method else NULL,
  trained_at = if (!is.null(model_bundle$trained_at)) model_bundle$trained_at else NULL,
  r_version = if (!is.null(model_bundle$r_version)) model_bundle$r_version else NULL,
  model_wss = if (!is.null(model_metrics$wss)) safe_number(model_metrics$wss) else NULL,
  model_silhouette_score = if (!is.null(model_metrics$silhouette_score)) safe_number(model_metrics$silhouette_score) else NULL,
  model_records_clustered = if (!is.null(model_metrics$records_clustered)) as.integer(model_metrics$records_clustered) else NULL
)

emit_json(response)
