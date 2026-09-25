# ==============================================================================
# AI Insight Hub — R Execution Engine Health & Diagnostic Test Script
# Confirms R execution environment, package status, and standard input/output bridge.
# ==============================================================================

# Configure user library paths
lib_user <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_user) && dir.exists(lib_user)) {
  .libPaths(c(lib_user, .libPaths()))
}

suppressPackageStartupMessages({
  has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
})

# Read optional input arguments or standard input
args <- commandArgs(trailingOnly = TRUE)
input_data <- NULL

if (length(args) > 0) {
  raw_arg <- paste(args, collapse = " ")
  if (has_jsonlite) {
    tryCatch({
      input_data <- jsonlite::fromJSON(raw_arg)
    }, error = function(e) {
      input_data <- raw_arg
    })
  } else {
    input_data <- raw_arg
  }
}

# Check status of core required packages
checked_packages <- list(
  rpart = requireNamespace("rpart", quietly = TRUE),
  class = requireNamespace("class", quietly = TRUE),
  cluster = requireNamespace("cluster", quietly = TRUE),
  ggplot2 = requireNamespace("ggplot2", quietly = TRUE),
  jsonlite = has_jsonlite,
  readr = requireNamespace("readr", quietly = TRUE)
)

# Build diagnostic payload
timestamp_iso <- strftime(as.POSIXlt(Sys.time(), "UTC"), "%Y-%m-%dT%H:%M:%SZ")

response_list <- list(
  success = TRUE,
  engine = "R",
  message = "R engine executed successfully",
  r_version = sprintf("%s.%s (%s)", R.version$major, R.version$minor, R.version$status),
  platform = R.version$platform,
  timestamp = timestamp_iso,
  packages = checked_packages,
  input_received = input_data
)

# Output JSON
if (has_jsonlite) {
  cat(jsonlite::toJSON(response_list, auto_unbox = TRUE, pretty = TRUE))
} else {
  # Robust base R JSON fallback
  pkg_json <- paste(sapply(names(checked_packages), function(k) {
    sprintf('"%s": %s', k, if (checked_packages[[k]]) "true" else "false")
  }), collapse = ", ")

  cat(sprintf('{\n  "success": true,\n  "engine": "R",\n  "message": "R engine executed successfully",\n  "r_version": "%s",\n  "timestamp": "%s",\n  "packages": { %s }\n}',
              response_list$r_version, timestamp_iso, pkg_json))
}
cat("\n")
