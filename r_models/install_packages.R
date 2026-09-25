# ==============================================================================
# AI Insight Hub — R Dependency Package Installer
# Installs all required packages for Phase 2–6 machine learning models
# ==============================================================================

# Configure user library path if system library is not writable
lib_user <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_user)) {
  if (!dir.exists(lib_user)) {
    dir.create(lib_user, showWarnings = FALSE, recursive = TRUE)
  }
  .libPaths(c(lib_user, .libPaths()))
}

required_packages <- c(
  "rpart",      # Decision Tree Classification (Phase 4)
  "class",      # K-Nearest Neighbors (KNN) (Phase 5)
  "cluster",    # K-Means Clustering & Silhouette Analysis (Phase 6)
  "ggplot2",    # High-quality visualization and plotting
  "jsonlite",   # Fast JSON serialization/deserialization for API bridge
  "readr"       # Fast tabular data parsing
)

cat("=====================================================\n")
cat("AI Insight Hub — Checking and Installing R Packages\n")
cat("=====================================================\n")
cat(sprintf("R Version: %s\n", R.version.string))
cat(sprintf("Active Library Paths:\n  %s\n\n", paste(.libPaths(), collapse = "\n  ")))

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) == 0) {
  cat("All required R packages are already installed and verified!\n")
} else {
  cat(sprintf("Installing %d missing package(s): %s\n", length(missing_packages), paste(missing_packages, collapse = ", ")))

  target_lib <- if (nzchar(lib_user)) lib_user else NULL
  if (!is.null(target_lib)) {
    install.packages(missing_packages, lib = target_lib, repos = "https://cloud.r-project.org")
  } else {
    install.packages(missing_packages, repos = "https://cloud.r-project.org")
  }
}

# Verification Report
cat("\n--- Package Verification Report ---\n")
results <- data.frame(
  Package = required_packages,
  Status = sapply(required_packages, function(pkg) {
    if (requireNamespace(pkg, quietly = TRUE)) "INSTALLED" else "MISSING"
  })
)
print(results, row.names = FALSE)
cat("=====================================================\n")
