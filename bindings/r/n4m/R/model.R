#' Fit a PLS regression model via the libn4m C ABI.
#'
#' Accepts a numeric matrix X (n x p) and a numeric vector or matrix Y
#' (n x q). Both are coerced to double precision and row-major contiguous
#' before being passed across the C boundary.
#'
#' `algo` selects the solver. Recognized values:
#'   "pls_nipals", "pls_orthogonal_scores", "pls_simpls",
#'   "pls_kernel_algorithm", "pls_wide_kernel", "pls_svd",
#'   "pls_power", "pls_randomized_svd", "pcr_svd", "opls_nipals".
#'
#' @param X Numeric matrix, n x p.
#' @param Y Numeric matrix or vector.
#' @param algo Character. Solver name (see Details).
#' @param n_components Integer >= 1.
#'
#' @return An external pointer wrapping the fitted model handle. Pass it
#'   to [n4m_predict()] to obtain predictions. The model is freed
#'   automatically when the external pointer is garbage-collected.
#' @param store_scores Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param center_x Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param scale_x Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param center_y Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param scale_y Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param embedded_snv_savgol Optional two-number vector `c(window_length,
#'   polyorder)`. The native N4MM model then embeds default SNV followed by
#'   Savitzky-Golay smoothing, and predicts directly from raw spectra.
#' @export
n4m_fit <- function(X, Y, algo, n_components,
                         store_scores = FALSE,
                         center_x = TRUE, scale_x = TRUE,
                         center_y = TRUE, scale_y = TRUE,
                         embedded_snv_savgol = NULL) {
  if (!is.numeric(X)) stop("X must be numeric")
  if (!is.matrix(X)) X <- as.matrix(X)
  if (is.null(dim(Y))) Y <- matrix(as.numeric(Y), ncol = 1L)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  if (nrow(X) != nrow(Y)) {
    stop(sprintf("nrow(X) (%d) must equal nrow(Y) (%d)",
                 nrow(X), nrow(Y)))
  }
  if (!is.null(embedded_snv_savgol)) {
    params <- embedded_snv_savgol
    if (!is.numeric(params) || length(params) != 2L ||
        anyNA(params) || any(!is.finite(params)) ||
        any(params != floor(params)) || params[[1L]] < 3L ||
        params[[1L]] %% 2L != 1L || params[[2L]] < 0L ||
        params[[2L]] >= params[[1L]] ||
        any(params > .Machine$integer.max))
      stop("embedded_snv_savgol must contain an odd window >= 3 and a smaller non-negative degree")
    embedded_snv_savgol <- as.double(params)
  }
  .Call("r_n4m_fit",
        X, Y,
        as.character(algo),
        as.integer(n_components),
        as.logical(store_scores),
        as.logical(center_x), as.logical(scale_x),
        as.logical(center_y), as.logical(scale_y),
        embedded_snv_savgol,
        PACKAGE = "n4m")
}


#' Predict with a fitted n4m model.
#'
#' @param model External pointer returned by [n4m_fit()].
#' @param X Numeric matrix, n_new x p.
#' @return Numeric matrix, n_new x n_targets.
#' @export
n4m_predict <- function(model, X) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  .Call("r_n4m_predict", model, X, PACKAGE = "n4m")
}

#' Export a fitted n4m model in the portable N4MM format
#'
#' Unlike an R external pointer or RDS file, these bytes can be imported by
#' another libn4m binding (including Python) with a compatible N4MM reader.
#' Treat model bytes as untrusted input when importing from another source.
#'
#' @param model External pointer returned by [n4m_fit()] or [n4m_model_import()].
#' @return A raw vector containing one N4MM model.
#' @export
n4m_model_export <- function(model) {
  .Call("r_n4m_model_export", model, PACKAGE = "n4m")
}

#' Import a portable N4MM fitted model
#'
#' @param bytes Non-empty raw vector returned by [n4m_model_export()] or an
#'   equivalent libn4m binding. The native parser validates its format.
#' @return An external pointer with the same lifecycle as [n4m_fit()].
#' @export
n4m_model_import <- function(bytes) {
  .Call("r_n4m_model_import", bytes, PACKAGE = "n4m")
}

#' Import a native affine predictor into the portable N4MM model format
#'
#' Wraps the n4m C ABI affine importer. Coefficients are features-by-targets
#' and predictions are `X %*% coefficients + intercept`. The returned model
#' can be exported to N4MM and imported in other n4m bindings. This function
#' does not fit coefficients or claim a PLS latent decomposition.
#'
#' @param coefficients Finite numeric features-by-targets matrix.
#' @param intercept Finite numeric vector of one intercept per target.
#' @param source_training_samples Non-negative training row count, or zero
#'   when the source format does not attest it.
#' @return A native model pointer accepted by [n4m_predict()] and
#'   [n4m_model_export()].
#' @export
n4m_model_import_linear_predictor <- function(coefficients, intercept,
                                               source_training_samples = 0L) {
  if (!is.matrix(coefficients) || !is.numeric(coefficients) ||
      any(dim(coefficients) < 1L) || anyNA(coefficients) ||
      any(!is.finite(coefficients)))
    stop("coefficients must be a finite numeric matrix", call. = FALSE)
  if (!is.numeric(intercept) || is.matrix(intercept) ||
      length(intercept) != ncol(coefficients) || anyNA(intercept) ||
      any(!is.finite(intercept)))
    stop("intercept must be one finite numeric value per target", call. = FALSE)
  if (!is.numeric(source_training_samples) ||
      length(source_training_samples) != 1L ||
      !is.finite(source_training_samples) ||
      source_training_samples < 0L ||
      source_training_samples > .Machine$integer.max ||
      source_training_samples != floor(source_training_samples))
    stop("source_training_samples must be a non-negative integer", call. = FALSE)
  storage.mode(coefficients) <- "double"
  intercept <- as.double(intercept)
  .Call("r_n4m_model_import_linear_predictor", coefficients, intercept,
        as.integer(source_training_samples), PACKAGE = "n4m")
}

#' Inspect portable N4MM model metadata before import
#'
#' @param bytes Non-empty raw N4MM vector.
#' @return A list with `format_version` and `writer_abi`.
#' @export
n4m_model_inspect <- function(bytes) {
  .Call("r_n4m_model_inspect", bytes, PACKAGE = "n4m")
}

#' Inspect the native preprocessing state embedded in N4MM bytes
#'
#' Uses the authoritative libn4m decoder and rejects malformed payloads.
#' Format-1 models return `present = FALSE`. For the supported format-2
#' profile, the returned fingerprint is a hexadecimal FNV-1a-64 identity.
#' @param bytes Non-empty raw N4MM vector.
#' @return A list with `present`, `semantic_profile`, Savitzky-Golay parameters,
#'   raw/model feature widths, and `fingerprint`.
#' @export
n4m_model_pipeline_info <- function(bytes) {
  .Call("r_n4m_model_pipeline_info", bytes, PACKAGE = "n4m")
}

#' Inspect a fully validated N4MM model descriptor
#'
#' Unlike the header-only inspection, this validates all sections before
#' reporting the numerical recipe and capabilities. Enum values are the
#' versioned C ABI values from `n4m.h`.
#' @param bytes Non-empty raw N4MM vector.
#' @return A list of wire version, algorithm, solver, deflation, feature/target
#'   dimensions, component count, and capability mask.
#' @export
n4m_model_descriptor <- function(bytes) {
  .Call("r_n4m_model_descriptor", bytes, PACKAGE = "n4m")
}
