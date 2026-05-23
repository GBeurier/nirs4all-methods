#' n4m: portable Partial Least Squares and NIRS engine
#'
#' Implements a portable Partial Least Squares (PLS) and Near-Infrared
#' Spectroscopy (NIRS) engine. The C++17 numerical core is vendored under
#' the package's `src/libn4m/` and compiled from source at install time;
#' no external system library is required.
#'
#' The package exposes:
#'
#' * `pls()` / `predict.n4m_fit()` — the canonical PLS regression fit
#'   and inference entry points, with the underlying solver selected via
#'   the `algo` argument (NIPALS, SIMPLS, SVD, kernel, wide-kernel, OPLS,
#'   orthogonal-scores, power, randomized SVD, PCR).
#' * Method-result fits (e.g. `cppls_fit`, `sparse_pls_fit`,
#'   `weighted_pls_fit`) — direct wrappers over the C ABI.
#' * Variable-selection methods (`spa_select`, `cars_select`,
#'   `variable_select_rank`, ...).
#' * Diagnostics (`pls_diagnostics_compute`, `approximate_press_compute`,
#'   `pls_monitoring_run`).
#' * Calibration transfer (`pds_fit`, `ds_fit`).
#'
#' Version + ABI introspection is available via `n4m_version()` and
#' `n4m_abi_version()`.
#'
#' @name n4m-package
#' @aliases n4m
#' @useDynLib n4m, .registration = TRUE
#' @keywords internal
"_PACKAGE"
