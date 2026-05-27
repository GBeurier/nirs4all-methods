#' pls4all: Partial Least Squares engine (PLS-core subset of nirs4all-methods)
#'
#' The slim, PLS-focused distribution of the nirs4all-methods engine: every
#' method built on the shared PLS core (NIPALS, SIMPLS, kernel, ...). The C++17
#' numerical core is vendored under the package's `src/vendor/` and compiled
#' from source at install time; no external system library is required. The
#' spectroscopy surface (preprocessing, augmentation, filters, signal-type)
#' lives in the full nirs4all-methods distribution.
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
#' @name pls4all-package
#' @aliases pls4all
#' @useDynLib pls4all, .registration = TRUE
#' @keywords internal
"_PACKAGE"
