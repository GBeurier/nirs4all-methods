/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * NIRS regression metrics — pure closed-form arithmetic over (y_true, y_pred).
 *
 * Mirrors `nirs4all.core.metrics` semantics. The "std" used in SD-based
 * metrics is the population (ddof=0) standard deviation, matching NumPy's
 * default `np.std(y)` (the formula used by nirs4all).
 *
 * All functions:
 *   - Return C4A_ERR_NULL_POINTER when any of y_true, y_pred, out is NULL.
 *   - Return C4A_ERR_INVALID_ARGUMENT when n <= 0.
 *   - Write the metric into *out and return C4A_OK on success.
 *   - Use NumPy "divide by zero → +inf, 0/0 → nan" semantics where the
 *     reference Python helper does the same; documented per function below.
 */
#ifndef CHEMOMETRICS4ALL_CORE_UTILITIES_NIRS_METRICS_H
#define CHEMOMETRICS4ALL_CORE_UTILITIES_NIRS_METRICS_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RMSE = sqrt(mean((y_pred - y_true)^2)). */
c4a_status_t c4a_metric_rmse_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out);

/* MAE = mean(|y_pred - y_true|). */
c4a_status_t c4a_metric_mae_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out);

/* Bias = mean(y_pred - y_true). */
c4a_status_t c4a_metric_bias_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out);

/* SEP = std(y_pred - y_true), ddof=0. */
c4a_status_t c4a_metric_sep_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out);

/* RPD = std(y_true) / SEP. If SEP == 0 returns +inf (matches Python). */
c4a_status_t c4a_metric_rpd_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out);

/* RPIQ = IQR(y_true) / RMSE. IQR via NumPy's "linear" interpolation
 * (np.percentile default).  If RMSE == 0 returns +inf. */
c4a_status_t c4a_metric_rpiq_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out);

/* R² = 1 - SSE / SST. SST = sum((y_true - mean(y_true))^2). If SST == 0 and
 * SSE == 0 returns 1.0 (perfect-constant case, matches sklearn default of
 * 0/0 → 0 is intentionally NOT used — we mirror the canonical formula). */
c4a_status_t c4a_metric_r2_impl(const double* y_true, const double* y_pred,
                                 int64_t n, double* out);

/* NRMSE = RMSE / mean(y_true). If mean == 0 returns +inf. Matches the
 * Python implementation which divides by `np.mean(y_true)`. */
c4a_status_t c4a_metric_nrmse_impl(const double* y_true, const double* y_pred,
                                    int64_t n, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_UTILITIES_NIRS_METRICS_H */
