/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * NIRS regression metrics — pure closed-form arithmetic over (y_true, y_pred).
 *
 * Mirrors `nirs4all.core.metrics` semantics. The "std" used in SD-based
 * metrics is the population (ddof=0) standard deviation, matching NumPy's
 * default `np.std(y)` (the formula used by nirs4all).
 *
 * All functions:
 *   - Return N4M_ERR_NULL_POINTER when any of y_true, y_pred, out is NULL.
 *   - Return N4M_ERR_INVALID_ARGUMENT when n <= 0.
 *   - Write the metric into *out and return N4M_OK on success.
 *   - Use NumPy "divide by zero → +inf, 0/0 → nan" semantics where the
 *     reference Python helper does the same; documented per function below.
 */
#ifndef N4M_CORE_UTILITIES_NIRS_METRICS_H
#define N4M_CORE_UTILITIES_NIRS_METRICS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RMSE = sqrt(mean((y_pred - y_true)^2)). */
n4m_status_t n4m_metric_rmse_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out);

/* MAE = mean(|y_pred - y_true|). */
n4m_status_t n4m_metric_mae_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out);

/* Bias = mean(y_pred - y_true). */
n4m_status_t n4m_metric_bias_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out);

/* SEP = std(y_pred - y_true), ddof=0. */
n4m_status_t n4m_metric_sep_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out);

/* RPD = std(y_true) / SEP. If SEP == 0 returns +inf (matches Python). */
n4m_status_t n4m_metric_rpd_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out);

/* RPIQ = IQR(y_true) / RMSE. IQR via NumPy's "linear" interpolation
 * (np.percentile default).  If RMSE == 0 returns +inf. */
n4m_status_t n4m_metric_rpiq_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out);

/* R² = 1 - SSE / SST. SST = sum((y_true - mean(y_true))^2). Constant-y
 * case (SST == 0) follows the sklearn convention: returns 1.0 when SSE == 0
 * (perfect prediction of a constant), 0.0 when SSE > 0 (any non-zero
 * residual against a constant ground truth gets no credit). */
n4m_status_t n4m_metric_r2_impl(const double* y_true, const double* y_pred,
                                 int64_t n, double* out);

/* NRMSE = RMSE / mean(y_true). If mean == 0 returns +inf. Matches the
 * Python implementation which divides by `np.mean(y_true)`. */
n4m_status_t n4m_metric_nrmse_impl(const double* y_true, const double* y_pred,
                                    int64_t n, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_UTILITIES_NIRS_METRICS_H */
