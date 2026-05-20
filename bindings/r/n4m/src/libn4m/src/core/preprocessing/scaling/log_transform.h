/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LogTransform — element-wise logarithm with optional safety offset.
 *
 * Reference: nirs4all.operators.transforms.nirs.LogTransform
 *   X_temp = X + offset
 *   if auto_offset and min(X_temp) <= 0:
 *       fitted_offset = offset + (min_value - min(X_temp))
 *   else:
 *       fitted_offset = offset
 *   X' = X + fitted_offset                # uses ORIGINAL X, not X_temp
 *   X' = where(X' <= 0, min_value, X')
 *   X' = log(X') / log(base)              # natural log if base == e
 *
 * Phase 5b lifecycle split:
 *   - When ``auto_offset == 0`` the operator is stateless: ``_apply`` (and the
 *     public ``_transform`` wrapper) may be called directly with no fit step.
 *     This preserves the original Phase 2 behaviour for callers that pass a
 *     literal offset.
 *   - When ``auto_offset != 0`` the offset is computed from the fit data via
 *     ``n4m_pp_log_state_fit``. Subsequent ``_apply`` calls reuse the cached
 *     ``fitted_offset``. Calling ``_apply`` before ``_fit`` in this mode
 *     returns N4M_ERR_NOT_FITTED.
 */
#ifndef N4M_CORE_PP_SCALING_LOG_TRANSFORM_H
#define N4M_CORE_PP_SCALING_LOG_TRANSFORM_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_log_state_t n4m_pp_log_state_t;

/* `base` is the log base; pass 0.0 (or e) to select the natural logarithm.
 * `offset` is the manual additive offset. `auto_offset` enables the
 * fit-time safety offset that keeps the post-offset minimum >= `min_value`.
 */
n4m_pp_log_state_t* n4m_pp_log_state_new(double base, double offset,
                                          int auto_offset, double min_value);

void n4m_pp_log_state_free(n4m_pp_log_state_t* state);

/* Fit step (auto_offset mode): scan X, compute and cache the safety offset
 * that makes min(X + offset) >= min_value. Sets the fitted flag. May be
 * called even when auto_offset == 0 (then it just marks fitted=1, leaving
 * the offset alone); calling it again replaces the prior fit. */
n4m_status_t n4m_pp_log_state_fit(n4m_pp_log_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols);

/* Apply step. When ``auto_offset != 0`` the state must have been fitted;
 * otherwise returns N4M_ERR_NOT_FITTED. When ``auto_offset == 0`` the
 * call is stateless and uses the user-supplied offset directly. */
n4m_status_t n4m_pp_log_apply(const n4m_pp_log_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out);

/* Introspection helper used by the C ABI wrapper. Returns 1 if the state
 * has been fitted (or 0 / 1 with auto_offset == 0 — the stateless path).
 * Never NULL-deref's; returns 0 for a NULL state. */
int n4m_pp_log_state_is_fitted(const n4m_pp_log_state_t* state);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCALING_LOG_TRANSFORM_H */
