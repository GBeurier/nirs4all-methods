/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/scaling.h — transform.scaling methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_SCALING_H
#define N4M_TRANSFORM_SCALING_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Normalize (column-wise) -------------------------------------- */
typedef struct n4m_pp_normalize_handle_t n4m_pp_normalize_handle_t;
/* `(feature_min, feature_max) == (-1, 1)` selects linalg-norm mode; any
 * other pair selects user-defined-range mode. */
N4M_API n4m_status_t n4m_transform_normalize_create(n4m_pp_normalize_handle_t** out,
                                              double feature_min,
                                              double feature_max);
N4M_API void         n4m_transform_normalize_destroy(n4m_pp_normalize_handle_t* handle);
N4M_API n4m_status_t n4m_transform_normalize_transform(
    const n4m_pp_normalize_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- SimpleScale (column-wise min-max → [0, 1]) ------------------- */
typedef struct n4m_pp_simple_scale_handle_t n4m_pp_simple_scale_handle_t;
N4M_API n4m_status_t n4m_transform_simple_scale_create(
    n4m_pp_simple_scale_handle_t** out);
N4M_API void         n4m_transform_simple_scale_destroy(
    n4m_pp_simple_scale_handle_t* handle);
N4M_API n4m_status_t n4m_transform_simple_scale_transform(
    const n4m_pp_simple_scale_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- LogTransform ------------------------------------------------- */
typedef struct n4m_pp_log_handle_t n4m_pp_log_handle_t;
/* `base == 0.0` selects the natural logarithm. `base` must be > 0 and != 1
 * otherwise. `min_value` must be > 0 (positive clamp target).
 *
 * Lifecycle (Phase 5b split):
 *
 *   - When ``auto_offset == 0`` the operator is STATELESS: the user-supplied
 *     ``offset`` is applied verbatim. ``_transform`` may be called directly
 *     without calling ``_fit`` first.
 *
 *   - When ``auto_offset != 0`` the offset is captured at FIT TIME: the
 *     caller must invoke ``n4m_transform_log_transform_fit`` once on the training matrix.
 *     The fitted offset is cached on the handle and re-used by subsequent
 *     ``_transform`` calls. Calling ``_transform`` before ``_fit`` returns
 *     N4M_ERR_NOT_FITTED. Calling ``_fit`` again replaces the prior fit
 *     (sklearn-style refit semantics).
 *
 *   This preserves the pre-Phase-5b behaviour of ``auto_offset == 0`` while
 *   giving ``auto_offset != 0`` proper sklearn-style fit-once/transform-many
 *   semantics.
 */
N4M_API n4m_status_t n4m_transform_log_transform_create(n4m_pp_log_handle_t** out,
                                        double base, double offset,
                                        int auto_offset, double min_value);
N4M_API void         n4m_transform_log_transform_destroy(n4m_pp_log_handle_t* handle);
N4M_API n4m_status_t n4m_transform_log_transform_fit(n4m_pp_log_handle_t* handle,
                                     n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_log_transform_is_fitted(const n4m_pp_log_handle_t* handle,
                                           int* out_fitted);
N4M_API n4m_status_t n4m_transform_log_transform_transform(const n4m_pp_log_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);

/* ---------- Baseline (column-mean centering) ----------------------------- */
/*
 * Per-column mean centering. `_fit` learns `mean[j] = mean(X[:, j])`.
 * `_transform` writes `out[i, j] = X[i, j] - mean[j]`.
 * `_inverse_transform` writes `out[i, j] = X[i, j] + mean[j]`.
 *
 * Matches `nirs4all.operators.transforms.signal.Baseline` — equivalent to
 * `sklearn.preprocessing.StandardScaler(with_std=False)`. Note the class
 * name is historical; the operator does NOT perform baseline correction in
 * the spectroscopic sense (a polynomial baseline removal lives in Phase 5).
 */
typedef struct n4m_pp_baseline_handle_t n4m_pp_baseline_handle_t;
N4M_API n4m_status_t n4m_transform_baseline_center_create(n4m_pp_baseline_handle_t** out);
N4M_API void         n4m_transform_baseline_center_destroy(n4m_pp_baseline_handle_t* handle);
N4M_API n4m_status_t n4m_transform_baseline_center_fit(n4m_pp_baseline_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_baseline_center_transform(
    const n4m_pp_baseline_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_baseline_center_inverse_transform(
    const n4m_pp_baseline_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_baseline_center_is_fitted(
    const n4m_pp_baseline_handle_t* handle,
    int* out_fitted);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_SCALING_H */
