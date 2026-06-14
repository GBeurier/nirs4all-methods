/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/domain_adaptation.h — domain_adaptation role header (ABI 2.0). */
#ifndef N4M_DOMAIN_ADAPTATION_H
#define N4M_DOMAIN_ADAPTATION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- External Parameter Orthogonalisation (Roger 2003) ---------- */
typedef struct n4m_pp_epo_handle_t n4m_pp_epo_handle_t;
N4M_API n4m_status_t n4m_domain_adaptation_epo_create(n4m_pp_epo_handle_t** out, int scale);
N4M_API void         n4m_domain_adaptation_epo_destroy(n4m_pp_epo_handle_t* handle);
N4M_API n4m_status_t n4m_domain_adaptation_epo_fit(n4m_pp_epo_handle_t* handle,
                                     n4m_matrix_view_t X,
                                     const double* d, int64_t d_len);
N4M_API n4m_status_t n4m_domain_adaptation_epo_transform(const n4m_pp_epo_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_domain_adaptation_epo_transform_with_d(
    const n4m_pp_epo_handle_t* handle,
    n4m_matrix_view_t X,
    const double* d, int64_t d_len,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_domain_adaptation_epo_inverse_transform(
    const n4m_pp_epo_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_domain_adaptation_epo_is_fitted(const n4m_pp_epo_handle_t* handle,
                                           int* out_fitted);

/* ============================================================================
 * 21. Phase 20 — Transfer metrics utility (n4m_transfer_*)
 * ============================================================================
 *
 * `n4m_domain_adaptation_transfer_metrics_compute` computes nine alignment metrics between two
 * datasets (source and target), as defined in
 * nirs4all/analysis/transfer_metrics.py. The struct is plain-data — every
 * field is a `double`. NaN encodes "metric not applicable" (e.g. Grassmann
 * when the two datasets do not share a feature count).
 */
typedef struct n4m_transfer_metrics_t {
    double centroid_distance;
    double cka_similarity;
    double grassmann_distance;
    double rv_coefficient;
    double procrustes_disparity;
    double trustworthiness;
    double spread_distance;
    double evr_source;
    double evr_target;
} n4m_transfer_metrics_t;

N4M_API n4m_status_t n4m_domain_adaptation_transfer_metrics_compute(
    n4m_matrix_view_t X_source,
    n4m_matrix_view_t X_target,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    n4m_transfer_metrics_t* out);

typedef struct n4m_pp_direct_standardization_handle_t
    n4m_pp_direct_standardization_handle_t;
N4M_API n4m_status_t n4m_domain_adaptation_direct_standardization_create(
    n4m_pp_direct_standardization_handle_t** out, int fit_intercept, double ridge);
N4M_API void n4m_domain_adaptation_direct_standardization_destroy(
    n4m_pp_direct_standardization_handle_t* handle);
N4M_API n4m_status_t n4m_domain_adaptation_direct_standardization_fit(
    n4m_pp_direct_standardization_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_domain_adaptation_direct_standardization_transform(
    const n4m_pp_direct_standardization_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_domain_adaptation_direct_standardization_is_fitted(
    const n4m_pp_direct_standardization_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_robust_direct_standardization_handle_t
    n4m_pp_robust_direct_standardization_handle_t;
N4M_API n4m_status_t n4m_domain_adaptation_robust_direct_standardization_create(
    n4m_pp_robust_direct_standardization_handle_t** out, int fit_intercept,
    double ridge, double trim_quantile, int32_t max_iter);
N4M_API void n4m_domain_adaptation_robust_direct_standardization_destroy(
    n4m_pp_robust_direct_standardization_handle_t* handle);
N4M_API n4m_status_t n4m_domain_adaptation_robust_direct_standardization_fit(
    n4m_pp_robust_direct_standardization_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_domain_adaptation_robust_direct_standardization_transform(
    const n4m_pp_robust_direct_standardization_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_domain_adaptation_robust_direct_standardization_is_fitted(
    const n4m_pp_robust_direct_standardization_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_piecewise_direct_standardization_handle_t
    n4m_pp_piecewise_direct_standardization_handle_t;
N4M_API n4m_status_t n4m_domain_adaptation_piecewise_direct_standardization_create(
    n4m_pp_piecewise_direct_standardization_handle_t** out,
    int32_t window_size, int fit_intercept, double ridge);
N4M_API void n4m_domain_adaptation_piecewise_direct_standardization_destroy(
    n4m_pp_piecewise_direct_standardization_handle_t* handle);
N4M_API n4m_status_t n4m_domain_adaptation_piecewise_direct_standardization_fit(
    n4m_pp_piecewise_direct_standardization_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_domain_adaptation_piecewise_direct_standardization_transform(
    const n4m_pp_piecewise_direct_standardization_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_domain_adaptation_piecewise_direct_standardization_is_fitted(
    const n4m_pp_piecewise_direct_standardization_handle_t* handle,
    int* out_fitted);

typedef struct n4m_pp_slope_bias_handle_t n4m_pp_slope_bias_handle_t;
N4M_API n4m_status_t n4m_domain_adaptation_slope_bias_create(
    n4m_pp_slope_bias_handle_t** out);
N4M_API void n4m_domain_adaptation_slope_bias_destroy(n4m_pp_slope_bias_handle_t* handle);
N4M_API n4m_status_t n4m_domain_adaptation_slope_bias_fit(
    n4m_pp_slope_bias_handle_t* handle,
    const double* source, const double* target, int64_t n);
N4M_API n4m_status_t n4m_domain_adaptation_slope_bias_transform(
    const n4m_pp_slope_bias_handle_t* handle,
    const double* y, int64_t n, double* out);
N4M_API n4m_status_t n4m_domain_adaptation_slope_bias_is_fitted(
    const n4m_pp_slope_bias_handle_t* handle, int* out_fitted);

/* Domain-invariant PLS (§13). Penalizes the SIMPLS direction's alignment
 * with the source-vs-target mean difference. The result contains:
 *   "coefficients"        (n_features x n_targets)
 *   "predictions"         (n_samples  x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse_source"  in-sample RMSE on source data
 */
N4M_API n4m_status_t n4m_domain_adaptation_di_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* Y_source,
    const n4m_matrix_view_t* X_target,
    double di_lambda,
    n4m_method_result_t** out_result);

/* PDS — Piecewise Direct Standardization (§13). Fits per-target-column
 * windowed least-squares maps from source instrument X_source to target
 * X_target. The result contains:
 *   "transformation"  (n_features_target x n_features_source)
 *   "predictions"     (n_samples x n_features_target) — X_source @ T^T
 *   scalar "rmse" — frobenius error vs X_target
 *   scalar "window_half_width"
 */
N4M_API n4m_status_t n4m_domain_adaptation_pds_fit(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* X_target,
    int32_t window_half_width,
    n4m_method_result_t** out_result);

/* DS — Direct Standardization (§13). Fits a full pt × ps least-squares
 * map from centered source X to centered target X plus a bias. The
 * result contains:
 *   "transformation" (n_features_source x n_features_target)
 *   "bias"           (1 x n_features_target)
 *   "predictions"    (n_samples x n_features_target)
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_domain_adaptation_ds_fit(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* X_target,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_DOMAIN_ADAPTATION_H */
