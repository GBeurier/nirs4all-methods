/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/outlier_detection.h — outlier_detection role header (ABI 2.0). */
#ifndef N4M_OUTLIER_DETECTION_H
#define N4M_OUTLIER_DETECTION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Shared filter result type --------------------------------- */
typedef struct n4m_filter_stats_t {
    int64_t n_samples;        /* total samples seen by the filter         */
    int64_t n_kept;           /* count of mask[i] == 1                    */
    int64_t n_excluded;       /* n_samples - n_kept                       */
    double  exclusion_rate;   /* n_excluded / n_samples; 0.0 when n == 0  */
} n4m_filter_stats_t;

/* ---------- Y-outlier method enum ------------------------------------- */
typedef enum n4m_y_outlier_method_t {
    N4M_Y_OUTLIER_IQR        = 0,
    N4M_Y_OUTLIER_ZSCORE     = 1,
    N4M_Y_OUTLIER_PERCENTILE = 2,
    N4M_Y_OUTLIER_MAD        = 3
} n4m_y_outlier_method_t;

/* ---------- YOutlierFilter -------------------------------------------- *
 *
 * The filter follows the classical sklearn fit/apply split:
 *   - `_create` constructs an unfitted handle from the method + thresholds.
 *   - `_fit(handle, y, n)` learns the per-method bounds from the training
 *     y vector (one shot; idempotent — recomputes on every call).
 *   - `_apply(handle, y, n, mask, stats)` applies the previously learned
 *     bounds to a (possibly different) y vector, returning the keep-mask
 *     and statistics. The handle must be fitted; returns
 *     `N4M_ERR_NOT_FITTED` when called on an unfitted handle.
 *   - `_is_fitted(handle, out)` writes 1 when `_fit` has been called at
 *     least once on this handle, 0 otherwise.
 */
typedef struct n4m_filter_y_outlier_handle_t n4m_filter_y_outlier_handle_t;
N4M_API n4m_status_t n4m_outlier_detection_y_outlier_create(
    n4m_filter_y_outlier_handle_t** out,
    int32_t method,
    double threshold, double lower_pct, double upper_pct);
N4M_API void         n4m_outlier_detection_y_outlier_destroy(
    n4m_filter_y_outlier_handle_t* handle);
N4M_API n4m_status_t n4m_outlier_detection_y_outlier_fit(
    n4m_filter_y_outlier_handle_t* handle,
    const double* y, int64_t n);
N4M_API n4m_status_t n4m_outlier_detection_y_outlier_apply(
    const n4m_filter_y_outlier_handle_t* handle,
    const double* y, int64_t n,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);
N4M_API n4m_status_t n4m_outlier_detection_y_outlier_is_fitted(
    const n4m_filter_y_outlier_handle_t* handle, int* out);

/* ---------- HighLeverageFilter (stateful) ----------------------------- */
typedef struct n4m_filter_leverage_handle_t n4m_filter_leverage_handle_t;
N4M_API n4m_status_t n4m_outlier_detection_high_leverage_create(
    n4m_filter_leverage_handle_t** out,
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute_threshold,
    double  absolute_threshold,
    int32_t n_components,
    int     center);
N4M_API void         n4m_outlier_detection_high_leverage_destroy(
    n4m_filter_leverage_handle_t* handle);
N4M_API n4m_status_t n4m_outlier_detection_high_leverage_fit(
    n4m_filter_leverage_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_outlier_detection_high_leverage_is_fitted(
    const n4m_filter_leverage_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_outlier_detection_high_leverage_apply(
    const n4m_filter_leverage_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);
N4M_API double       n4m_outlier_detection_high_leverage_threshold(
    const n4m_filter_leverage_handle_t* handle);

/* ---------- SpectralQualityFilter (stateless) ------------------------- */
typedef struct n4m_filter_quality_handle_t n4m_filter_quality_handle_t;
N4M_API n4m_status_t n4m_outlier_detection_spectral_quality_create(
    n4m_filter_quality_handle_t** out,
    double max_nan_ratio, double max_zero_ratio,
    double min_variance,
    int use_max, double max_value,
    int use_min, double min_value,
    int check_inf);
N4M_API void         n4m_outlier_detection_spectral_quality_destroy(
    n4m_filter_quality_handle_t* handle);
N4M_API n4m_status_t n4m_outlier_detection_spectral_quality_apply(
    const n4m_filter_quality_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);

/* ---------- CompositeFilter ------------------------------------------- */
typedef enum n4m_composite_mode_t {
    N4M_COMPOSITE_ANY = 0,    /* exclude if ANY sub-filter excludes */
    N4M_COMPOSITE_ALL = 1     /* exclude only if ALL sub-filters exclude */
} n4m_composite_mode_t;

typedef struct n4m_filter_composite_handle_t n4m_filter_composite_handle_t;
N4M_API n4m_status_t n4m_outlier_detection_composite_create(
    n4m_filter_composite_handle_t** out, int32_t mode);
N4M_API void         n4m_outlier_detection_composite_destroy(
    n4m_filter_composite_handle_t* handle);
N4M_API n4m_status_t n4m_outlier_detection_composite_add_leverage(
    n4m_filter_composite_handle_t* handle,
    n4m_filter_leverage_handle_t* sub);
N4M_API n4m_status_t n4m_outlier_detection_composite_add_quality(
    n4m_filter_composite_handle_t* handle,
    n4m_filter_quality_handle_t* sub);
N4M_API n4m_status_t n4m_outlier_detection_composite_apply(
    const n4m_filter_composite_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);

/* ---------- 20.3 Multivariate outlier statistics ---------------------- */
/* Hotelling T² on the leading PCA directions of the column-centred X
 * (corrected two-pass mean: a constant column centres to exact zeros).
 * Only resolved directions are used: sigma_j > 16 * max(n,p) * DBL_EPSILON
 * * sigma_1 (the SVD's absolute accuracy). With r <= n_components such
 * directions, T² sums over the leading r and the UCL uses r degrees of
 * freedom: r (n-1)(n+1) / (n (n-r)) * F^{-1}(1 - alpha; r, n - r). A
 * rank-deficient X (e.g. a column that is an exact combination of others)
 * therefore gets r < n_components; r = 0 (constant X) gives T² = 0 and
 * UCL = 0. T²_i = (n-1) * sum_{j<r} U_ij^2 (equal to sum t_ij^2 / lambda_j,
 * without forming sigma_j^2), exactly invariant under power-of-two scaling
 * of X. The F quantile is found on the upper tail with an unbounded
 * bracket; a UCL that is not a finite normal double (extreme alpha only) is
 * N4M_ERR_NUMERICAL_FAILURE. Requires rows >= cols. */
N4M_API n4m_status_t n4m_outlier_detection_hotelling_t2(n4m_matrix_view_t X,
                                            int32_t n_components,
                                            double alpha,
                                            double* t2_per_sample,
                                            int64_t n_samples,
                                            double* ucl_out);
/* ABI 2.20. n4m_outlier_detection_hotelling_t2 that also publishes the rank
 * actually used (rank_used, one element, must not be NULL). */
N4M_API n4m_status_t n4m_outlier_detection_hotelling_t2_resolved(
    n4m_matrix_view_t X, int32_t n_components, double alpha, double* t2_per_sample,
    int64_t n_samples, double* ucl_out, int32_t* rank_used);
/* Q residuals (squared prediction error) of the column-centred X (same
 * corrected two-pass centring as Hotelling T²) outside its leading
 * n_components PCA directions, and the Jackson-Mudholkar UCL from the
 * residual eigenvalue moments theta_1..3 and the upper normal quantile at
 * alpha. Requires rows >= cols.
 *  - Computed at one power-of-two scale (max |Xc| in [0.5, 1)) and rescaled:
 *    the moments cannot overflow or underflow, and scaling X by 2^s scales
 *    every Q and the UCL by exactly 4^s.
 *  - Unresolved residual subspace: when no residual singular value is above
 *    16 * max(n,p) * DBL_EPSILON * sigma_1 (n_components = cols, constant X,
 *    or numerical rank <= n_components), the reconstructed residual is
 *    rounding noise and Q = 0 and UCL = 0 exactly. UCL = 0 occurs only in
 *    that state; otherwise the UCL is a positive normal double.
 *  - Accuracy is absolute: Q_i is computed from the explicit residual and is
 *    rounding noise at or below (16 * max(n,p) * DBL_EPSILON * sigma_1)^2.
 *    The quantile approximation has a relative error of about 1.15e-9.
 *  - N4M_ERR_NUMERICAL_FAILURE (never 0, Inf or NaN with N4M_OK) when a Q
 *    above that noise level or the UCL is not a finite normal double, or
 *    when the Jackson-Mudholkar base is not positive (possible only for
 *    alpha > 0.5, where the approximation has no value). */
N4M_API n4m_status_t n4m_outlier_detection_q_residuals(n4m_matrix_view_t X,
                                           int32_t n_components,
                                           double alpha,
                                           double* q_per_sample,
                                           int64_t n_samples,
                                           double* ucl_out);

typedef enum n4m_filter_x_outlier_method_t {
    N4M_X_OUTLIER_MAHALANOBIS         = 0,
    N4M_X_OUTLIER_ROBUST_MAHALANOBIS  = 1,
    N4M_X_OUTLIER_PCA_RESIDUAL        = 2,
    N4M_X_OUTLIER_PCA_LEVERAGE        = 3,
    N4M_X_OUTLIER_ISOLATION_FOREST    = 4,
    N4M_X_OUTLIER_LOF                 = 5
} n4m_filter_x_outlier_method_t;

typedef struct n4m_filter_x_outlier_handle_t n4m_filter_x_outlier_handle_t;
N4M_API n4m_status_t n4m_outlier_detection_x_outlier_create(
    n4m_filter_x_outlier_handle_t** out,
    int32_t  method,
    int      use_threshold,
    double   threshold,
    int32_t  n_components,
    double   contamination,
    uint64_t seed,
    int32_t  n_estimators,
    int64_t  max_samples);
N4M_API void n4m_outlier_detection_x_outlier_destroy(n4m_filter_x_outlier_handle_t* handle);
N4M_API n4m_status_t n4m_outlier_detection_x_outlier_fit(
    n4m_filter_x_outlier_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_outlier_detection_x_outlier_is_fitted(
    const n4m_filter_x_outlier_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_outlier_detection_x_outlier_apply(
    const n4m_filter_x_outlier_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_OUTLIER_DETECTION_H */
