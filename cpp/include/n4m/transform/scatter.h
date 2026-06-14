/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/scatter.h — transform.scatter methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_SCATTER_H
#define N4M_TRANSFORM_SCATTER_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- SNV (Standard Normal Variate) -------------------------------- */
typedef struct n4m_pp_snv_handle_t n4m_pp_snv_handle_t;
N4M_API n4m_status_t n4m_transform_snv_create(n4m_pp_snv_handle_t** out,
                                        int with_mean, int with_std, int ddof);
N4M_API void         n4m_transform_snv_destroy(n4m_pp_snv_handle_t* handle);
N4M_API n4m_status_t n4m_transform_snv_transform(const n4m_pp_snv_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);

/* ---------- LocalSNV (sliding-window SNV) -------------------------------- */
typedef struct n4m_pp_lsnv_handle_t n4m_pp_lsnv_handle_t;
typedef enum n4m_pp_lsnv_pad_mode_t {
    N4M_PP_LSNV_PAD_REFLECT  = 0,
    N4M_PP_LSNV_PAD_EDGE     = 1,
    N4M_PP_LSNV_PAD_CONSTANT = 2
} n4m_pp_lsnv_pad_mode_t;
/* `window` must be an odd integer >= 3. */
N4M_API n4m_status_t n4m_transform_local_snv_create(n4m_pp_lsnv_handle_t** out,
                                         int32_t window, int32_t pad_mode,
                                         double constant_value);
N4M_API void         n4m_transform_local_snv_destroy(n4m_pp_lsnv_handle_t* handle);
N4M_API n4m_status_t n4m_transform_local_snv_transform(const n4m_pp_lsnv_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- RobustSNV (median + k * MAD) --------------------------------- */
typedef struct n4m_pp_rnv_handle_t n4m_pp_rnv_handle_t;
N4M_API n4m_status_t n4m_transform_robust_snv_create(n4m_pp_rnv_handle_t** out,
                                        int with_center, int with_scale,
                                        double k);
N4M_API void         n4m_transform_robust_snv_destroy(n4m_pp_rnv_handle_t* handle);
N4M_API n4m_status_t n4m_transform_robust_snv_transform(const n4m_pp_rnv_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);

/* ---------- AreaNormalization -------------------------------------------- */
typedef struct n4m_pp_area_handle_t n4m_pp_area_handle_t;
typedef enum n4m_pp_area_method_t {
    N4M_PP_AREA_SUM     = 0,
    N4M_PP_AREA_ABS_SUM = 1,
    N4M_PP_AREA_TRAPZ   = 2
} n4m_pp_area_method_t;
N4M_API n4m_status_t n4m_transform_area_normalization_create(n4m_pp_area_handle_t** out,
                                         int32_t method);
N4M_API void         n4m_transform_area_normalization_destroy(n4m_pp_area_handle_t* handle);
N4M_API n4m_status_t n4m_transform_area_normalization_transform(const n4m_pp_area_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- MSC (Multiplicative Scatter Correction) ---------------------- */
/*
 * Conventional row-wise scatter correction calibrated against the mean
 * spectrum of the training matrix:
 *   reference[j] = mean(X[:, j])              # length n_features
 *   (offset[i], slope[i]) = ols(X[i, :] ~ 1 + reference)
 *   X'[i, j] = (X[i, j] - offset[i]) / slope[i]
 * Inverse uses the row coefficients stored by the last `_transform` call on
 * the same handle:
 *   X[i, j] = X'[i, j] * slope[i] + offset[i]
 *
 * Matches `prospectr::msc` / `pls::msc` for the default reference-spectrum
 * contract. The historical nirs4all column-regression variant is intentionally
 * not the validation target for this ABI.
 *
 * `_fit` requires at least one row and at least two columns.
 */
typedef struct n4m_pp_msc_handle_t n4m_pp_msc_handle_t;
N4M_API n4m_status_t n4m_transform_msc_create(n4m_pp_msc_handle_t** out);
N4M_API void         n4m_transform_msc_destroy(n4m_pp_msc_handle_t* handle);
N4M_API n4m_status_t n4m_transform_msc_fit(n4m_pp_msc_handle_t* handle,
                                     n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_msc_transform(n4m_pp_msc_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_msc_inverse_transform(
    const n4m_pp_msc_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_msc_is_fitted(const n4m_pp_msc_handle_t* handle,
                                           int* out_fitted);

/* ---------- EMSC (Extended Multiplicative Scatter Correction) ------------ */
/*
 * Per-sample scatter correction with polynomial wavelength terms. For each
 * row x_i, solve the least-squares system
 *     x_i ≈ c[0] * ref + sum_{d=1..degree} c[d] * wavelengths^d
 * then subtract the polynomial portion and divide by the reference
 * coefficient:
 *     x'_i = (x_i - sum_{d=1..degree} c[d] * wavelengths^d) / c[0]
 *
 * `ref` is the per-column mean of the training matrix (length n_features).
 * `wavelengths` is the integer index 0, 1, …, n_features - 1 (matching the
 * nirs4all convention; rescaling the axis to [-1, 1] would change the
 * polynomial coefficients but not the result after subtraction-and-divide).
 *
 * No inverse transform: the polynomial subtraction is per-row data-driven
 * and discards the row-specific coefficients, so inversion would require
 * re-storing them.
 *
 * `_create` requires `degree >= 1`.
 * `_fit`    requires `X.rows >= 1` and `X.cols >= degree + 2` (basis
 *           dimensionality + reference).
 *
 * Matches `nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`
 * with `scale=False`.
 */
typedef struct n4m_pp_emsc_handle_t n4m_pp_emsc_handle_t;
N4M_API n4m_status_t n4m_transform_emsc_create(n4m_pp_emsc_handle_t** out,
                                         int32_t degree);
N4M_API void         n4m_transform_emsc_destroy(n4m_pp_emsc_handle_t* handle);
N4M_API n4m_status_t n4m_transform_emsc_fit(n4m_pp_emsc_handle_t* handle,
                                      n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_emsc_transform(const n4m_pp_emsc_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_emsc_is_fitted(const n4m_pp_emsc_handle_t* handle,
                                            int* out_fitted);

typedef struct n4m_pp_local_centering_handle_t n4m_pp_local_centering_handle_t;
N4M_API n4m_status_t n4m_transform_local_centering_create(
    n4m_pp_local_centering_handle_t** out);
N4M_API void n4m_transform_local_centering_destroy(
    n4m_pp_local_centering_handle_t* handle);
N4M_API n4m_status_t n4m_transform_local_centering_fit(
    n4m_pp_local_centering_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_transform_local_centering_transform(
    const n4m_pp_local_centering_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_local_centering_is_fitted(
    const n4m_pp_local_centering_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_weighted_snv_handle_t n4m_pp_weighted_snv_handle_t;
N4M_API n4m_status_t n4m_transform_weighted_snv_create(
    n4m_pp_weighted_snv_handle_t** out, const double* weights,
    int64_t n_weights, int32_t ddof, double eps);
N4M_API void n4m_transform_weighted_snv_destroy(n4m_pp_weighted_snv_handle_t* handle);
N4M_API n4m_status_t n4m_transform_weighted_snv_fit(
    n4m_pp_weighted_snv_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_weighted_snv_transform(
    const n4m_pp_weighted_snv_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_weighted_snv_is_fitted(
    const n4m_pp_weighted_snv_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_vsn_handle_t n4m_pp_vsn_handle_t;
N4M_API n4m_status_t n4m_transform_vsn_create(n4m_pp_vsn_handle_t** out, double eps);
N4M_API void n4m_transform_vsn_destroy(n4m_pp_vsn_handle_t* handle);
N4M_API n4m_status_t n4m_transform_vsn_fit(n4m_pp_vsn_handle_t* handle,
                                    n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_vsn_transform(
    const n4m_pp_vsn_handle_t* handle, n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_vsn_is_fitted(
    const n4m_pp_vsn_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_piecewise_snv_handle_t n4m_pp_piecewise_snv_handle_t;
N4M_API n4m_status_t n4m_transform_piecewise_snv_create(
    n4m_pp_piecewise_snv_handle_t** out, int32_t window,
    int32_t ddof, double eps);
N4M_API void n4m_transform_piecewise_snv_destroy(n4m_pp_piecewise_snv_handle_t* handle);
N4M_API n4m_status_t n4m_transform_piecewise_snv_fit(
    n4m_pp_piecewise_snv_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_piecewise_snv_transform(
    const n4m_pp_piecewise_snv_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_piecewise_snv_is_fitted(
    const n4m_pp_piecewise_snv_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_piecewise_msc_handle_t n4m_pp_piecewise_msc_handle_t;
typedef struct n4m_pp_localized_msc_handle_t n4m_pp_localized_msc_handle_t;
N4M_API n4m_status_t n4m_transform_piecewise_msc_create(
    n4m_pp_piecewise_msc_handle_t** out, const double* reference,
    int64_t n_reference, int32_t window, double eps);
N4M_API void n4m_transform_piecewise_msc_destroy(
    n4m_pp_piecewise_msc_handle_t* handle);
N4M_API n4m_status_t n4m_transform_piecewise_msc_fit(
    n4m_pp_piecewise_msc_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_piecewise_msc_transform(
    const n4m_pp_piecewise_msc_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_piecewise_msc_is_fitted(
    const n4m_pp_piecewise_msc_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_transform_localized_msc_create(
    n4m_pp_localized_msc_handle_t** out, const double* reference,
    int64_t n_reference, int32_t window, double eps);
N4M_API void n4m_transform_localized_msc_destroy(
    n4m_pp_localized_msc_handle_t* handle);
N4M_API n4m_status_t n4m_transform_localized_msc_fit(
    n4m_pp_localized_msc_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_localized_msc_transform(
    const n4m_pp_localized_msc_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_localized_msc_is_fitted(
    const n4m_pp_localized_msc_handle_t* handle, int* out_fitted);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_SCATTER_H */
