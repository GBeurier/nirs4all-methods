/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/wavelet.h — transform.wavelet methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_WAVELET_H
#define N4M_TRANSFORM_WAVELET_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_pp_wavelet_family_t {
    N4M_PP_WAVELET_HAAR  = 0,
    N4M_PP_WAVELET_DB4   = 1,
    N4M_PP_WAVELET_SYM4  = 2,
    N4M_PP_WAVELET_COIF1 = 3
} n4m_pp_wavelet_family_t;

typedef enum n4m_pp_wavelet_boundary_t {
    N4M_PP_WAVELET_BOUNDARY_PERIODIZATION = 0,
    N4M_PP_WAVELET_BOUNDARY_SYMMETRIC     = 1,
    N4M_PP_WAVELET_BOUNDARY_ZERO          = 2
} n4m_pp_wavelet_boundary_t;

typedef enum n4m_pp_wavelet_threshold_t {
    N4M_PP_WAVELET_THRESHOLD_SOFT = 0,
    N4M_PP_WAVELET_THRESHOLD_HARD = 1
} n4m_pp_wavelet_threshold_t;

typedef enum n4m_pp_wavelet_noise_t {
    N4M_PP_WAVELET_NOISE_MEDIAN = 0,
    N4M_PP_WAVELET_NOISE_STD    = 1
} n4m_pp_wavelet_noise_t;

typedef enum n4m_pp_wavelet_features_entropy_t {
    N4M_PP_WAVELET_FEATURES_ENTROPY_ENERGY    = 0,
    N4M_PP_WAVELET_FEATURES_ENTROPY_HISTOGRAM = 1
} n4m_pp_wavelet_features_entropy_t;

typedef struct n4m_pp_wavelet_handle_t          n4m_pp_wavelet_handle_t;
typedef struct n4m_pp_haar_handle_t             n4m_pp_haar_handle_t;
typedef struct n4m_pp_wavelet_denoise_handle_t  n4m_pp_wavelet_denoise_handle_t;
typedef struct n4m_pp_wavelet_features_handle_t n4m_pp_wavelet_features_handle_t;
typedef struct n4m_pp_wavelet_pca_handle_t      n4m_pp_wavelet_pca_handle_t;
typedef struct n4m_pp_wavelet_svd_handle_t      n4m_pp_wavelet_svd_handle_t;

N4M_API n4m_status_t n4m_transform_wavelet_create(n4m_pp_wavelet_handle_t** out,
                                            n4m_pp_wavelet_family_t   family,
                                            n4m_pp_wavelet_boundary_t mode);
N4M_API void         n4m_transform_wavelet_destroy(n4m_pp_wavelet_handle_t* handle);
N4M_API n4m_status_t n4m_transform_wavelet_output_cols(
    const n4m_pp_wavelet_handle_t* handle,
    int64_t input_cols, int64_t* out_cols);
N4M_API n4m_status_t n4m_transform_wavelet_transform(
    const n4m_pp_wavelet_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_transform_haar_create(n4m_pp_haar_handle_t** out);
N4M_API void         n4m_transform_haar_destroy(n4m_pp_haar_handle_t* handle);
N4M_API n4m_status_t n4m_transform_haar_output_cols(int64_t input_cols,
                                              int64_t* out_cols);
N4M_API n4m_status_t n4m_transform_haar_transform(const n4m_pp_haar_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_transform_wavelet_denoise_create(
    n4m_pp_wavelet_denoise_handle_t** out,
    n4m_pp_wavelet_family_t    family,
    n4m_pp_wavelet_boundary_t  mode,
    int32_t                    level,
    n4m_pp_wavelet_threshold_t threshold_mode,
    n4m_pp_wavelet_noise_t     noise_estimator);
N4M_API void         n4m_transform_wavelet_denoise_destroy(
    n4m_pp_wavelet_denoise_handle_t* handle);
N4M_API n4m_status_t n4m_transform_wavelet_denoise_transform(
    const n4m_pp_wavelet_denoise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_transform_wavelet_features_create(
    n4m_pp_wavelet_features_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level);
N4M_API n4m_status_t n4m_transform_wavelet_features_create_ex(
    n4m_pp_wavelet_features_handle_t** out,
    n4m_pp_wavelet_family_t            family,
    n4m_pp_wavelet_boundary_t          mode,
    int32_t                            max_level,
    n4m_pp_wavelet_features_entropy_t  entropy_mode);
N4M_API void         n4m_transform_wavelet_features_destroy(
    n4m_pp_wavelet_features_handle_t* handle);
N4M_API n4m_status_t n4m_transform_wavelet_features_output_cols(
    const n4m_pp_wavelet_features_handle_t* handle,
    int64_t input_cols, int64_t* out_cols);
N4M_API n4m_status_t n4m_transform_wavelet_features_transform(
    const n4m_pp_wavelet_features_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_transform_wavelet_pca_create(
    n4m_pp_wavelet_pca_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components);
N4M_API void         n4m_transform_wavelet_pca_destroy(
    n4m_pp_wavelet_pca_handle_t* handle);
N4M_API n4m_status_t n4m_transform_wavelet_pca_fit(
    n4m_pp_wavelet_pca_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_wavelet_pca_transform(
    const n4m_pp_wavelet_pca_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_wavelet_pca_is_fitted(
    const n4m_pp_wavelet_pca_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_transform_wavelet_pca_output_cols(
    const n4m_pp_wavelet_pca_handle_t* handle, int64_t* out_cols);

N4M_API n4m_status_t n4m_transform_wavelet_svd_create(
    n4m_pp_wavelet_svd_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components);
N4M_API void         n4m_transform_wavelet_svd_destroy(
    n4m_pp_wavelet_svd_handle_t* handle);
N4M_API n4m_status_t n4m_transform_wavelet_svd_fit(
    n4m_pp_wavelet_svd_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_wavelet_svd_transform(
    const n4m_pp_wavelet_svd_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_wavelet_svd_is_fitted(
    const n4m_pp_wavelet_svd_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_transform_wavelet_svd_output_cols(
    const n4m_pp_wavelet_svd_handle_t* handle, int64_t* out_cols);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_WAVELET_H */
