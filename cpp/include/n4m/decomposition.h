/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/decomposition.h — decomposition role header (ABI 2.0). */
#ifndef N4M_DECOMPOSITION_H
#define N4M_DECOMPOSITION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- FlexiblePCA ------------------------------------------------ */
typedef struct n4m_pp_flex_pca_handle_t n4m_pp_flex_pca_handle_t;
N4M_API n4m_status_t n4m_decomposition_flexible_pca_create(n4m_pp_flex_pca_handle_t** out,
                                             double n_components);
N4M_API void         n4m_decomposition_flexible_pca_destroy(n4m_pp_flex_pca_handle_t* handle);
N4M_API n4m_status_t n4m_decomposition_flexible_pca_fit(n4m_pp_flex_pca_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_decomposition_flexible_pca_transform(
    const n4m_pp_flex_pca_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_decomposition_flexible_pca_is_fitted(
    const n4m_pp_flex_pca_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_decomposition_flexible_pca_output_cols(
    const n4m_pp_flex_pca_handle_t* handle, int64_t* out_cols);

/* ---------- FlexibleSVD ------------------------------------------------ */
typedef struct n4m_pp_flex_svd_handle_t n4m_pp_flex_svd_handle_t;
N4M_API n4m_status_t n4m_decomposition_flexible_svd_create(n4m_pp_flex_svd_handle_t** out,
                                             double n_components);
N4M_API void         n4m_decomposition_flexible_svd_destroy(n4m_pp_flex_svd_handle_t* handle);
N4M_API n4m_status_t n4m_decomposition_flexible_svd_fit(n4m_pp_flex_svd_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_decomposition_flexible_svd_transform(
    const n4m_pp_flex_svd_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_decomposition_flexible_svd_is_fitted(
    const n4m_pp_flex_svd_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_decomposition_flexible_svd_output_cols(
    const n4m_pp_flex_svd_handle_t* handle, int64_t* out_cols);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_DECOMPOSITION_H */
