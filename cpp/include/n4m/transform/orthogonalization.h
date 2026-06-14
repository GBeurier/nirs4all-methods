/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/orthogonalization.h — transform.orthogonalization methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_ORTHOGONALIZATION_H
#define N4M_TRANSFORM_ORTHOGONALIZATION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Orthogonal Signal Correction (Wold 1998) ------------------- */
typedef struct n4m_pp_osc_handle_t n4m_pp_osc_handle_t;
N4M_API n4m_status_t n4m_transform_osc_create(n4m_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
N4M_API void         n4m_transform_osc_destroy(n4m_pp_osc_handle_t* handle);
N4M_API n4m_status_t n4m_transform_osc_fit(n4m_pp_osc_handle_t* handle,
                                     n4m_matrix_view_t X,
                                     const double* y, int64_t y_len);
N4M_API n4m_status_t n4m_transform_osc_transform(const n4m_pp_osc_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_osc_inverse_transform(
    const n4m_pp_osc_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_osc_is_fitted(const n4m_pp_osc_handle_t* handle,
                                           int* out_fitted);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_ORTHOGONALIZATION_H */
