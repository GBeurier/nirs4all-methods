/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/specialized.h — transform.specialized methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_SPECIALIZED_H
#define N4M_TRANSFORM_SPECIALIZED_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 22. Phase 21 — FCK static transformer (n4m_pp_fck_static_*)
 * ============================================================================
 *
 * FCKStaticTransformer applies a precomputed bank of L = n_orders * n_scales
 * fractional convolutional kernels to each spectrum row. For an input of
 * shape (n, p) the output has shape (n, L * p) with the L convolved signals
 * horizontally concatenated in (sample, kernel, feature) order.
 *
 * Each kernel of length `kernel_size` is the standard Gaussian × fractional
 * spatial-derivative recipe (see docs/algorithms/fck_static.md); the bank is
 * the Cartesian product of `filter_orders` × `filter_scales` with `alpha`
 * varying slowest.
 */
typedef struct n4m_pp_fck_static_handle_t n4m_pp_fck_static_handle_t;
N4M_API n4m_status_t n4m_transform_fck_static_create(
    n4m_pp_fck_static_handle_t** out,
    int32_t kernel_size,
    const double* filter_orders, int32_t n_orders,
    const double* filter_scales, int32_t n_scales);
N4M_API void n4m_transform_fck_static_destroy(n4m_pp_fck_static_handle_t* handle);
N4M_API n4m_status_t n4m_transform_fck_static_transform(
    const n4m_pp_fck_static_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_fck_static_output_cols(int32_t n_kernels,
                                                    int32_t n_features,
                                                    int32_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_SPECIALIZED_H */
