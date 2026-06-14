/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/lowlevel.h — lowlevel role header (ABI 2.0). */
#ifndef N4M_LOWLEVEL_H
#define N4M_LOWLEVEL_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Raw row-additive sufficient statistics for moment-based linear screens.
 *
 * n4m_lowlevel_moments_compute accumulates all rows; n4m_lowlevel_moments_subset_compute
 * accumulates the rows named by `row_indices`; n4m_lowlevel_moments_subtract computes
 * lhs - rhs from two compatible moment results. The subtraction path keeps the
 * raw additive buffers and recomputes centered moments from the remaining
 * train-row sums, which is the exact fold-subtraction primitive needed by PLS
 * / Ridge screens.
 *
 * Result double matrices:
 *   "x_sum" (1 x p), "y_sum" (1 x q)
 *   "xtx" (p x p), "xty" (p x q), "yty" (q x q)       raw moments
 *   "x_mean" (1 x p), "y_mean" (1 x q)
 *   "cxx" (p x p), "cxy" (p x q), "cyy" (q x q)       centered moments
 * Scalars:
 *   "n_samples", "n_features", "n_targets"
 *
 * Inputs may be F64 or F32 matrix views. Internal accumulation and returned
 * buffers are always F64. In CUDA builds the Gram products go through the
 * existing compile-time linalg dispatch; this is not yet the fused CUDA
 * sweep/grinder API.
 */
N4M_API n4m_status_t n4m_lowlevel_moments_compute(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

N4M_API n4m_status_t n4m_lowlevel_moments_subset_compute(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int64_t* row_indices,
    int64_t n_indices,
    n4m_method_result_t** out_result);

N4M_API n4m_status_t n4m_lowlevel_moments_subtract(
    n4m_context_t* ctx,
    const n4m_method_result_t* lhs,
    const n4m_method_result_t* rhs,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_LOWLEVEL_H */
