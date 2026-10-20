/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Hotelling T² statistic on the PCA-reduced score space.
 *
 * For a (n_samples x n_features) matrix X:
 *   1. Centre by column means (corrected two-pass: constant columns centre
 *      to exact zeros).
 *   2. Compute the thin SVD of the centred matrix: Xc = U Σ V^T.
 *   3. Take the first k = n_components columns of T = U Σ as the scores.
 *   4. Eigenvalues of the sample covariance are λ_j = σ_j² / (n_samples - 1).
 *   5. T²[i] = sum_{j<k} (T[i, j]² / λ_j).
 *
 * Only resolved directions are used: sigma_j > 16 max(n,p) eps sigma_1
 * (the SVD's absolute accuracy, core/common/svd.h). With r <= k such
 * directions (rank_used, may be NULL), T² sums over j < r and the UCL is
 * the standard F-distribution bound with r degrees of freedom:
 *     UCL = r * (n - 1) * (n + 1) / (n * (n - r)) * F^{-1}(1 - alpha; r, n - r)
 * r = 0 (constant X) gives T² = 0 and UCL = 0.
 * Implemented internally via a numerical inverse of the regularised
 * incomplete beta function.
 */
#ifndef N4M_CORE_UTILITIES_HOTELLING_T2_H
#define N4M_CORE_UTILITIES_HOTELLING_T2_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

n4m_status_t n4m_util_hotelling_t2_impl(const double* X,
                                         int64_t rows, int64_t cols,
                                         int32_t n_components,
                                         double alpha,
                                         double* t2_per_sample,
                                         double* ucl_out,
                                         int32_t* rank_used);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_UTILITIES_HOTELLING_T2_H */
