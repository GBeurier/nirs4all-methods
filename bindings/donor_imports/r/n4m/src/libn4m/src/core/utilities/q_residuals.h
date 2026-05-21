/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Q-residuals (squared prediction error, SPE) on the PCA-reduced subspace.
 *
 * For a (n_samples x n_features) matrix X centred by column means:
 *   1. Thin SVD Xc = U Σ V^T.
 *   2. Reconstruct on the first k = n_components components:
 *        Xc_hat[i, :] = sum_{j<k} U[i, j] * Σ[j] * V[:, j]^T.
 *   3. Q[i] = sum_d (Xc[i, d] - Xc_hat[i, d])².
 *
 * UCL via the Jackson-Mudholkar approximation. With the residual
 * eigenvalues θ_a = sum_{j>=k} (σ_j² / (n_samples - 1))^a for a = 1, 2, 3
 * and h0 = 1 - (2 θ_1 θ_3) / (3 θ_2²):
 *
 *     ca = sqrt(2) * erf^{-1}(1 - 2 alpha)
 *     UCL = θ_1 * ( ca * sqrt(2 θ_2 h0²) / θ_1
 *                  + 1
 *                  + θ_2 h0 (h0 - 1) / θ_1² )^{1 / h0}.
 *
 * When fewer than (k + 1) singular values are non-zero (rank-deficient
 * matrix or `n_components >= rank`) the UCL is reported as zero.
 */
#ifndef N4M_CORE_UTILITIES_Q_RESIDUALS_H
#define N4M_CORE_UTILITIES_Q_RESIDUALS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

n4m_status_t n4m_util_q_residuals_impl(const double* X,
                                        int64_t rows, int64_t cols,
                                        int32_t n_components,
                                        double alpha,
                                        double* q_per_sample,
                                        double* ucl_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_UTILITIES_Q_RESIDUALS_H */
