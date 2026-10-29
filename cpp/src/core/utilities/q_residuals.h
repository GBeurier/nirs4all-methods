/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Q-residuals (squared prediction error, SPE) on the PCA-reduced subspace.
 *
 * For a (n_samples x n_features) matrix X (tall or wide):
 *   1. Centre the columns (corrected two-pass mean, column_centring.h) at one
 *      power-of-two scale 2^e: Xc = (X - mean) 2^-e, max|Xc| in [0.5, 1).
 *   2. Thin SVD Xc = U Σ V^T.
 *   3. Reconstruct on the first k = n_components components:
 *        Xc_hat[i, :] = sum_{j<k} U[i, j] * Σ[j] * V[:, j]^T.
 *   4. Q[i] = 4^e sum_d (Xc[i, d] - Xc_hat[i, d])².
 *
 * UCL via the Jackson-Mudholkar approximation, in the scaled space and
 * multiplied by 4^e. With θ_a = sum_{j>=k} (σ_j² / (n_samples - 1))^a,
 * h0 = 1 - (2 θ_1 θ_3) / (3 θ_2²) (h0 <= 1/3, of either sign) and ca the
 * upper normal quantile at alpha:
 *
 *     UCL = θ_1 * ( ca * h0 * sqrt(2 θ_2) / θ_1
 *                  + 1
 *                  + θ_2 h0 (h0 - 1) / θ_1² )^{1 / h0},
 *
 * with the SIGNED h0: (Q / θ_1)^h0 decreases with Q when h0 < 0, so the upper
 * quantile of Q is the lower quantile of the approximating normal
 * (Nomikos & MacGregor 1995, appendix B). Evaluated as
 * θ_1 exp(log1p(h0 C) / h0), C = ca sqrt(2 θ_2) / θ_1 + (h0 - 1) θ_2 / θ_1²,
 * continuous through h0 = 0 (limit θ_1 exp(C)).
 *
 * When no residual singular value is resolved (σ_k <= 16 max(n,p) eps σ_1,
 * or k = n_features), Q = 0 and UCL = 0 exactly. Unrepresentable results
 * are N4M_ERR_NUMERICAL_FAILURE, and so is a non-positive bracket (the
 * approximation then has no alpha-quantile): it is always positive when
 * h0 ca >= 0, or when h0 < 0 and alpha >= 1 - Phi(sqrt 2) ~ 0.0786, and can
 * be non-positive only for h0 > 0 with alpha > 0.5 or h0 < 0 with
 * alpha < 0.0786; see n4m/outlier_detection.h.
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
