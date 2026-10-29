/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Hotelling T² statistic on the PCA-reduced score space.
 *
 * For a (n_samples x n_features) matrix X (tall or wide):
 *   1. Centre by column means (corrected two-pass, column_centring.h:
 *      constant columns centre to exact zeros) at one power-of-two scale.
 *   2. Compute the thin SVD of the centred matrix: Xc = U Σ V^T.
 *   3. Take the first k = n_components columns of T = U Σ as the scores.
 *   4. Eigenvalues of the sample covariance are λ_j = σ_j² / (n_samples - 1).
 *   5. T²[i] = sum_{j<k} (T[i, j]² / λ_j) = (n_samples - 1) sum_{j<k} U[i, j]²,
 *      evaluated in the second form (no σ², exactly scale-invariant).
 *
 * Only resolved directions are used: sigma_j > 16 max(n,p) eps sigma_1
 * (the SVD's absolute accuracy, core/common/svd.h). With r <= k such
 * directions (rank_used, may be NULL), T² sums over j < r and the UCL is
 * the standard F-distribution bound with r degrees of freedom:
 *     UCL = r * (n - 1) * (n + 1) / (n * (n - r)) * F^{-1}(1 - alpha; r, n - r)
 * r = 0 (constant X) gives T² = 0 and UCL = 0.
 * The UCL is solved directly: Y = UCL / (kappa + UCL) ~ Beta(r/2, (n-r)/2)
 * with kappa = (n^2 - 1) / n. Its small tail (upper for alpha <= 0.5, lower
 * otherwise, 1 - alpha being exact there) is evaluated in logarithms from
 * UCL / kappa (no beta argument through a complement close to 1, no
 * overflowing product) and inverted by bisection on the normal doubles, the
 * bracket saturating at DBL_MIN and DBL_MAX. Measured relative error below
 * 1e-12 against mpmath references (n from 3 to 10^6, alpha from 1e-300 to
 * 1 - 2^-53); not a proven bound. A UCL that is not a
 * finite normal double (extreme alpha only) is N4M_ERR_NUMERICAL_FAILURE.
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
