/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Hotelling T² statistic on the PCA-reduced score space.
 *
 * For a (n_samples x n_features) matrix X:
 *   1. Centre by column means.
 *   2. Compute the thin SVD of the centred matrix: Xc = U Σ V^T.
 *   3. Take the first k = n_components columns of T = U Σ as the scores.
 *   4. Eigenvalues of the sample covariance are λ_j = σ_j² / (n_samples - 1).
 *   5. T²[i] = sum_{j<k} (T[i, j]² / λ_j).
 *
 * The UCL is the standard F-distribution bound:
 *     UCL = k * (n - 1) * (n + 1) / (n * (n - k)) * F^{-1}(1 - alpha; k, n - k)
 * Implemented internally via a numerical inverse of the regularised
 * incomplete beta function.
 */
#ifndef CHEMOMETRICS4ALL_CORE_UTILITIES_HOTELLING_T2_H
#define CHEMOMETRICS4ALL_CORE_UTILITIES_HOTELLING_T2_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

c4a_status_t c4a_util_hotelling_t2_impl(const double* X,
                                         int64_t rows, int64_t cols,
                                         int32_t n_components,
                                         double alpha,
                                         double* t2_per_sample,
                                         double* ucl_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_UTILITIES_HOTELLING_T2_H */
