/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Column centring shared by the PCA outlier statistics (Hotelling T²,
 * Q residuals).
 *
 * Xc receives the column-centred X at one power-of-two scale:
 *
 *     Xc[i, j] = (X[i, j] - mean_j) * 2^-(*scale_exp)
 *
 * with max |Xc| in [0.5, 1), or Xc = 0 and *scale_exp = 0 when every column
 * is constant. Each column is first rescaled exactly by 2^-e_j (max|x| in
 * [0.5, 1)), so the sums cannot overflow whatever the input magnitude, and
 * centred with a corrected two-pass mean: subtract the rounded mean m, then
 * the mean r of the residuals x - m. A plain rounded mean leaves the same
 * offset (up to n eps |mean|) on every observation, i.e. a spurious
 * direction along the ones vector; with the correction the offset falls to
 * O(n eps) of the column's centred variation, and a constant column centres
 * to exact zeros at every scale (x - m is exact by Sterbenz and the same for
 * every row, a multiple of one ulp with at most log2(2n) significant bits,
 * so the n partial sums of r are exact for n < 2^26). The centred columns
 * are then brought to the common scale set by the largest centred variation;
 * constant columns do not influence it.
 *
 * The power-of-two rescalings are exact except for entries that would fall
 * below the normal range of the common scale, i.e. more than 2^1022 below
 * the largest centred value: those are rounded (possibly to 0), a change far
 * below the absolute accuracy of every decomposition that follows. Entries
 * between 2^-1074 and 2^-1022 of it stay nonzero and make the shared SVD
 * refuse its input (dynamic range above 2^480, N4M_ERR_NUMERICAL_FAILURE).
 *
 * Returns N4M_ERR_NUMERICAL_FAILURE for a non-finite entry (Xc unspecified),
 * N4M_OK otherwise. X and Xc are n x p row-major and must not overlap.
 */
#ifndef N4M_CORE_UTILITIES_COLUMN_CENTRING_H
#define N4M_CORE_UTILITIES_COLUMN_CENTRING_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

n4m_status_t n4m_util_centre_columns_pow2(const double* X, int64_t n, int64_t p, double* Xc,
                                          int* scale_exp);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_UTILITIES_COLUMN_CENTRING_H */
