/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Minimum Covariance Determinant (MCD) — vendored simplified implementation.
 *
 * Estimates a robust location vector and covariance matrix using a
 * FAST-MCD-style concentration loop (Rousseeuw & Van Driessen, 1999),
 * but with a single seeded sub-sample rather than the full multi-start
 * ensemble. This is simpler than sklearn's MinCovDet but gives the
 * correct asymptotic robustness for parity-style validation.
 *
 * Algorithm:
 *   1. Sample h = ceil((n + p + 1) / 2) random rows (PCG64-seeded).
 *   2. Compute mean and covariance of the sub-sample.
 *   3. Concentration step: for every row in the full matrix compute the
 *      Mahalanobis distance against the current (mean, cov); keep the h
 *      rows with the smallest distances and recompute mean + covariance.
 *   4. Iterate step 3 until the determinant of the covariance stabilises
 *      (relative change < 1e-12) or max_iters reached (default 30).
 *   5. Apply Rousseeuw's consistency correction: scale by
 *      median(MD^2) / chi2_inv(0.5, p)  — matches the
 *      sklearn ``correct_covariance`` step.
 *
 * Limitations vs. sklearn (documented as "FAST-MCD ensemble deferred"):
 *   - Single random start (sklearn does ``n_trials = 500``).
 *   - No support_fraction parameter beyond the canonical h.
 *
 * Pure C, INTERNAL.
 */
#ifndef N4M_CORE_FILTERS_VENDORED_MIN_COV_DET_H
#define N4M_CORE_FILTERS_VENDORED_MIN_COV_DET_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_mcd_t n4m_mcd_t;

n4m_mcd_t* n4m_mcd_new(void);
void       n4m_mcd_free(n4m_mcd_t* mcd);

/* Fit on (rows, cols) row-major matrix X. Seed drives the initial
 * sub-sample. */
n4m_status_t n4m_mcd_fit(n4m_mcd_t* mcd,
                          const double* X, int64_t rows, int64_t cols,
                          uint64_t seed);

/* Compute squared Mahalanobis distances per row of X (X_rows, cols) against
 * the fitted location + precision. Writes to ``mahal_sq`` (X_rows,). */
n4m_status_t n4m_mcd_mahalanobis_sq(const n4m_mcd_t* mcd,
                                     const double* X, int64_t X_rows,
                                     int64_t cols, double* mahal_sq);

/* Access the fitted location (length cols). Returns NULL if not fitted. */
const double* n4m_mcd_location(const n4m_mcd_t* mcd);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_FILTERS_VENDORED_MIN_COV_DET_H */
