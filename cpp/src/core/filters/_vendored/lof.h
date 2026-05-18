/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Local Outlier Factor — vendored minimal implementation.
 *
 * Per Breunig, Kriegel, Ng & Sander (2000). For each training sample i
 * with k-nearest-neighbour set N_k(i):
 *
 *   k-distance(i)        = distance to its k-th nearest neighbour.
 *   reach-distance(i,j)  = max(k-distance(j), dist(i, j)).
 *   lrd_i                = 1 / mean_{j in N_k(i)} reach-distance(i, j).
 *   LOF_i                = mean_{j in N_k(i)} lrd_j / lrd_i.
 *
 * For prediction on new points we use the sklearn ``novelty=True`` mode
 * with the training-set neighbourhood; scoring is the same formula with
 * ``i`` now coming from ``X_test`` and ``N_k(i)`` selected from the
 * training matrix.
 *
 * For NIRS-typical n (<= a few thousand) we use a brute-force k-NN: O(n^2 d)
 * to fit, O(m * n * d) to score. A kd-tree is deferred to Phase 22+.
 *
 * Pure C, INTERNAL. Deterministic given fixed k and a stable tie-break
 * (we use sample index on equal distance).
 */
#ifndef CHEMOMETRICS4ALL_CORE_FILTERS_VENDORED_LOF_H
#define CHEMOMETRICS4ALL_CORE_FILTERS_VENDORED_LOF_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_lof_t c4a_lof_t;

c4a_lof_t* c4a_lof_new(void);
void       c4a_lof_free(c4a_lof_t* lof);

/* Fit on training matrix ``X`` (rows, cols). ``k`` is the neighbourhood size
 * (sklearn default min(20, rows - 1)). */
c4a_status_t c4a_lof_fit(c4a_lof_t* lof,
                          const double* X, int64_t rows, int64_t cols,
                          int k);

/* Score samples in ``X_new`` against the training neighbourhood, writing
 * LOF_i to ``scores[rows_new]``. Higher = more anomalous (LOF > 1 means a
 * sample is in a region less dense than its neighbours'). */
c4a_status_t c4a_lof_score(const c4a_lof_t* lof,
                            const double* X_new, int64_t rows_new,
                            int64_t cols,
                            double* scores);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_FILTERS_VENDORED_LOF_H */
