/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SPlit (data-twinning) splitter. Public ABI
 * lives in n4m.h §15.
 *
 * Reference: nirs4all.operators.splitters.SPlitSplitter (which calls
 * `_twin` in the same module).
 *
 * Algorithm (Vakayil & Joseph 2022):
 *   1. Preprocess X: drop constant columns, then z-score the remaining
 *      columns using population std (ddof=0).
 *   2. Pick starting index u1 via PCG64-drawn uniform * N (Python uses
 *      np.random.randint(N)). The fixture generator uses the same PCG64
 *      draw.
 *   3. While at least r active samples remain:
 *        - Compute squared distance from current to all active samples.
 *        - Pick the r nearest active samples; the closest becomes the
 *          representative for the smaller (test) twin.
 *        - Deactivate the r nearest.
 *        - The next 'current' is the active sample nearest to the
 *          farthest deactivated one.
 *   4. If the test twin has fewer than ceil(N/r) elements, append the
 *      current sample.
 *
 * Output: test = the twin indices in pick order; train = complement,
 * ordered by sample index ascending (matches np.delete(np.arange(N), twin)
 * which preserves ascending order of the remaining indices).
 */
#ifndef N4M_CORE_SPLITTERS_SPLIT_SPLITTER_H
#define N4M_CORE_SPLITTERS_SPLIT_SPLITTER_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_split_splt_state_t {
    double   test_size;   /* 0 < test_size < 1; r = int(1 / test_size). */
    uint64_t seed;
} n4m_split_splt_state_t;

n4m_split_splt_state_t* n4m_split_splt_state_new(double test_size,
                                                  uint64_t seed);
void n4m_split_splt_state_free(n4m_split_splt_state_t* state);

n4m_status_t n4m_split_splt_apply(const n4m_split_splt_state_t* state,
                                   const double* X, int64_t rows, int64_t cols,
                                   n4m_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_SPLITTERS_SPLIT_SPLITTER_H */
