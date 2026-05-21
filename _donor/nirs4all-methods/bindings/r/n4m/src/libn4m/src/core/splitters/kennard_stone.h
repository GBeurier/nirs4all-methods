/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Kennard-Stone sample-partitioning splitter
 * (max-min Euclidean distance on X). The public C ABI lives in n4m.h §15
 * and is implemented in c_api/c_api_splitters.cpp which wraps this engine.
 *
 * Reference: nirs4all.operators.splitters.KennardStoneSplitter
 *
 * Algorithm (matches NumPy reference):
 *   1. Compute the full N x N Euclidean distance matrix on X.
 *   2. Find the two globally farthest points (np.argmax + unravel_index).
 *   3. Iteratively pick the sample maximising the minimum distance to the
 *      already-selected training set.
 *
 * Output: integer index arrays produced by n4m_split_max_min_selection.
 */
#ifndef N4M_CORE_SPLITTERS_KENNARD_STONE_H
#define N4M_CORE_SPLITTERS_KENNARD_STONE_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_split_ks_state_t {
    double test_size;
} n4m_split_ks_state_t;

n4m_split_ks_state_t* n4m_split_ks_state_new(double test_size);
void                  n4m_split_ks_state_free(n4m_split_ks_state_t* state);

/* Run Kennard-Stone selection on a (rows x cols) row-major F64 matrix.
 * Allocates train/test buffers inside `out` via n4m_split_result_alloc. */
n4m_status_t n4m_split_ks_apply(const n4m_split_ks_state_t* state,
                                const double* X, int64_t rows, int64_t cols,
                                n4m_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_SPLITTERS_KENNARD_STONE_H */
