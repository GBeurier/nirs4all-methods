/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for RangeDiscretizer (range-based discretization via
 * `np.digitize(X, bins, right=False)`). The public C ABI lives in n4m.h
 * and is implemented in c_api/c_api_resampling.cpp.
 *
 * Reference: nirs4all.operators.transforms.targets.RangeDiscretizer.
 *
 * Algorithm (stateless):
 *   flat = X.flatten()                         # row-major flatten
 *   labels = np.digitize(flat, bins, right=False)
 *   out = labels.reshape(-1, 1).astype(int32)
 *
 * Edge ordering matches np.digitize: label i means
 *   bins[i-1] <= x < bins[i]
 * with `bins[-1] = -inf` and `bins[len(bins)] = +inf` virtual sentinels (so
 * labels span [0, n_bins] where n_bins = len(bins) + 1).
 */
#ifndef N4M_CORE_PP_RESAMPLING_RANGE_DISCRETIZER_H
#define N4M_CORE_PP_RESAMPLING_RANGE_DISCRETIZER_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_range_disc_state_t n4m_pp_range_disc_state_t;

/* Allocate a range-discretizer state. `bins` must be a strictly ascending
 * array of length `n_edges >= 0`. With `n_edges == 0` everything maps to 0.
 * The state owns an internal copy of the edges. */
n4m_pp_range_disc_state_t* n4m_pp_range_disc_state_new(const double* bins,
                                                        int64_t n_edges);

void n4m_pp_range_disc_state_free(n4m_pp_range_disc_state_t* state);

/* Apply per-element digitize. Output is a row-major int32 matrix flattened
 * to (rows * cols, 1). Caller allocates `out` with size rows * cols. */
n4m_status_t n4m_pp_range_disc_apply(const n4m_pp_range_disc_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      int32_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_RESAMPLING_RANGE_DISCRETIZER_H */
