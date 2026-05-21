/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping baseline
 * (Ryan 1988, Morháč 1997).
 *
 * Pure arithmetic algorithm (no linear algebra). For each row y of length n,
 * this follows pybaselines.smooth.snip with filter_order=2:
 *
 *   1. linearly extrapolate max_half_window samples at both edges
 *   2. for w in 1..max_half_window:
 *        filters[i] := (baseline[i - w] + baseline[i + w]) / 2
 *        baseline[i] := min(baseline[i], filters[i]) for the whole slice
 *   3. out := original_y - unpadded_baseline
 *
 * Reference: pybaselines.Baseline().snip(..., filter_order=2)
 */
#ifndef N4M_CORE_PREPROCESSING_BASELINES_SNIP_H
#define N4M_CORE_PREPROCESSING_BASELINES_SNIP_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_snip_state_t n4m_pp_snip_state_t;

n4m_pp_snip_state_t* n4m_pp_snip_state_new(int32_t max_half_window);
void                  n4m_pp_snip_state_free(n4m_pp_snip_state_t* state);

n4m_status_t n4m_pp_snip_state_apply(const n4m_pp_snip_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_SNIP_H */
