/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the NorrisWilliams operator. Public C ABI lives
 * in c4a.h (§10, Phase 4).
 *
 * Algorithm (matches nirs4all.operators.transforms.norris_williams):
 *
 *   for _ in range(derivative_order):
 *       X = segment_smooth(X, segment)         # centred moving average
 *       X = gap_derivative(X, gap, delta)      # finite differences
 *
 * segment_smooth uses edge padding with half = segment // 2 then averages
 * over a window of `segment` samples.
 *
 * gap_derivative uses edge padding with width `gap` then computes
 *   out[i, j] = (padded[i, j + 2 gap] - padded[i, j]) / (2 gap delta)
 *
 * Both passes preserve the input shape.  `derivative_order` must be 1 or 2;
 * the C-API wrapper validates this before reaching the engine.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_NORRIS_WILLIAMS_H
#define CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_NORRIS_WILLIAMS_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_norris_williams_state_t c4a_pp_norris_williams_state_t;

c4a_pp_norris_williams_state_t*
c4a_pp_norris_williams_state_new(int32_t segment, int32_t gap,
                                  int32_t derivative_order, double delta);

void c4a_pp_norris_williams_state_free(
    c4a_pp_norris_williams_state_t* state);

c4a_status_t c4a_pp_norris_williams_state_apply(
    const c4a_pp_norris_williams_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_NORRIS_WILLIAMS_H */
