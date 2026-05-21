/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SecondDerivative operator. Public C ABI lives
 * in n4m.h (§10, Phase 4).
 *
 * Algorithm: two passes of `np.gradient(X, delta, axis=1, edge_order=...)`.
 * The first pass computes the first derivative; the second pass differentiates
 * the result.  Output shape equals input shape (shape-preserving via the
 * same one-sided edge formula as FirstDerivative).
 */
#ifndef N4M_CORE_PP_DERIVATIVES_SECOND_DERIVATIVE_H
#define N4M_CORE_PP_DERIVATIVES_SECOND_DERIVATIVE_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_second_derivative_state_t n4m_pp_second_derivative_state_t;

n4m_pp_second_derivative_state_t*
n4m_pp_second_derivative_state_new(double delta, int32_t edge_order);

void n4m_pp_second_derivative_state_free(
    n4m_pp_second_derivative_state_t* state);

n4m_status_t n4m_pp_second_derivative_state_apply(
    const n4m_pp_second_derivative_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_DERIVATIVES_SECOND_DERIVATIVE_H */
