/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SecondDerivative operator. Public C ABI lives
 * in c4a.h (§10, Phase 4).
 *
 * Algorithm: two passes of `np.gradient(X, delta, axis=1, edge_order=...)`.
 * The first pass computes the first derivative; the second pass differentiates
 * the result.  Output shape equals input shape (shape-preserving via the
 * same one-sided edge formula as FirstDerivative).
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_SECOND_DERIVATIVE_H
#define CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_SECOND_DERIVATIVE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_second_derivative_state_t c4a_pp_second_derivative_state_t;

c4a_pp_second_derivative_state_t*
c4a_pp_second_derivative_state_new(double delta, int32_t edge_order);

void c4a_pp_second_derivative_state_free(
    c4a_pp_second_derivative_state_t* state);

c4a_status_t c4a_pp_second_derivative_state_apply(
    const c4a_pp_second_derivative_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_SECOND_DERIVATIVE_H */
