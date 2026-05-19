/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Derivate operator (finite-difference derivative
 * along axis=1). The public C ABI lives in c4a.h.
 *
 * Algorithm:
 *   out = np.diff(X, n=order, axis=1) / delta ** order
 *
 * Output shape is `(rows, cols - order)`. Each iteration of `np.diff`
 * shortens the column count by one; `n=order` repeats the operation.
 *
 * Lifecycle: stateless mathematically but exposes `_create / _fit /
 * _transform / _destroy` to satisfy the stateful-operator ABI contract
 * (Phase 3). `_fit` is a no-op other than setting the fitted flag and
 * recording the column count for shape validation on the subsequent
 * `_transform`.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_DERIVATE_H
#define CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_DERIVATE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_derivate_state_t c4a_pp_derivate_state_t;

/* Allocate a fresh Derivate state. `order` must be >= 1; `delta` must be
 * non-zero. */
c4a_pp_derivate_state_t* c4a_pp_derivate_state_new(int32_t order, double delta);

/* Release a state allocated by c4a_pp_derivate_state_new. NULL-safe. */
void c4a_pp_derivate_state_free(c4a_pp_derivate_state_t* state);

int c4a_pp_derivate_state_is_fitted(const c4a_pp_derivate_state_t* state);

/* Validate dimensions and mark the state fitted. Requires cols > order. */
c4a_status_t c4a_pp_derivate_state_fit(c4a_pp_derivate_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

/* Apply n-th order finite difference along axis=1 with division by
 * delta^order. Input shape is (rows, cols); output shape is
 * (rows, cols - order). */
c4a_status_t c4a_pp_derivate_state_apply(const c4a_pp_derivate_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out);

/* Returns the output column count for a given (order, input_cols).
 * Returns 0 when order >= input_cols or input_cols <= 0. */
int64_t c4a_pp_derivate_output_cols_helper(int32_t order, int64_t input_cols);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_DERIVATE_H */
