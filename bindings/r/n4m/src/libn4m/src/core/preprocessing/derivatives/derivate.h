/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Derivate operator (finite-difference derivative
 * along axis=1). The public C ABI lives in n4m.h.
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
#ifndef N4M_CORE_PP_DERIVATIVES_DERIVATE_H
#define N4M_CORE_PP_DERIVATIVES_DERIVATE_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_derivate_state_t n4m_pp_derivate_state_t;

/* Allocate a fresh Derivate state. `order` must be >= 1; `delta` must be
 * non-zero. */
n4m_pp_derivate_state_t* n4m_pp_derivate_state_new(int32_t order, double delta);

/* Release a state allocated by n4m_pp_derivate_state_new. NULL-safe. */
void n4m_pp_derivate_state_free(n4m_pp_derivate_state_t* state);

int n4m_pp_derivate_state_is_fitted(const n4m_pp_derivate_state_t* state);

/* Validate dimensions and mark the state fitted. Requires cols > order. */
n4m_status_t n4m_pp_derivate_state_fit(n4m_pp_derivate_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

/* Apply n-th order finite difference along axis=1 with division by
 * delta^order. Input shape is (rows, cols); output shape is
 * (rows, cols - order). */
n4m_status_t n4m_pp_derivate_state_apply(const n4m_pp_derivate_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out);

/* Returns the output column count for a given (order, input_cols).
 * Returns 0 when order >= input_cols or input_cols <= 0. */
int64_t n4m_pp_derivate_output_cols_helper(int32_t order, int64_t input_cols);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_DERIVATIVES_DERIVATE_H */
