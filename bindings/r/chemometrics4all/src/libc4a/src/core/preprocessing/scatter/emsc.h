/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Extended Multiplicative Scatter Correction
 * (EMSC) preprocessing operator. The public C ABI lives in c4a.h and is
 * implemented in c_api/c_api_preprocessing.cpp which wraps this engine.
 *
 * Reference: nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection
 * with `scale=False`.
 *
 * Algorithm:
 *   fit(X):
 *       reference[j] = mean(X[:, j])              # per-column mean, length p
 *       wavelengths  = 0, 1, 2, ..., p-1
 *
 *   transform(X):
 *       for each row x_i:
 *           B = [reference, wavelengths^1, wavelengths^2, ..., wavelengths^degree]
 *           c, _, _, _ = numpy.linalg.lstsq(B, x_i)
 *           poly_part = sum_{d=1..degree} c[d] * wavelengths^d
 *           x'_i = (x_i - poly_part) / c[0]
 *
 * Internally we solve the (p x m) least-squares system via the normal
 * equations B^T B @ c = B^T y, using Cholesky on the small (m x m) Gram
 * matrix. m = degree + 1 (one ref column + `degree` polynomial columns),
 * so the system is well-conditioned for small degree and large p.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCATTER_EMSC_H
#define CHEMOMETRICS4ALL_CORE_PP_SCATTER_EMSC_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_emsc_state_t c4a_pp_emsc_state_t;

/* Allocate a fresh EMSC state with the given polynomial degree (>= 1). */
c4a_pp_emsc_state_t* c4a_pp_emsc_state_new(int32_t degree);

/* Release a state allocated by c4a_pp_emsc_state_new. NULL-safe. */
void c4a_pp_emsc_state_free(c4a_pp_emsc_state_t* state);

/* Returns 1 if `_fit` has been called successfully on this state, 0
 * otherwise. NULL-safe (returns 0 on NULL). */
int c4a_pp_emsc_state_is_fitted(const c4a_pp_emsc_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 1 and cols >= degree + 2.
 * Replaces any prior fitted state. */
c4a_status_t c4a_pp_emsc_state_fit(c4a_pp_emsc_state_t* state,
                                    const double* X,
                                    int64_t rows, int64_t cols);

/* Apply the fitted operator on X (rows x cols), writing to `out` of the
 * same shape. */
c4a_status_t c4a_pp_emsc_state_apply(const c4a_pp_emsc_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCATTER_EMSC_H */
