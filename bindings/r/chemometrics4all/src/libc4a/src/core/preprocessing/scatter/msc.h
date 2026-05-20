/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Multiplicative Scatter Correction (MSC)
 * preprocessing operator. The public C ABI lives in c4a.h and is implemented
 * in c_api/c_api_preprocessing.cpp which wraps this engine.
 *
 * Reference: prospectr::msc / pls::msc.
 *
 * Algorithm:
 *   fit(X):
 *       reference[j] = mean(X[:, j])           # reference spectrum, length p
 *
 *   transform(X):
 *       offset[i], slope[i] = ols(X[i, :] ~ 1 + reference)
 *       out[i, j] = (X[i, j] - offset[i]) / slope[i]
 *
 *   inverse_transform(X):
 *       Uses the row coefficients from the last transform call on the same
 *       handle. This preserves the legacy inverse-transform API while the
 *       forward MSC contract matches external implementations.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCATTER_MSC_H
#define CHEMOMETRICS4ALL_CORE_PP_SCATTER_MSC_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_msc_state_t c4a_pp_msc_state_t;

/* Allocate a fresh MSC state. The state starts in the unfitted state. */
c4a_pp_msc_state_t* c4a_pp_msc_state_new(void);

/* Release a state allocated by c4a_pp_msc_state_new. NULL-safe. */
void c4a_pp_msc_state_free(c4a_pp_msc_state_t* state);

/* Returns 1 if `_fit` has been called successfully on this state, 0
 * otherwise. NULL-safe (returns 0 on NULL). */
int c4a_pp_msc_state_is_fitted(const c4a_pp_msc_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 1 and cols >= 2. Replaces any
 * prior fitted state. Returns C4A_OK on success, C4A_ERR_NULL_POINTER for
 * NULL state / X, C4A_ERR_INVALID_ARGUMENT for invalid shape,
 * C4A_ERR_NUMERICAL_FAILURE when the reference spectrum has zero variance
 * (row-wise slopes undefined), and C4A_ERR_OUT_OF_MEMORY on alloc failure. */
c4a_status_t c4a_pp_msc_state_fit(c4a_pp_msc_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing the result to
 * `out` (same shape, may overlap with X). Returns C4A_ERR_NOT_FITTED if the
 * state has not yet been fitted. */
c4a_status_t c4a_pp_msc_state_apply(c4a_pp_msc_state_t* state,
                                     const double* X,
                                     int64_t rows, int64_t cols,
                                     double* out);

/* Apply the inverse of the fitted operator. Same shape contract as apply. */
c4a_status_t c4a_pp_msc_state_inverse_apply(const c4a_pp_msc_state_t* state,
                                             const double* X,
                                             int64_t rows, int64_t cols,
                                             double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCATTER_MSC_H */
