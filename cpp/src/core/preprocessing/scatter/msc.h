/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Multiplicative Scatter Correction (MSC)
 * preprocessing operator. The public C ABI lives in n4m.h and is implemented
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
#ifndef N4M_CORE_PP_SCATTER_MSC_H
#define N4M_CORE_PP_SCATTER_MSC_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_msc_state_t n4m_pp_msc_state_t;

/* Allocate a fresh MSC state. The state starts in the unfitted state. */
n4m_pp_msc_state_t* n4m_pp_msc_state_new(void);

/* Release a state allocated by n4m_pp_msc_state_new. NULL-safe. */
void n4m_pp_msc_state_free(n4m_pp_msc_state_t* state);

/* Returns 1 if `_fit` has been called successfully on this state, 0
 * otherwise. NULL-safe (returns 0 on NULL). */
int n4m_pp_msc_state_is_fitted(const n4m_pp_msc_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 1 and cols >= 2. Replaces any
 * prior fitted state. Returns N4M_OK on success, N4M_ERR_NULL_POINTER for
 * NULL state / X, N4M_ERR_INVALID_ARGUMENT for invalid shape,
 * N4M_ERR_NUMERICAL_FAILURE when the reference spectrum has zero variance
 * (row-wise slopes undefined), and N4M_ERR_OUT_OF_MEMORY on alloc failure. */
n4m_status_t n4m_pp_msc_state_fit(n4m_pp_msc_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing the result to
 * `out` (same shape, may overlap with X). Returns N4M_ERR_NOT_FITTED if the
 * state has not yet been fitted. */
n4m_status_t n4m_pp_msc_state_apply(n4m_pp_msc_state_t* state,
                                     const double* X,
                                     int64_t rows, int64_t cols,
                                     double* out);

/* Apply the inverse of the fitted operator. Same shape contract as apply. */
n4m_status_t n4m_pp_msc_state_inverse_apply(const n4m_pp_msc_state_t* state,
                                             const double* X,
                                             int64_t rows, int64_t cols,
                                             double* out);

/* Number of features in the fitted reference spectrum (0 if unfitted).
 * NULL-safe (returns 0). Lets a binding serialize the fitted state. */
int64_t n4m_pp_msc_state_n_features(const n4m_pp_msc_state_t* state);

/* Copy the fitted reference spectrum (length n_features) into `out`. Returns
 * N4M_ERR_NOT_FITTED when unfitted, N4M_ERR_SHAPE_MISMATCH when `cols` differs
 * from the fitted feature count. */
n4m_status_t n4m_pp_msc_state_get_reference(const n4m_pp_msc_state_t* state,
                                             double* out, int64_t cols);

/* Restore a fitted state directly from a reference spectrum (length `cols`,
 * >= 2). Recomputes the cached mean/denominator and marks the state fitted —
 * the inverse of get_reference. This is how a binding round-trips MSC state
 * (e.g. for predict-later) without re-fitting on the original training data. */
n4m_status_t n4m_pp_msc_state_set_reference(n4m_pp_msc_state_t* state,
                                             const double* reference,
                                             int64_t cols);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCATTER_MSC_H */
