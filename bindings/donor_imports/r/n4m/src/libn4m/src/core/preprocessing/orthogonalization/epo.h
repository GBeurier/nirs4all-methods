/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the External Parameter Orthogonalization (EPO)
 * preprocessing operator. The public C ABI lives in n4m.h §13 and is
 * implemented in c_api/c_api_orthogonalization.cpp.
 *
 * Reference: nirs4all.operators.transforms.orthogonalization.EPO
 * (Roger 2003).
 *
 * Algorithm (univariate external parameter d, length n_samples):
 *
 *   fit(X, d):
 *       X_mean = mean(X, axis=0)
 *       d_mean = mean(d)
 *       Xc = X - X_mean                       # if scale
 *       dc = d - d_mean
 *       # Least squares: solve  dc @ B = Xc   for B (1 x cols)
 *       # Equivalent to: B[j] = (dc · Xc[:, j]) / (dc · dc)
 *       B = lstsq(dc, Xc)
 *
 *   transform(X):
 *       Xc = X - X_mean                        # if scale
 *       Xc -= outer(d_mean_used_at_fit_time - d_mean, B)  # zero when scale,
 *       out = Xc + X_mean                                  #  since d is unknown
 *
 *   fit_transform(X, d):
 *       Xc = X - X_mean
 *       dc = d - d_mean
 *       Xf = Xc - outer(dc, B)
 *       out = Xf + X_mean
 *
 * The non-fit_transform `_transform` matches the nirs4all behaviour: at
 * transform time `d` is unknown, so we assume it sits at the calibration
 * mean (dc == 0) and the EPO correction reduces to centering + uncentering
 * (a no-op).  The projection matrix B is still stored in the handle so
 * downstream consumers can apply it explicitly if they have d (this is
 * exposed via a sibling ABI in a future phase if needed; today we ship the
 * minimal stateful operator only).
 */
#ifndef N4M_CORE_PP_ORTHOG_EPO_H
#define N4M_CORE_PP_ORTHOG_EPO_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_epo_state_t n4m_pp_epo_state_t;

/* Allocate a fresh EPO state. The state starts in the unfitted state.
 * `scale != 0` enables centering of both X and d by their per-column
 * sample means.  Univariate d (1 external parameter). */
n4m_pp_epo_state_t* n4m_pp_epo_state_new(int scale);

/* Release a state allocated by n4m_pp_epo_state_new. NULL-safe. */
void n4m_pp_epo_state_free(n4m_pp_epo_state_t* state);

/* Returns 1 if `_fit` has been called successfully on this state, 0
 * otherwise. NULL-safe. */
int n4m_pp_epo_state_is_fitted(const n4m_pp_epo_state_t* state);

/* Fit on (X, d).  X is rows x cols row-major double, d is length-`rows`.
 *   - Requires rows >= 2, cols >= 1, and Var(d) > 0.
 *   - Returns N4M_ERR_NULL_POINTER for NULL state / X / d.
 *   - Returns N4M_ERR_INVALID_ARGUMENT for invalid shape.
 *   - Returns N4M_ERR_NUMERICAL_FAILURE when d has zero variance
 *     (the least-squares system is singular).
 *   - Returns N4M_ERR_OUT_OF_MEMORY on allocation failure. */
n4m_status_t n4m_pp_epo_state_fit(n4m_pp_epo_state_t* state,
                                   const double* X,
                                   const double* d,
                                   int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing to `out`.
 *
 * Since `d` is unknown at transform time the canonical EPO projection
 * reduces to centering + uncentering (a no-op) — matching nirs4all's
 * documented behaviour. The projection matrix B is still stored in the
 * handle but is *not* applied here (it can only be applied when the
 * caller has d for the new samples). This delivers the
 * "calibration-only" transform contract.
 *
 * Returns N4M_ERR_NOT_FITTED before fit, N4M_ERR_SHAPE_MISMATCH if
 * `cols` disagrees with the fit-time column count. */
n4m_status_t n4m_pp_epo_state_apply(const n4m_pp_epo_state_t* state,
                                     const double* X,
                                     int64_t rows, int64_t cols,
                                     double* out);

/* Convenience: fit_transform style — apply on the training (X, d) pair
 * using the *just-fitted* projection.  Operates on the same rows used at
 * fit time. Caller must call _fit first. Same shape / status contracts
 * as _apply, with one extra: rows must equal the fit-time row count
 * (since d is per-sample and bound to that count). */
n4m_status_t n4m_pp_epo_state_apply_with_d(const n4m_pp_epo_state_t* state,
                                            const double* X,
                                            const double* d,
                                            int64_t rows, int64_t cols,
                                            double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_ORTHOG_EPO_H */
