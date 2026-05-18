/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Orthogonal Signal Correction (OSC / DOSC)
 * preprocessing operator. The public C ABI lives in c4a.h §13 and is
 * implemented in c_api/c_api_orthogonalization.cpp which wraps this engine.
 *
 * Reference: nirs4all.operators.transforms.orthogonalization.OSC
 * (Direct OSC variant; Wold 1998 + Westerhuis 2001).
 *
 * Algorithm (univariate target):
 *
 *   fit(X, y):
 *       # Center + scale (ddof=1) by default
 *       X_mean   = mean(X, axis=0)
 *       X_std    = std(X, axis=0, ddof=1, zeros-replaced-with-1)
 *       y_mean   = mean(y)
 *       y_std    = std(y, ddof=1, zero-replaced-with-1)
 *       Xc       = (X - X_mean) / X_std
 *       yc       = (y - y_mean) / y_std
 *
 *       # Y-predictive direction (SVD-fallback path: the leading right-singular
 *       # vector of (Xc, yc) reduces, for univariate y, to a normalised X^T y).
 *       w_pls    = Xc.T @ yc / ||Xc.T @ yc||
 *       X_res    = Xc.copy()
 *
 *       for i in 0..n_components_-1:
 *           w        = X_res.T @ yc / ||X_res.T @ yc||      # PLS weight (univariate)
 *           t        = X_res @ w
 *           p        = X_res.T @ t / (t.T @ t)              # PLS loading
 *           # Gram-Schmidt: subtract w_pls projection from loading p
 *           proj     = (p.T @ w_pls) * w_pls
 *           w_ortho  = (p - proj) / ||p - proj||
 *           t_ortho  = X_res @ w_ortho
 *           p_ortho  = X_res.T @ t_ortho / (t_ortho.T @ t_ortho)
 *           X_res   -= outer(t_ortho, p_ortho)
 *           W_ortho[:, i] = w_ortho
 *           P_ortho[:, i] = p_ortho
 *
 *   transform(X):
 *       Xc = (X - X_mean) / X_std            # if scale
 *       for i in 0..n_components_-1:
 *           t_ortho = Xc @ W_ortho[:, i]
 *           Xc     -= outer(t_ortho, P_ortho[:, i])
 *       out = Xc * X_std + X_mean            # if scale
 *
 * OSC is NOT invertible — the orthogonal-component subtraction is many-to-one
 * (it lives in the row span of X_res after deflation). The public ABI returns
 * C4A_ERR_UNSUPPORTED from inverse_transform.
 *
 * SVD-fallback divergence: for the univariate y case, the SVD fallback path
 * (leading right-singular vector of the rank-1 matrix y x^T) is bit-equal to
 * the normalised X^T y direction. We document this in docs/algorithms/osc.md.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_ORTHOG_OSC_H
#define CHEMOMETRICS4ALL_CORE_PP_ORTHOG_OSC_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_osc_state_t c4a_pp_osc_state_t;

/* Allocate a fresh OSC state. The state starts in the unfitted state.
 * `n_components` must be >= 1 (caller already validated this in the wrapper).
 * `scale != 0` enables center+scale of both X and y by their sample mean
 * and ddof=1 standard deviation; `scale == 0` skips both centering and
 * scaling. */
c4a_pp_osc_state_t* c4a_pp_osc_state_new(int32_t n_components, int scale);

/* Release a state allocated by c4a_pp_osc_state_new. NULL-safe. */
void c4a_pp_osc_state_free(c4a_pp_osc_state_t* state);

/* Returns 1 if `_fit` has been called successfully on this state, 0
 * otherwise. NULL-safe (returns 0 on NULL). */
int c4a_pp_osc_state_is_fitted(const c4a_pp_osc_state_t* state);

/* Fit on (X, y) — X is rows x cols row-major double, y is length-`rows`.
 *
 *   - Requires rows >= 2, cols >= 1, n_components <= min(cols - 1, rows - 2).
 *     Bounds-saturate: if the requested n_components exceeds the feasible
 *     maximum, we clip down to min(cols - 1, rows - 2) (matching nirs4all).
 *   - Returns C4A_ERR_NULL_POINTER for NULL state / X / y.
 *   - Returns C4A_ERR_INVALID_ARGUMENT for invalid shape.
 *   - Returns C4A_ERR_NUMERICAL_FAILURE when X has no Y-correlation
 *     (||X^T y|| < 1e-10), matching the reference.
 *   - Returns C4A_ERR_OUT_OF_MEMORY on allocation failure.
 *   - Returns C4A_ERR_CONVERGENCE_FAILED when extracting orthogonal
 *     components produces a degenerate norm: we clip n_components_ to the
 *     last successful iteration (matching nirs4all's early-break behaviour).
 *     No status code is raised — early break is the documented contract. */
c4a_status_t c4a_pp_osc_state_fit(c4a_pp_osc_state_t* state,
                                   const double* X,
                                   const double* y,
                                   int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing the result to
 * `out` (same shape, may overlap with X). Returns C4A_ERR_NOT_FITTED if the
 * state has not yet been fitted, C4A_ERR_SHAPE_MISMATCH if `cols` differs
 * from the column count seen at fit time. */
c4a_status_t c4a_pp_osc_state_apply(const c4a_pp_osc_state_t* state,
                                     const double* X,
                                     int64_t rows, int64_t cols,
                                     double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_ORTHOG_OSC_H */
