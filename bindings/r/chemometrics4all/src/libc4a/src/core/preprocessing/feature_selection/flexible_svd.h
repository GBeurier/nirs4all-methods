/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the FlexibleSVD operator (Phase 9).
 *
 * Reference: nirs4all.operators.transforms.feature_selection.FlexibleSVD,
 * which wraps sklearn's TruncatedSVD with a flexible ``n_components``
 * argument:
 *
 *   - int >= 1:           keep exactly this many components
 *   - float in (0, 1):    keep enough components for cumulative variance >= ratio
 *
 * Algorithm (matches sklearn TruncatedSVD with full SVD, modulo sign
 * convention — see below):
 *
 *   fit(X):
 *       U, S, Vt = svd(X, full_matrices=False)       # NO centering
 *       U, Vt    = svd_flip(U, Vt, u_based_decision=True)
 *       X_proj   = U * S                              # m x k
 *       explained_variance = var(X_proj, axis=0)
 *       full_var           = sum(var(X, axis=0))
 *       evar_ratio         = explained_variance / full_var
 *       # variance mode: pick smallest k' s.t. cumsum[k'] >= ratio
 *       # integer mode:  k' = clamp(n_components, 1, k)
 *       components_ = Vt[:k', :]                      # shape (k', n)
 *
 *   transform(X):
 *       out = X @ components_.T                       # shape (m_new, k')
 *
 * Sign convention difference from sklearn TruncatedSVD: sklearn uses
 * ``u_based_decision=False`` for TruncatedSVD. The Phase 9 c4a engine
 * uses ``u_based_decision=True`` uniformly across FlexiblePCA and
 * FlexibleSVD so the same SVD helper can serve both — see
 * ``cpp/src/core/common/svd.c`` for the canonicalisation step. The
 * frozen NumPy reference in
 * ``parity/python_generator/src/c4a_parity_orthog_ref/flexible_svd.py``
 * matches the c4a convention exactly.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_FEATURE_SELECTION_FLEXIBLE_SVD_H
#define CHEMOMETRICS4ALL_CORE_PP_FEATURE_SELECTION_FLEXIBLE_SVD_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_flex_svd_state_t c4a_pp_flex_svd_state_t;

/* Allocate a fresh FlexibleSVD state.
 *
 * `n_components_param` is the unified flexible specifier (see the
 * FlexiblePCA header for the count/ratio mode semantics). Values <= 0
 * and NaN are rejected with NULL. */
c4a_pp_flex_svd_state_t* c4a_pp_flex_svd_state_new(double n_components_param);

/* Release a state allocated by c4a_pp_flex_svd_state_new. NULL-safe. */
void c4a_pp_flex_svd_state_free(c4a_pp_flex_svd_state_t* state);

int c4a_pp_flex_svd_state_is_fitted(const c4a_pp_flex_svd_state_t* state);

int64_t c4a_pp_flex_svd_state_n_components(const c4a_pp_flex_svd_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 2 and cols >= 1.
 * Replaces any prior fitted state. */
c4a_status_t c4a_pp_flex_svd_state_fit(c4a_pp_flex_svd_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing the result to
 * `out` (rows x n_components_learned). */
c4a_status_t c4a_pp_flex_svd_state_apply(const c4a_pp_flex_svd_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_FEATURE_SELECTION_FLEXIBLE_SVD_H */
