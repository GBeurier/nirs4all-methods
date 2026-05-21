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
 * ``u_based_decision=False`` for TruncatedSVD. The Phase 9 n4m engine
 * uses ``u_based_decision=True`` uniformly across FlexiblePCA and
 * FlexibleSVD so the same SVD helper can serve both — see
 * ``cpp/src/core/common/svd.c`` for the canonicalisation step. The
 * frozen NumPy reference in
 * ``parity/python_generator/src/n4m_parity_orthog_ref/flexible_svd.py``
 * matches the n4m convention exactly.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_PP_FEATURE_SELECTION_FLEXIBLE_SVD_H
#define N4M_CORE_PP_FEATURE_SELECTION_FLEXIBLE_SVD_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_flex_svd_state_t n4m_pp_flex_svd_state_t;

/* Allocate a fresh FlexibleSVD state.
 *
 * `n_components_param` is the unified flexible specifier (see the
 * FlexiblePCA header for the count/ratio mode semantics). Values <= 0
 * and NaN are rejected with NULL. */
n4m_pp_flex_svd_state_t* n4m_pp_flex_svd_state_new(double n_components_param);

/* Release a state allocated by n4m_pp_flex_svd_state_new. NULL-safe. */
void n4m_pp_flex_svd_state_free(n4m_pp_flex_svd_state_t* state);

int n4m_pp_flex_svd_state_is_fitted(const n4m_pp_flex_svd_state_t* state);

int64_t n4m_pp_flex_svd_state_n_components(const n4m_pp_flex_svd_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 2 and cols >= 1.
 * Replaces any prior fitted state. */
n4m_status_t n4m_pp_flex_svd_state_fit(n4m_pp_flex_svd_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing the result to
 * `out` (rows x n_components_learned). */
n4m_status_t n4m_pp_flex_svd_state_apply(const n4m_pp_flex_svd_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_FEATURE_SELECTION_FLEXIBLE_SVD_H */
