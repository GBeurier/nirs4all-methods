/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the FlexiblePCA operator (Phase 9).
 *
 * Reference: nirs4all.operators.transforms.feature_selection.FlexiblePCA,
 * which wraps sklearn's PCA with a flexible ``n_components`` argument:
 *
 *   - int >= 1:           keep exactly this many components
 *   - float in (0, 1):    keep enough components for cumulative variance >= ratio
 *
 * Algorithm (matches sklearn PCA with ``svd_solver='full'``):
 *
 *   fit(X):
 *       mean = X.mean(axis=0)                  # shape (n,)
 *       Xc   = X - mean                        # centered, shape (m, n)
 *       U, S, Vt = svd(Xc, full_matrices=False) # k = min(m, n)
 *       U, Vt = svd_flip(U, Vt, u_based_decision=True)
 *       explained_variance       = S^2 / (m - 1)
 *       explained_variance_ratio = explained_variance / sum(explained_variance)
 *       # variance mode: pick smallest k' s.t. cumsum[k'] >= ratio
 *       # integer mode:  k' = clamp(n_components, 1, k)
 *       components_ = Vt[:k', :]               # shape (k', n)
 *
 *   transform(X):
 *       out = (X - mean) @ components_.T       # shape (m_new, k')
 *
 * The signs are canonicalised so the largest-absolute-value element of
 * each U column is positive (sklearn ``svd_flip`` with
 * ``u_based_decision=True``). This is implemented inside the shared SVD
 * helper (``cpp/src/core/common/svd.c``).
 *
 * Whitening is NOT implemented in this Phase 9 partial — the brief is
 * silent on it and the parity reference uses ``whiten=False``.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_FEATURE_SELECTION_FLEXIBLE_PCA_H
#define CHEMOMETRICS4ALL_CORE_PP_FEATURE_SELECTION_FLEXIBLE_PCA_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_flex_pca_state_t c4a_pp_flex_pca_state_t;

/* Allocate a fresh FlexiblePCA state.
 *
 * `n_components_param` is the unified flexible specifier:
 *   - integer encoded as a positive whole-number double (>= 1.0) selects
 *     count mode (e.g. 10.0 == "keep 10 components"),
 *   - float in (0, 1) selects variance-ratio mode (e.g. 0.95 == "smallest
 *     k such that cumulative variance ratio >= 0.95").
 *
 * `n_components_param` <= 0 or NaN is rejected with NULL.
 *
 * The state starts in the unfitted state. */
c4a_pp_flex_pca_state_t* c4a_pp_flex_pca_state_new(double n_components_param);

/* Release a state allocated by c4a_pp_flex_pca_state_new. NULL-safe. */
void c4a_pp_flex_pca_state_free(c4a_pp_flex_pca_state_t* state);

/* Returns 1 if `_fit` has been called successfully on this state, 0
 * otherwise. NULL-safe (returns 0 on NULL). */
int c4a_pp_flex_pca_state_is_fitted(const c4a_pp_flex_pca_state_t* state);

/* Returns the learned number of components after a successful fit. Returns
 * 0 when the state has not been fitted. */
int64_t c4a_pp_flex_pca_state_n_components(const c4a_pp_flex_pca_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 2 and cols >= 1.
 * Replaces any prior fitted state. */
c4a_status_t c4a_pp_flex_pca_state_fit(c4a_pp_flex_pca_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

/* Apply the fitted operator on `X` (rows x cols), writing the result to
 * `out` (rows x n_components_learned). */
c4a_status_t c4a_pp_flex_pca_state_apply(const c4a_pp_flex_pca_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_FEATURE_SELECTION_FLEXIBLE_PCA_H */
