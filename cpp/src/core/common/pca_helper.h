/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared PCA helper — minimal wrapper over the Jacobi SVD in svd.h.
 *
 * Given a (rows, cols) row-major matrix X, the helper:
 *   1. Computes column means (so callers can centre).
 *   2. Builds the centred matrix X_c = X - mean.
 *   3. Runs the compact Jacobi SVD on X_c: X_c = U @ diag(S) @ Vt.
 *   4. Keeps the top-k components (k = min(requested, n_samples, n_features)).
 *   5. Returns the principal axes (k, cols) row-major (rows of Vt),
 *      training scores (n_samples, k) (= X_c @ Vt^T = U[:, :k] * S[:k]),
 *      and explained_variance[k] = S[k]^2 / (rows - 1) (sklearn semantics).
 *
 * The components have the sklearn sign convention: each component row's
 * maximum-absolute element is non-negative (the c4a_svd_compact helper
 * canonicalises U[:, i]; here we additionally enforce the sign on the
 * component rows themselves to match sklearn.decomposition.PCA).
 *
 * The Vt-based sign canonicalisation in c4a_svd_compact uses U as the
 * anchor. sklearn anchors on Vt; we transform here. Net effect is
 * bit-identical sign behaviour to sklearn for the parity matrices used
 * by Phase 13 (X-outlier filter, pca_residual / pca_leverage methods).
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef CHEMOMETRICS4ALL_CORE_COMMON_PCA_HELPER_H
#define CHEMOMETRICS4ALL_CORE_COMMON_PCA_HELPER_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pca_fit_t {
    int64_t rows;          /* training rows                                  */
    int64_t cols;          /* training cols                                  */
    int64_t k;              /* effective components kept                     */
    double* mean;           /* (cols,)                                        */
    double* components;     /* (k, cols) row-major — principal axes           */
    double* scores;         /* (rows, k) row-major — training scores          */
    double* explained_var;  /* (k,) = S^2 / (rows - 1)                         */
} c4a_pca_fit_t;

/* Fit PCA on X (rows, cols), keeping top-k components.
 *
 * Parameters:
 *   - X            : (rows, cols) row-major, may be NULL if rows == 0.
 *   - rows         : number of samples (>= 2).
 *   - cols         : number of features (>= 1).
 *   - n_components : requested component count; <= 0 selects min(rows, cols).
 *                    Capped at min(rows, cols).
 *   - center       : non-zero applies mean centring before SVD.
 *
 * On success the caller-allocated ``out`` struct receives newly
 * allocated buffers (mean, components, explained_var) which must be
 * freed by ``c4a_pca_fit_free``.
 *
 * Returns:
 *   - C4A_OK on success.
 *   - C4A_ERR_INVALID_ARGUMENT on bad shapes.
 *   - C4A_ERR_OUT_OF_MEMORY on alloc failure.
 *   - C4A_ERR_CONVERGENCE_FAILED if the underlying SVD fails to converge.
 */
c4a_status_t c4a_pca_fit(const double* X, int64_t rows, int64_t cols,
                          int64_t n_components, int center,
                          c4a_pca_fit_t* out);

/* Release every buffer owned by ``fit``. The struct itself is caller-owned;
 * after the call all internal pointers are NULL. NULL-safe. */
void c4a_pca_fit_free(c4a_pca_fit_t* fit);

/* Project ``X_new`` (n_rows, fit->cols) onto the PCA score space, writing
 * the result to ``scores`` (n_rows, fit->k). Subtracts ``fit->mean`` from
 * each row first.
 *
 * Returns C4A_ERR_NULL_POINTER on null inputs, C4A_OK otherwise. */
c4a_status_t c4a_pca_transform(const c4a_pca_fit_t* fit,
                                const double* X_new, int64_t n_rows,
                                double* scores);

/* Reconstruct ``X_new`` (n_rows, fit->cols) from its PCA scores via
 *   X_hat = scores @ components + mean
 * Writes the reconstruction to ``X_hat`` (n_rows, fit->cols). */
c4a_status_t c4a_pca_inverse_transform(const c4a_pca_fit_t* fit,
                                        const double* scores,
                                        int64_t n_rows,
                                        double* X_hat);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_COMMON_PCA_HELPER_H */
