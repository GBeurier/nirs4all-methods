/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Transfer metrics — computes a 9-element vector quantifying how well two
 * datasets (source / target) are aligned for transfer-learning workflows.
 *
 * The reference implementation is the canonical Python utility in
 *   nirs4all/analysis/transfer_metrics.py (class TransferMetricsComputer).
 *
 * Pipeline per call:
 *
 *   1. Center each dataset by its per-column mean.
 *   2. Compute PCA via the symmetric eigendecomposition of the centered
 *      X^T X covariance matrix (Jacobi rotations on a small p × p block).
 *      Truncate to `r_use = min(n_components, p, n_src - 0, n_tgt - 0)`
 *      effective components, then align both source and target to that
 *      common rank.
 *   3. Compute the nine metrics over the resulting (Z_src, U_src, Z_tgt, U_tgt):
 *
 *        centroid_distance      Euclidean distance between PCA-space means
 *        cka_similarity         Linear CKA on the centered covariance matrices
 *        grassmann_distance     sqrt(sum(theta^2)) of the principal angles
 *                               between the two PCA subspaces in the original
 *                               feature space (defined only when both datasets
 *                               share the same feature count — otherwise NaN)
 *        rv_coefficient         trace(C_s C_t) / sqrt(trace(C_s^2) trace(C_t^2))
 *        procrustes_disparity   Procrustes superposition residual on the first
 *                               two PCA components (matches scipy.spatial.procrustes
 *                               disparity)
 *        trustworthiness        Neighbourhood preservation between Z_src and
 *                               Z_tgt for k = min(k_neighbors, n - 2),
 *                               n = min(n_src, n_tgt)
 *        spread_distance        ||cov(Z_s) - cov(Z_t)||_F  +  mean min Euclidean
 *                               distance between subsampled point clouds
 *                               (subsampling uses a deterministic Lehmer-style
 *                               permutation derived from `seed`)
 *        evr_source             explained-variance ratio of the truncated PCA
 *                               on the source dataset
 *        evr_target             explained-variance ratio of the truncated PCA
 *                               on the target dataset
 *
 * Status semantics follow the universal rules at the top of n4m.h:
 *
 *   - N4M_ERR_NULL_POINTER         X.data, source data, or out is NULL.
 *   - N4M_ERR_INVALID_ARGUMENT     n_components < 1, k_neighbors < 2, or
 *                                  either dataset has fewer than 2 rows.
 *   - N4M_ERR_OUT_OF_MEMORY        scratch allocation failed.
 *   - N4M_ERR_NUMERICAL_FAILURE    eigendecomposition did not converge in
 *                                  the budgeted Jacobi sweep count.
 *
 * Pure C, INTERNAL. Lives under cpp/src/core/utilities/.
 */
#ifndef N4M_CORE_UTILITIES_TRANSFER_METRICS_H
#define N4M_CORE_UTILITIES_TRANSFER_METRICS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-metric result struct. Mirrors the public n4m_transfer_metrics_t one-to-one
 * but lives in the internal layer to keep the public ABI consumer-only. */
typedef struct n4m_transfer_metrics_result_t {
    double centroid_distance;
    double cka_similarity;
    double grassmann_distance;
    double rv_coefficient;
    double procrustes_disparity;
    double trustworthiness;
    double spread_distance;
    double evr_source;
    double evr_target;
} n4m_transfer_metrics_result_t;

/* Compute the 9-metric vector. All inputs row-major contiguous F64 (the
 * c_api wrapper enforces this). `n_src` and `n_tgt` may differ; `p_src` and
 * `p_tgt` may differ — but Grassmann is only meaningful when they agree.
 *
 * On feature-count mismatch the Grassmann field is set to NaN. All other
 * metrics are computed in the truncated PCA space and remain well-defined.
 */
n4m_status_t n4m_transfer_metrics_compute_impl(
    const double* X_src, int64_t n_src, int64_t p_src,
    const double* X_tgt, int64_t n_tgt, int64_t p_tgt,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    n4m_transfer_metrics_result_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_UTILITIES_TRANSFER_METRICS_H */
