/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * XOutlierFilter — X-feature based sample outlier detection with six
 * selectable methods.
 *
 * Mirrors ``nirs4all.operators.filters.XOutlierFilter`` (Phase 13 — new
 * member of the ``c4a_filter_*`` ABI category). Stateful: fit on
 * training X, then apply on training or new X with the same column count.
 *
 * Methods (encoded as int32_t internally; the public enum lands in c4a.h
 * §18 at integration):
 *
 *   0  C4A_X_OUTLIER_MAHALANOBIS         Empirical-cov Mahalanobis distance.
 *                                          Threshold default = sqrt(chi²_0.975(p)).
 *   1  C4A_X_OUTLIER_ROBUST_MAHALANOBIS  Mahalanobis with MinCovDet
 *                                          (simplified FAST-MCD).
 *   2  C4A_X_OUTLIER_PCA_RESIDUAL        Q-statistic (sum sq. reconstruction
 *                                          residual). Threshold default =
 *                                          95th percentile of training Q.
 *   3  C4A_X_OUTLIER_PCA_LEVERAGE        Hotelling T² in PCA score space.
 *                                          Threshold default = 95th percentile
 *                                          of training T².
 *   4  C4A_X_OUTLIER_ISOLATION_FOREST    Sklearn-style ensemble. Threshold
 *                                          set by contamination quantile.
 *   5  C4A_X_OUTLIER_LOF                 Local Outlier Factor (k=min(20,n-1)).
 *                                          Threshold set by contamination
 *                                          quantile.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI
 * directly.
 */
#ifndef CHEMOMETRICS4ALL_CORE_FILTERS_X_OUTLIER_H
#define CHEMOMETRICS4ALL_CORE_FILTERS_X_OUTLIER_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

#define C4A_CORE_X_OUTLIER_MAHALANOBIS         0
#define C4A_CORE_X_OUTLIER_ROBUST_MAHALANOBIS  1
#define C4A_CORE_X_OUTLIER_PCA_RESIDUAL        2
#define C4A_CORE_X_OUTLIER_PCA_LEVERAGE        3
#define C4A_CORE_X_OUTLIER_ISOLATION_FOREST    4
#define C4A_CORE_X_OUTLIER_LOF                 5

/* Shared filter-stats struct, byte-compatible with the public
 * c4a_filter_stats_t (declared in c4a.h §18). */
typedef struct c4a_core_x_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} c4a_core_x_filter_stats_t;

typedef struct c4a_filter_x_outlier_state_t c4a_filter_x_outlier_state_t;

/* Allocate an unfitted handle. Returns NULL on:
 *   - method outside {0..5}
 *   - threshold < 0 with use_threshold non-zero
 *   - contamination outside (0, 0.5]
 *   - n_components < 0
 *
 * Parameters:
 *   method          — 0..5
 *   use_threshold   — non-zero overrides the method's default threshold
 *                     with ``threshold``
 *   threshold       — user-supplied threshold value (Mahalanobis distance,
 *                     Q, T², iforest mean-depth, or LOF as applicable)
 *   n_components    — PCA component count for pca_* / mahalanobis_* paths;
 *                     <= 0 selects min(rows-1, cols, default_cap)
 *   contamination   — outlier fraction in (0, 0.5] (default 0.1) for
 *                     iforest / lof
 *   seed            — RNG seed for iforest / robust-mahalanobis
 *   n_estimators    — iforest tree count (default 100)
 *   max_samples     — iforest sub-sample size per tree (default 256)
 */
c4a_filter_x_outlier_state_t* c4a_filter_x_outlier_state_new(
    int32_t method,
    int     use_threshold,
    double  threshold,
    int32_t n_components,
    double  contamination,
    uint64_t seed,
    int32_t n_estimators,
    int64_t max_samples);

void c4a_filter_x_outlier_state_free(c4a_filter_x_outlier_state_t* state);

/* Fit on training matrix X (rows, cols).
 *
 * Returns C4A_OK on success, C4A_ERR_INVALID_ARGUMENT for bad shapes,
 * C4A_ERR_OUT_OF_MEMORY on alloc failure. */
c4a_status_t c4a_filter_x_outlier_state_fit(
    c4a_filter_x_outlier_state_t* state,
    const double* X, int64_t rows, int64_t cols);

int c4a_filter_x_outlier_state_is_fitted(
    const c4a_filter_x_outlier_state_t* state);

/* Apply on (rows, cols) row-major matrix. cols must match the fit-time cols.
 * mask_out[i] = 1 keeps sample i, 0 excludes. */
c4a_status_t c4a_filter_x_outlier_state_apply(
    const c4a_filter_x_outlier_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    uint8_t* mask_out,
    c4a_core_x_filter_stats_t* stats_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_FILTERS_X_OUTLIER_H */
