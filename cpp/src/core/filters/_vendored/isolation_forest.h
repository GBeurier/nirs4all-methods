/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Isolation Forest — vendored minimal implementation.
 *
 * Following the Liu, Ting, Zhou (2008) algorithm and sklearn's defaults:
 *
 *   - Ensemble of ``n_estimators`` isolation trees built from sub-samples
 *     of ``max_samples = min(256, n_samples)`` rows each.
 *   - Each tree recursively splits on a random feature with a random split
 *     value in [min, max] of that feature's column within the current node.
 *   - Recursion stops on node depth >= ceil(log2(max_samples)) (sklearn
 *     ``max_depth``) or when fewer than 2 samples remain.
 *   - At evaluation time, each sample is pushed through every tree; the
 *     observed path length is the depth at which the sample reaches a leaf,
 *     with a Harmonic-c(n_leaf) correction for unsplit nodes:
 *
 *       c(n) = 2 * (ln(n - 1) + 0.5772156649) - 2 * (n - 1) / n
 *
 *   - Average path length E_h is averaged across trees, then normalised
 *     score = -E_h / c(max_samples). sklearn's ``score_samples`` returns
 *     ``-2^(-E_h / c(max_samples))`` (negated anomaly score); here we
 *     keep the unsigned form because the XOutlierFilter uses
 *     contamination quantile thresholding rather than the sklearn
 *     score sign convention.
 *
 *   - For deterministic / parity output we use PCG64 from
 *     core/common/rng_pcg64.h, seeded from the caller-supplied uint64.
 *
 * Pure C, INTERNAL. Lifecycle is "fit + score". Single transform-only API
 * matches the c4a_filter_x_outlier_state usage.
 */
#ifndef CHEMOMETRICS4ALL_CORE_FILTERS_VENDORED_ISOLATION_FOREST_H
#define CHEMOMETRICS4ALL_CORE_FILTERS_VENDORED_ISOLATION_FOREST_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_iforest_t c4a_iforest_t;

/* Allocate an empty handle; caller-owned. */
c4a_iforest_t* c4a_iforest_new(void);
void           c4a_iforest_free(c4a_iforest_t* f);

/* Fit ``n_estimators`` trees on ``X`` (rows, cols). The random sub-sample
 * size for each tree is ``min(max_samples, rows)`` (sklearn default 256).
 * ``seed`` drives the PCG64 RNG; bit-reproducible across runs. */
c4a_status_t c4a_iforest_fit(c4a_iforest_t* f,
                              const double* X, int64_t rows, int64_t cols,
                              int n_estimators, int64_t max_samples,
                              uint64_t seed);

/* Score samples in ``X`` (rows, cols). For each sample writes the
 * normalised mean path length (lower = more anomalous) to ``scores[rows]``.
 *
 * The sklearn-equivalent ``score_samples`` (negated) is then
 *   ``- 2 ** (- score[i])``; our pca/iforest test path uses the unsigned
 * mean depth directly for thresholding by contamination quantile. */
c4a_status_t c4a_iforest_score(const c4a_iforest_t* f,
                                const double* X, int64_t rows, int64_t cols,
                                double* scores);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_FILTERS_VENDORED_ISOLATION_FOREST_H */
