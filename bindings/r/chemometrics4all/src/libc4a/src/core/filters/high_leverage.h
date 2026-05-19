/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * HighLeverage sample filter — internal engine.
 *
 * Computes the hat-matrix diagonal h_ii = X_i (X^T X)^{-1} X_i^T (the
 * leverage of row i) using either:
 *
 *   - method == 0 (hat): direct hat-matrix path in feature space, requires
 *     rows > cols; auto-falls back to PCA when rows <= cols (matches the
 *     nirs4all behaviour).
 *   - method == 1 (pca): explicit PCA / SVD-based score-space leverage
 *     computation. ``n_components <= 0`` selects min(rows - 1, cols, 50).
 *
 * Threshold semantics (matches HighLeverageFilter in nirs4all):
 *
 *   - When ``absolute_threshold`` is in (0, 1), it overrides any multiplier.
 *   - Otherwise threshold = multiplier * mean(training_leverages).
 *
 * Lifecycle: fit on the training matrix (stores mean / precision /
 * components / threshold), transform writes the keep-mask on subsequent
 * matrices that have the same column count as the training one.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef CHEMOMETRICS4ALL_CORE_FILTERS_HIGH_LEVERAGE_H
#define CHEMOMETRICS4ALL_CORE_FILTERS_HIGH_LEVERAGE_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_filter_leverage_state_t c4a_filter_leverage_state_t;

/* Method enum kept on the internal side to avoid leaking another public
 * enum: 0 = "hat" (auto-falls back to PCA for rows <= cols), 1 = "pca". */
#define C4A_FILTER_LEVERAGE_METHOD_HAT 0
#define C4A_FILTER_LEVERAGE_METHOD_PCA 1

/*
 * Allocate a HighLeverage handle. Returns NULL on:
 *   - method not in {0, 1}
 *   - threshold_multiplier <= 0 AND absolute_threshold not in (0, 1)
 *   - absolute_threshold set but not in (0, 1)
 *
 * The handle starts in the unfitted state.
 *
 * Parameters:
 *   method                — 0 (hat) / 1 (pca)
 *   threshold_multiplier  — multiplier on mean training leverage
 *   use_absolute          — non-zero selects the absolute threshold mode
 *   absolute_threshold    — absolute threshold value in (0, 1)
 *   n_components          — PCA component count; <= 0 selects auto
 *   center                — non-zero applies per-column mean centring
 */
c4a_filter_leverage_state_t* c4a_filter_leverage_state_new(
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute,
    double  absolute_threshold,
    int32_t n_components,
    int     center);

void c4a_filter_leverage_state_free(c4a_filter_leverage_state_t* state);

/* Fit the operator on a (rows, cols) row-major matrix. After fitting:
 *   - mean_, precision_, optional components_ are populated;
 *   - the per-row training leverages are computed and the threshold is set.
 *
 * Returns:
 *   - C4A_OK on success.
 *   - C4A_ERR_INVALID_ARGUMENT if rows < 2 or cols < 1.
 *   - C4A_ERR_OUT_OF_MEMORY on alloc failure.
 *   - C4A_ERR_NUMERICAL_FAILURE on a singular regularised X^T X (extremely
 *     rare given the 1e-10 Tikhonov shift). */
c4a_status_t c4a_filter_leverage_state_fit(c4a_filter_leverage_state_t* state,
                                            const double* X,
                                            int64_t rows, int64_t cols);

/* Return 1 if the state is fitted, 0 otherwise. */
int c4a_filter_leverage_state_is_fitted(const c4a_filter_leverage_state_t* state);

/* Apply the fitted operator on a (rows, cols) row-major matrix, writing the
 * keep mask to ``mask_out`` (1 = keep, 0 = exclude). ``cols`` must match the
 * column count seen at fit time.
 *
 * If non-NULL, ``stats_out`` is filled with (n_samples, n_kept, n_excluded).
 *
 * Returns:
 *   - C4A_OK on success.
 *   - C4A_ERR_NOT_FITTED if the state was never fit.
 *   - C4A_ERR_SHAPE_MISMATCH on cols != fitted cols.
 *   - C4A_ERR_OUT_OF_MEMORY on alloc failure. */
c4a_status_t c4a_filter_leverage_state_apply(
    const c4a_filter_leverage_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    uint8_t* mask_out,
    int64_t* out_n_samples,
    int64_t* out_n_kept,
    int64_t* out_n_excluded);

/* Read-only accessor: the threshold computed at fit time. Returns NaN if
 * the state is not fitted. */
double c4a_filter_leverage_state_threshold(const c4a_filter_leverage_state_t* state);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_FILTERS_HIGH_LEVERAGE_H */
