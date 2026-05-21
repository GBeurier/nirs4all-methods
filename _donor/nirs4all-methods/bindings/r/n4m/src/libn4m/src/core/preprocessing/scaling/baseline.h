/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for Baseline (column-mean centering). The public C ABI
 * lives in n4m.h and is implemented in c_api/c_api_preprocessing.cpp.
 *
 * Reference: nirs4all.operators.transforms.signal.Baseline. Note that the
 * class name is historical: this is column-mean centering (equivalent to
 * `sklearn.preprocessing.StandardScaler(with_std=False)`), NOT a baseline
 * correction in the spectroscopic sense.
 *
 * Algorithm:
 *   fit(X):  mean[j] = mean(X[:, j])      # per-column mean, length p
 *   transform(X):           out[i, j] = X[i, j] - mean[j]
 *   inverse_transform(X):   out[i, j] = X[i, j] + mean[j]
 */
#ifndef N4M_CORE_PP_SCALING_BASELINE_H
#define N4M_CORE_PP_SCALING_BASELINE_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_baseline_state_t n4m_pp_baseline_state_t;

/* Allocate a fresh Baseline state. Starts unfitted. */
n4m_pp_baseline_state_t* n4m_pp_baseline_state_new(void);

/* Release a state allocated by n4m_pp_baseline_state_new. NULL-safe. */
void n4m_pp_baseline_state_free(n4m_pp_baseline_state_t* state);

int n4m_pp_baseline_state_is_fitted(const n4m_pp_baseline_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 1 and cols >= 1. */
n4m_status_t n4m_pp_baseline_state_fit(n4m_pp_baseline_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

n4m_status_t n4m_pp_baseline_state_apply(const n4m_pp_baseline_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          double* out);

n4m_status_t n4m_pp_baseline_state_inverse_apply(
    const n4m_pp_baseline_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCALING_BASELINE_H */
