/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for Baseline (column-mean centering). The public C ABI
 * lives in c4a.h and is implemented in c_api/c_api_preprocessing.cpp.
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
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCALING_BASELINE_H
#define CHEMOMETRICS4ALL_CORE_PP_SCALING_BASELINE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_baseline_state_t c4a_pp_baseline_state_t;

/* Allocate a fresh Baseline state. Starts unfitted. */
c4a_pp_baseline_state_t* c4a_pp_baseline_state_new(void);

/* Release a state allocated by c4a_pp_baseline_state_new. NULL-safe. */
void c4a_pp_baseline_state_free(c4a_pp_baseline_state_t* state);

int c4a_pp_baseline_state_is_fitted(const c4a_pp_baseline_state_t* state);

/* Fit on X (rows x cols). Requires rows >= 1 and cols >= 1. */
c4a_status_t c4a_pp_baseline_state_fit(c4a_pp_baseline_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols);

c4a_status_t c4a_pp_baseline_state_apply(const c4a_pp_baseline_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          double* out);

c4a_status_t c4a_pp_baseline_state_inverse_apply(
    const c4a_pp_baseline_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCALING_BASELINE_H */
