/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ArPLS — Asymmetrically Reweighted Penalized Least Squares (Baek 2015).
 *
 * Same skeleton as AsLS/AirPLS. Weight update:
 *   d := y - z
 *   neg := d[d < 0]
 *   m   := mean(neg);   s := std(neg, ddof=1)
 *   new_w := 1 / (1 + exp((2/s) * (d - (2*s - m))))      // logistic
 * Convergence: relative L2 difference between successive w arrays.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_ARPLS_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_ARPLS_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_arpls_state_t c4a_pp_arpls_state_t;

c4a_pp_arpls_state_t* c4a_pp_arpls_state_new(double lam,
                                              int32_t max_iter, double tol);
void                   c4a_pp_arpls_state_free(c4a_pp_arpls_state_t* state);

c4a_status_t c4a_pp_arpls_state_apply(const c4a_pp_arpls_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_ARPLS_H */
