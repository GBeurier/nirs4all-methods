/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * AsLS — Asymmetric Least Squares baseline (Eilers & Boelens 2005).
 *
 * For each row y of X, iteratively solve
 *    (diag(w) + lam * D2^T D2) z = w * y
 * with the asymmetric weight update
 *    new_w[i] = p     if y[i] > z[i]
 *               1-p   otherwise.
 * Convergence: relative L2 difference between successive w arrays drops below
 * `tol`. Otherwise we silently return the last iterate (matching pybaselines).
 *
 * Output: out[i] = y[i] - z[i].
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_ASLS_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_ASLS_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_asls_state_t c4a_pp_asls_state_t;

c4a_pp_asls_state_t* c4a_pp_asls_state_new(double lam, double p,
                                            int32_t max_iter, double tol);
void                  c4a_pp_asls_state_free(c4a_pp_asls_state_t* state);

c4a_status_t c4a_pp_asls_state_apply(const c4a_pp_asls_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_ASLS_H */
