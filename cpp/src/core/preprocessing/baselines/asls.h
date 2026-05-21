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
#ifndef N4M_CORE_PREPROCESSING_BASELINES_ASLS_H
#define N4M_CORE_PREPROCESSING_BASELINES_ASLS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_asls_state_t n4m_pp_asls_state_t;

n4m_pp_asls_state_t* n4m_pp_asls_state_new(double lam, double p,
                                            int32_t max_iter, double tol);
void                  n4m_pp_asls_state_free(n4m_pp_asls_state_t* state);

n4m_status_t n4m_pp_asls_state_apply(const n4m_pp_asls_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_ASLS_H */
