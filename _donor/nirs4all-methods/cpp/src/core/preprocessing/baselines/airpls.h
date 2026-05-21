/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * AirPLS — Adaptive Iteratively Reweighted Penalized Least Squares
 *   (Zhang 2010).
 *
 * Same skeleton as AsLS: iteratively solve (diag(w) + lam D2^T D2) z = w y.
 * Weight update is exponential of clipped negative residuals; convergence
 * test is |sum of negative residuals| / |y|_1 < tol.
 */
#ifndef N4M_CORE_PREPROCESSING_BASELINES_AIRPLS_H
#define N4M_CORE_PREPROCESSING_BASELINES_AIRPLS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_airpls_state_t n4m_pp_airpls_state_t;

n4m_pp_airpls_state_t* n4m_pp_airpls_state_new(double lam,
                                                int32_t max_iter, double tol);
void                    n4m_pp_airpls_state_free(n4m_pp_airpls_state_t* state);

n4m_status_t n4m_pp_airpls_state_apply(const n4m_pp_airpls_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols,
                                        double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_AIRPLS_H */
