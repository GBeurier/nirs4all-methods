/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * AirPLS — Adaptive Iteratively Reweighted Penalized Least Squares
 *   (Zhang 2010).
 *
 * Same skeleton as AsLS: iteratively solve (diag(w) + lam D2^T D2) z = w y.
 * Weight update is exponential of clipped negative residuals; convergence
 * test is |sum of negative residuals| / |y|_1 < tol.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_AIRPLS_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_AIRPLS_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_airpls_state_t c4a_pp_airpls_state_t;

c4a_pp_airpls_state_t* c4a_pp_airpls_state_new(double lam,
                                                int32_t max_iter, double tol);
void                    c4a_pp_airpls_state_free(c4a_pp_airpls_state_t* state);

c4a_status_t c4a_pp_airpls_state_apply(const c4a_pp_airpls_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols,
                                        double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_AIRPLS_H */
