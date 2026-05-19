/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * IAsLS — Improved Asymmetric Least Squares (He et al. 2014).
 *
 * Step 1: fit a polynomial baseline z0 of degree `polyorder` to (positions, y)
 *         to get an initial baseline guess.
 * Step 2: initialise weights via the AsLS asymmetry rule using z0 as a
 *         reference baseline: w[i] = p if y[i] > z0[i] else (1 - p).
 * Step 3: run pybaselines-compatible IAsLS iterations against
 *         `lam * D_2^T D_2 + lam_1 * D_1^T D_1`, with squared AsLS weights
 *         and the first-derivative residual term on the right-hand side.
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/iasls.py
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_IASLS_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_IASLS_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_iasls_state_t c4a_pp_iasls_state_t;

c4a_pp_iasls_state_t* c4a_pp_iasls_state_new(double lam, double p,
                                              int32_t polyorder,
                                              int32_t max_iter, double tol);
c4a_pp_iasls_state_t* c4a_pp_iasls_state_new_ex(double lam, double p,
                                                 double lam_1,
                                                 int32_t polyorder,
                                                 int32_t diff_order,
                                                 int32_t max_iter,
                                                 double tol);
void                  c4a_pp_iasls_state_free(c4a_pp_iasls_state_t* state);

c4a_status_t c4a_pp_iasls_state_apply(const c4a_pp_iasls_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_IASLS_H */
