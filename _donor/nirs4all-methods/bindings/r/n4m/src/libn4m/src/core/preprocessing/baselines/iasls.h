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
 * Internal parity fixture: parity/python_generator/src/n4m_parity_pybaselines_ref/iasls.py
 */
#ifndef N4M_CORE_PREPROCESSING_BASELINES_IASLS_H
#define N4M_CORE_PREPROCESSING_BASELINES_IASLS_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_iasls_state_t n4m_pp_iasls_state_t;

n4m_pp_iasls_state_t* n4m_pp_iasls_state_new(double lam, double p,
                                              int32_t polyorder,
                                              int32_t max_iter, double tol);
n4m_pp_iasls_state_t* n4m_pp_iasls_state_new_ex(double lam, double p,
                                                 double lam_1,
                                                 int32_t polyorder,
                                                 int32_t diff_order,
                                                 int32_t max_iter,
                                                 double tol);
void                  n4m_pp_iasls_state_free(n4m_pp_iasls_state_t* state);

n4m_status_t n4m_pp_iasls_state_apply(const n4m_pp_iasls_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_IASLS_H */
