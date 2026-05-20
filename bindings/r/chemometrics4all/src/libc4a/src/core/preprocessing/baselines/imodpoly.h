/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * IModPoly — Improved ModPoly with σ-based stopping (Gan, Ruan, Mo 2006).
 *
 * For each row, fit polynomial baseline z; compute residual stdev; threshold
 * y[i] := y[i] if y[i] < z[i] + devr else z[i] + devr  (clip above-baseline
 * peaks but keep the threshold devr); convergence is on the relative change
 * of devr between iterations.
 *
 * Internal parity fixture: parity/python_generator/src/c4a_parity_pybaselines_ref/imodpoly.py
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_IMODPOLY_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_IMODPOLY_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_imodpoly_state_t c4a_pp_imodpoly_state_t;

c4a_pp_imodpoly_state_t* c4a_pp_imodpoly_state_new(int32_t polyorder,
                                                     int32_t max_iter,
                                                     double tol);
void                      c4a_pp_imodpoly_state_free(c4a_pp_imodpoly_state_t* state);

c4a_status_t c4a_pp_imodpoly_state_apply(const c4a_pp_imodpoly_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_IMODPOLY_H */
