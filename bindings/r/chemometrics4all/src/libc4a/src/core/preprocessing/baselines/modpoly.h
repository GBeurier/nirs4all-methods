/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ModPoly — Lieber & Mahadevan-Jansen 2003 modified polynomial baseline.
 *
 * For each row y of length n the algorithm fits a polynomial of degree
 * `polyorder` to (positions, y), then replaces y[i] with min(y[i], z[i])
 * (the "clip peaks" step). The new y is fitted again, and so on until the
 * baseline change is below `tol` (relative L2) or `max_iter` is reached.
 *
 * Output: original_y - z_final (baseline-subtracted spectrum).
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/modpoly.py
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_MODPOLY_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_MODPOLY_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_modpoly_state_t c4a_pp_modpoly_state_t;

c4a_pp_modpoly_state_t* c4a_pp_modpoly_state_new(int32_t polyorder,
                                                   int32_t max_iter,
                                                   double tol);
void                     c4a_pp_modpoly_state_free(c4a_pp_modpoly_state_t* state);

c4a_status_t c4a_pp_modpoly_state_apply(const c4a_pp_modpoly_state_t* state,
                                         const double* X,
                                         int64_t rows, int64_t cols,
                                         double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_MODPOLY_H */
