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
 * Internal parity fixture: parity/python_generator/src/n4m_parity_pybaselines_ref/modpoly.py
 */
#ifndef N4M_CORE_PREPROCESSING_BASELINES_MODPOLY_H
#define N4M_CORE_PREPROCESSING_BASELINES_MODPOLY_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_modpoly_state_t n4m_pp_modpoly_state_t;

n4m_pp_modpoly_state_t* n4m_pp_modpoly_state_new(int32_t polyorder,
                                                   int32_t max_iter,
                                                   double tol);
void                     n4m_pp_modpoly_state_free(n4m_pp_modpoly_state_t* state);

n4m_status_t n4m_pp_modpoly_state_apply(const n4m_pp_modpoly_state_t* state,
                                         const double* X,
                                         int64_t rows, int64_t cols,
                                         double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_MODPOLY_H */
