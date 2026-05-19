/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BEADS — Baseline Estimation And Denoising with Sparsity
 * (Ning & Selesnick 2014), matching pybaselines' full banded algorithm for
 * the public c4a parameter surface.
 *
 * For each row y of length n:
 *
 *   1. subtract pybaselines' endpoint parabola;
 *   2. build the second-order high-pass filter matrices A and B;
 *   3. iteratively solve the 9-diagonal BEADS system with L1-v2 derivative
 *      weights, asymmetric Huber-like signal weights, and lam_0/lam_1/lam_2;
 *   4. reconstruct the low-pass baseline and return y - baseline.
 *
 * Fixed pybaselines-compatible options: freq_cutoff=0.005, asymmetry=6,
 * filter_type=1, cost_function=2, eps_0=eps_1=1e-6, fit_parabola=TRUE.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_BEADS_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_BEADS_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_beads_state_t c4a_pp_beads_state_t;

c4a_pp_beads_state_t* c4a_pp_beads_state_new(double lam_0, double lam_1,
                                              double lam_2,
                                              int32_t max_iter, double tol);
void                   c4a_pp_beads_state_free(c4a_pp_beads_state_t* state);

c4a_status_t c4a_pp_beads_state_apply(const c4a_pp_beads_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_BEADS_H */
