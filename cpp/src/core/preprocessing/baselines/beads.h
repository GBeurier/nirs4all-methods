/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BEADS (simplified) — Baseline Estimation And Denoising with Sparsity
 * (Ning & Selesnick 2014).
 *
 * This is the simplified Phase 5b variant that uses a reweighted-L2
 * sparsity surrogate over a banded pentadiagonal system:
 *
 *   For each row y of length n:
 *     w := ones(n)
 *     for iter in 0..max_iter:
 *         A := diag(w) + lam_1 * D_2^T D_2 + lam_2 * D_2^T D_2
 *              (== diag(w) + (lam_1 + lam_2) * D_2^T D_2 in the simplified
 *               variant — see docs/algorithms/beads.md)
 *         rhs := lam_0 * (w * y)
 *         z := A^{-1} rhs
 *         d := y - z
 *         w_new[i] := 1 / (|d[i]| + eps)         # reweighted L2 step
 *         normalise so that sum(w_new) = n        # keep scale comparable
 *         if rel_l2_diff(z, z_old) < tol: break
 *         w := w_new
 *     out := y - z
 *
 * The full BEADS algorithm uses a 7-diagonal system combining 1st and 2nd
 * differences with a Chebyshev approximation of |·|. The simplified Phase 5b
 * version stays pentadiagonal so the existing banded5 LDLT solver applies
 * unchanged — see docs/algorithms/beads.md for the deferred variant.
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/beads.py
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
