/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SNIP — Statistics-sensitive Non-linear Iterative Peak-clipping baseline
 * (Ryan 1988, Morháč 1997).
 *
 * Pure arithmetic algorithm (no linear algebra). For each row y of length n:
 *
 *   1. v[i] := log(log(sqrt(|y[i]| + 1) + 1) + 1)        # LLS transform
 *   2. for w in 1..max_half_window:
 *        for i in [w, n - w]:
 *            v[i] := min(v[i], (v[i - w] + v[i + w]) / 2)
 *   3. inverse LLS: z[i] := (exp(exp(v[i]) - 1) - 1)^2 - 1
 *   4. out := original_y - z
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/snip.py
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_SNIP_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_SNIP_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_snip_state_t c4a_pp_snip_state_t;

c4a_pp_snip_state_t* c4a_pp_snip_state_new(int32_t max_half_window);
void                  c4a_pp_snip_state_free(c4a_pp_snip_state_t* state);

c4a_status_t c4a_pp_snip_state_apply(const c4a_pp_snip_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_SNIP_H */
