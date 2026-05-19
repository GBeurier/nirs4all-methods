/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the Haar operator (Phase 6).  Convenience
 * wrapper that forces ``family = haar`` and ``mode = periodization``
 * on the shared Wavelet kernel, matching ``nirs4all.Haar``.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_WAVELETS_HAAR_H
#define CHEMOMETRICS4ALL_CORE_PP_WAVELETS_HAAR_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_haar_state_t c4a_pp_haar_state_t;

c4a_pp_haar_state_t* c4a_pp_haar_state_new(void);

void c4a_pp_haar_state_free(c4a_pp_haar_state_t* state);

c4a_status_t c4a_pp_haar_state_apply(
    const c4a_pp_haar_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_WAVELETS_HAAR_H */
