/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the Haar operator (Phase 6).  Convenience
 * wrapper that forces ``family = haar`` and ``mode = periodization``
 * on the shared Wavelet kernel, matching ``nirs4all.Haar``.
 */
#ifndef N4M_CORE_PP_WAVELETS_HAAR_H
#define N4M_CORE_PP_WAVELETS_HAAR_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_haar_state_t n4m_pp_haar_state_t;

n4m_pp_haar_state_t* n4m_pp_haar_state_new(void);

void n4m_pp_haar_state_free(n4m_pp_haar_state_t* state);

n4m_status_t n4m_pp_haar_state_apply(
    const n4m_pp_haar_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_WAVELETS_HAAR_H */
