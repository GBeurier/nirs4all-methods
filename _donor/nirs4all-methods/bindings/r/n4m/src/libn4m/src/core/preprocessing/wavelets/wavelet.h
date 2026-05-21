/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Wavelet operator (Phase 6).
 *
 * Single-level discrete wavelet transform applied row-by-row.  For a
 * row of length ``cols`` the output has length ``2 * m`` where ``m``
 * is the output length of the chosen (family, mode) pair — the
 * approximation and detail coefficients are concatenated:
 *
 *   out[i, :m]  = cA(X[i])
 *   out[i, m:]  = cD(X[i])
 *
 * Stateless apart from the constructor parameters.
 *
 * Pure C, no dependencies.  INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_PP_WAVELETS_WAVELET_H
#define N4M_CORE_PP_WAVELETS_WAVELET_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_wavelet_state_t n4m_pp_wavelet_state_t;

n4m_pp_wavelet_state_t* n4m_pp_wavelet_state_new(
    n4m_wavelet_family_t family, n4m_wavelet_mode_t mode);

void n4m_pp_wavelet_state_free(n4m_pp_wavelet_state_t* state);

n4m_wavelet_family_t n4m_pp_wavelet_state_family(
    const n4m_pp_wavelet_state_t* state);
n4m_wavelet_mode_t n4m_pp_wavelet_state_mode(
    const n4m_pp_wavelet_state_t* state);

n4m_status_t n4m_pp_wavelet_state_apply(
    const n4m_pp_wavelet_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_WAVELETS_WAVELET_H */
