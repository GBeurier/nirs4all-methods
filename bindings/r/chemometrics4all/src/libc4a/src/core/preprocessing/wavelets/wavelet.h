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
#ifndef CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_H
#define CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_wavelet_state_t c4a_pp_wavelet_state_t;

c4a_pp_wavelet_state_t* c4a_pp_wavelet_state_new(
    c4a_wavelet_family_t family, c4a_wavelet_mode_t mode);

void c4a_pp_wavelet_state_free(c4a_pp_wavelet_state_t* state);

c4a_wavelet_family_t c4a_pp_wavelet_state_family(
    const c4a_pp_wavelet_state_t* state);
c4a_wavelet_mode_t c4a_pp_wavelet_state_mode(
    const c4a_pp_wavelet_state_t* state);

c4a_status_t c4a_pp_wavelet_state_apply(
    const c4a_pp_wavelet_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_H */
