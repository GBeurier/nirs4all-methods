/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the WaveletSVD operator (Phase 6).
 *
 * Same DWT-flatten pipeline as WaveletPCA, but the projection step
 * uses Truncated SVD (no mean subtraction).  ``transform`` returns the
 * scores ``X @ Vt.T``, matching ``sklearn.decomposition.TruncatedSVD``
 * (also identical to the chemometrics4all FlexibleSVD operator).
 *
 * Pure C, no dependencies.  INTERNAL: never exposed by the public ABI.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_SVD_H
#define CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_SVD_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_wavelet_svd_state_t c4a_pp_wavelet_svd_state_t;

c4a_pp_wavelet_svd_state_t* c4a_pp_wavelet_svd_state_new(
    c4a_wavelet_family_t family,
    c4a_wavelet_mode_t   mode,
    int32_t              max_level,
    double               n_components_param);

void c4a_pp_wavelet_svd_state_free(c4a_pp_wavelet_svd_state_t* state);

int c4a_pp_wavelet_svd_state_is_fitted(
    const c4a_pp_wavelet_svd_state_t* state);

int64_t c4a_pp_wavelet_svd_state_n_components(
    const c4a_pp_wavelet_svd_state_t* state);

c4a_status_t c4a_pp_wavelet_svd_state_fit(
    c4a_pp_wavelet_svd_state_t* state,
    const double* X, int64_t rows, int64_t cols);

c4a_status_t c4a_pp_wavelet_svd_state_apply(
    const c4a_pp_wavelet_svd_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_SVD_H */
