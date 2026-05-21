/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the WaveletSVD operator (Phase 6).
 *
 * Same DWT-flatten pipeline as WaveletPCA, but the projection step
 * uses Truncated SVD (no mean subtraction).  ``transform`` returns the
 * scores ``X @ Vt.T``, matching ``sklearn.decomposition.TruncatedSVD``
 * (also identical to the nirs4all-methods FlexibleSVD operator).
 *
 * Pure C, no dependencies.  INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_PP_WAVELETS_WAVELET_SVD_H
#define N4M_CORE_PP_WAVELETS_WAVELET_SVD_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_wavelet_svd_state_t n4m_pp_wavelet_svd_state_t;

n4m_pp_wavelet_svd_state_t* n4m_pp_wavelet_svd_state_new(
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t   mode,
    int32_t              max_level,
    double               n_components_param);

void n4m_pp_wavelet_svd_state_free(n4m_pp_wavelet_svd_state_t* state);

int n4m_pp_wavelet_svd_state_is_fitted(
    const n4m_pp_wavelet_svd_state_t* state);

int64_t n4m_pp_wavelet_svd_state_n_components(
    const n4m_pp_wavelet_svd_state_t* state);

n4m_status_t n4m_pp_wavelet_svd_state_fit(
    n4m_pp_wavelet_svd_state_t* state,
    const double* X, int64_t rows, int64_t cols);

n4m_status_t n4m_pp_wavelet_svd_state_apply(
    const n4m_pp_wavelet_svd_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_WAVELETS_WAVELET_SVD_H */
