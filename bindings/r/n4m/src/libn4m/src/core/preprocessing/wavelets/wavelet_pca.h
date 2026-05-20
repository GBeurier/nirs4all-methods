/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the WaveletPCA operator (Phase 6).
 *
 * Pipeline: each training row is run through a multi-level DWT and the
 * resulting coefficient stack is flattened into a single feature
 * vector.  The full coefficient matrix (n_samples x flat_dim) is then
 * fed to the same PCA path used by FlexiblePCA (centering + Jacobi SVD
 * via ``core/common/svd.c``), with ``n_components`` resolved through
 * the FlexiblePCA "flexible" specifier (integer count >= 1 OR variance
 * ratio in (0, 1)).
 *
 * Transform: same DWT-flatten step on each incoming row, then project
 * through the learned PCA components.
 *
 * Pure C, no dependencies.  INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_PP_WAVELETS_WAVELET_PCA_H
#define N4M_CORE_PP_WAVELETS_WAVELET_PCA_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_wavelet_pca_state_t n4m_pp_wavelet_pca_state_t;

n4m_pp_wavelet_pca_state_t* n4m_pp_wavelet_pca_state_new(
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t   mode,
    int32_t              max_level,
    double               n_components_param);

void n4m_pp_wavelet_pca_state_free(n4m_pp_wavelet_pca_state_t* state);

int n4m_pp_wavelet_pca_state_is_fitted(
    const n4m_pp_wavelet_pca_state_t* state);

int64_t n4m_pp_wavelet_pca_state_n_components(
    const n4m_pp_wavelet_pca_state_t* state);

n4m_status_t n4m_pp_wavelet_pca_state_fit(
    n4m_pp_wavelet_pca_state_t* state,
    const double* X, int64_t rows, int64_t cols);

n4m_status_t n4m_pp_wavelet_pca_state_apply(
    const n4m_pp_wavelet_pca_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_WAVELETS_WAVELET_PCA_H */
