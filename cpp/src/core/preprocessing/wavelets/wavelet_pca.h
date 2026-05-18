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
#ifndef CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_PCA_H
#define CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_PCA_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_wavelet_pca_state_t c4a_pp_wavelet_pca_state_t;

c4a_pp_wavelet_pca_state_t* c4a_pp_wavelet_pca_state_new(
    c4a_wavelet_family_t family,
    c4a_wavelet_mode_t   mode,
    int32_t              max_level,
    double               n_components_param);

void c4a_pp_wavelet_pca_state_free(c4a_pp_wavelet_pca_state_t* state);

int c4a_pp_wavelet_pca_state_is_fitted(
    const c4a_pp_wavelet_pca_state_t* state);

int64_t c4a_pp_wavelet_pca_state_n_components(
    const c4a_pp_wavelet_pca_state_t* state);

c4a_status_t c4a_pp_wavelet_pca_state_fit(
    c4a_pp_wavelet_pca_state_t* state,
    const double* X, int64_t rows, int64_t cols);

c4a_status_t c4a_pp_wavelet_pca_state_apply(
    const c4a_pp_wavelet_pca_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_PCA_H */
