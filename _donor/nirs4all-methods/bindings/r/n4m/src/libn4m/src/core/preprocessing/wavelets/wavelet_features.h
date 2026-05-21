/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the WaveletFeatures operator (Phase 6).
 *
 * Performs a multi-level DWT and extracts four statistical features per
 * decomposition band:
 *
 *   - mean    : arithmetic mean
 *   - std     : population standard deviation (ddof=0, ``np.std`` default)
 *   - energy  : sum of squares
 *   - entropy : selectable. ENERGY uses the squared-coefficient
 *                 distribution; HISTOGRAM uses 10 equal-width bins and
 *                 scipy.stats.entropy-compatible normalization.
 *
 * For ``level = K`` the output has ``4 * (K + 1)`` columns per row, laid
 * out as ``[approx_stats, d_K_stats, d_{K-1}_stats, ..., d_1_stats]``
 * matching the PyWavelets ``wavedec`` ordering.  ``level`` is clamped to
 * the maximum allowed by ``pywt.dwt_max_level`` for the given (family,
 * n) pair.  ``output_cols`` should be queried via the helper.
 *
 * Pure C, no dependencies.  INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_PP_WAVELETS_WAVELET_FEATURES_H
#define N4M_CORE_PP_WAVELETS_WAVELET_FEATURES_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_wavelet_features_state_t n4m_pp_wavelet_features_state_t;

n4m_pp_wavelet_features_state_t* n4m_pp_wavelet_features_state_new(
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t   mode,
    int32_t              max_level);

n4m_pp_wavelet_features_state_t* n4m_pp_wavelet_features_state_new_ex(
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t   mode,
    int32_t              max_level,
    n4m_pp_wavelet_features_entropy_t entropy_mode);

void n4m_pp_wavelet_features_state_free(n4m_pp_wavelet_features_state_t* state);

/* Compute the actual (clamped) decomposition level for an input length.
 * Returns the smaller of the constructor's ``max_level`` and
 * ``pywt.dwt_max_level(cols, family)``. */
int32_t n4m_pp_wavelet_features_actual_level(
    const n4m_pp_wavelet_features_state_t* state, int64_t cols);

/* Compute the number of output features per row: ``4 * (actual_level + 1)``. */
int64_t n4m_pp_wavelet_features_output_size(
    const n4m_pp_wavelet_features_state_t* state, int64_t cols);

n4m_status_t n4m_pp_wavelet_features_state_apply(
    const n4m_pp_wavelet_features_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_WAVELETS_WAVELET_FEATURES_H */
