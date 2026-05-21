/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Gaussian operator. Public C ABI lives in n4m.h
 * (§10, Phase 4).
 *
 * Algorithm: `scipy.ndimage.gaussian_filter1d(X, sigma, order, axis=1,
 * mode, cval, truncate)`.  Kernel radius is `lw = int(truncate * sigma + 0.5)`
 * and the kernel has length `2*lw + 1`.  Build via the Hermite-polynomial
 * recursion that matches `_gaussian_kernel1d` bit-for-bit, then reverse and
 * apply as a cross-correlation (matching scipy's correlate1d semantics).
 *
 * Five boundary modes (reflect, constant, nearest, mirror, wrap) come from
 * the shared n4m_pad_resolve_index helper.
 *
 * Lifecycle: stateless.  Kernel is precomputed in `_create`.
 */
#ifndef N4M_CORE_PP_SMOOTHING_GAUSSIAN_H
#define N4M_CORE_PP_SMOOTHING_GAUSSIAN_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_gaussian_state_t n4m_pp_gaussian_state_t;

n4m_pp_gaussian_state_t* n4m_pp_gaussian_state_new(
    double sigma, int32_t order,
    n4m_pp_gaussian_mode_t mode,
    double cval, double truncate);

void n4m_pp_gaussian_state_free(n4m_pp_gaussian_state_t* state);

n4m_status_t n4m_pp_gaussian_state_apply(
    const n4m_pp_gaussian_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SMOOTHING_GAUSSIAN_H */
