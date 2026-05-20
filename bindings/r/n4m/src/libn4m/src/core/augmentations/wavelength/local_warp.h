/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LocalWavelengthWarp augmenter — applies a non-linear shift profile defined
 * by `n_control_points` random shifts, linearly interpolated to all
 * wavelengths, then linear-interpolates the spectrum at the warped axis.
 *
 * Algorithm (linear-interp parity floor — DIFFERS from nirs4all's cubic
 * splrep/splev path, which is not bit-exact reproducible without scipy).
 *
 *   ctrl_x = np.linspace(lambdas[0], lambdas[-1], n_control_points)
 *   for each sample i:
 *     ctrl_y = rng.uniform(-max_shift, max_shift, size=n_control_points)
 *     shift[j]  = np.interp(lambdas, ctrl_x, ctrl_y)
 *     query[j]  = lambdas[j] - shift[j]
 *     out[i]    = np.interp(query, lambdas, X[i])
 *
 * The Python reference under n4m_parity_spectral_aug_ref mirrors this exact
 * linear path; the bit-exact parity contract sits at 1e-15 abs with PCG64.
 */
#ifndef N4M_CORE_AUG_LOCAL_WARP_H
#define N4M_CORE_AUG_LOCAL_WARP_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_local_warp_state_t n4m_aug_local_warp_state_t;

n4m_aug_local_warp_state_t* n4m_aug_local_warp_state_new(
    n4m_rng_pcg64* rng,
    int32_t n_control_points,
    double max_shift,
    const double* wavelengths, int64_t n_wavelengths);

void n4m_aug_local_warp_state_free(
    n4m_aug_local_warp_state_t* state);

n4m_status_t n4m_aug_local_warp_state_apply(
    n4m_aug_local_warp_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_LOCAL_WARP_H */
