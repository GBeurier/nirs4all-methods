/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SmoothMagnitudeWarp augmenter — multiplies each spectrum by a smooth
 * per-sample gain curve defined by `n_control_points` random gain values
 * interpolated to all wavelengths.
 *
 * Algorithm (linear-interp parity floor; the original nirs4all reference
 * uses scipy splrep/splev cubic, but parity to numpy is locked to the
 * deterministic np.interp path):
 *
 *   ctrl_x  = np.linspace(lambdas[0], lambdas[-1], n_control_points)
 *   for each sample i:
 *     ctrl_y  = rng.uniform(gain_lo, gain_hi, size=n_control_points)
 *     gains   = np.interp(lambdas, ctrl_x, ctrl_y)
 *     out[i]  = X[i] * gains
 *
 * The Python reference under c4a_parity_spectral_aug_ref enforces the same
 * linear path so the C engine matches at the 1e-15 PCG64 contract.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_MAGNITUDE_WARP_H
#define CHEMOMETRICS4ALL_CORE_AUG_MAGNITUDE_WARP_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_magnitude_warp_state_t c4a_aug_magnitude_warp_state_t;

c4a_aug_magnitude_warp_state_t* c4a_aug_magnitude_warp_state_new(
    c4a_rng_pcg64* rng,
    int32_t n_control_points,
    double gain_lo, double gain_hi,
    const double* wavelengths, int64_t n_wavelengths);

void c4a_aug_magnitude_warp_state_free(c4a_aug_magnitude_warp_state_t* state);

c4a_status_t c4a_aug_magnitude_warp_state_apply(
    c4a_aug_magnitude_warp_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_MAGNITUDE_WARP_H */
