/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * PolynomialBaselineDrift — Phase 15 augmenter (c4a_aug_poly_drift_*).
 *
 * Adds a per-sample polynomial baseline of configurable degree:
 *
 *     coeffs[i, k] = lo_k + (hi_k - lo_k) * u_ik         # u_ik ~ U[0,1)
 *     drift[i, j]  = sum_{k=0}^{degree} coeffs[i, k] * lambda_j^k
 *     out[i, j]    = X[i, j] + drift[i, j]
 *
 * The wavelength axis defaults to `linspace(-1, 1, n_features)` so polynomial
 * powers stay numerically tame (the same normalization nirs4all applies in
 * the implicit-wavelengths branch).
 *
 * `coeff_min[k]` / `coeff_max[k]` arrays have length `degree + 1`. They are
 * copied into the state — caller owns the originals.
 *
 * RNG draw order (locked): the (rows * (degree + 1)) coefficient draws are
 * iterated **per order, then per row** to mirror the
 * `for k: rng.uniform(lo_k, hi_k, size=n_samples)` pattern in
 * `nirs4all.operators.augmentation.spectral.PolynomialBaselineDrift`.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_DRIFT_POLY_DRIFT_H
#define CHEMOMETRICS4ALL_CORE_AUG_DRIFT_POLY_DRIFT_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_poly_drift_state_t c4a_aug_poly_drift_state_t;

c4a_aug_poly_drift_state_t* c4a_aug_poly_drift_state_new(
    c4a_rng_pcg64* rng,
    int32_t degree,
    const double* coeff_min,
    const double* coeff_max);

void c4a_aug_poly_drift_state_free(c4a_aug_poly_drift_state_t* state);

c4a_status_t c4a_aug_poly_drift_state_apply(
    c4a_aug_poly_drift_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_DRIFT_POLY_DRIFT_H */
