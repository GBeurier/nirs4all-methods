/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared RNG primitives for the c4a_aug_* augmenters (Phases 15-18).
 *
 * The augmenter ABI requires the caller to supply a PCG64 handle that
 * advances per call. To implement the per-augmenter operations we expose:
 *
 *   - c4a_aug_rng_uniform        : NumPy-style uniform in [low, high)
 *   - c4a_aug_rng_uniform_fill   : fill an array
 *   - c4a_aug_rng_normal         : single normal draw (with scaling)
 *   - c4a_aug_rng_normal_fill    : Ziggurat-backed scaled fill
 *   - c4a_aug_rng_integers       : NumPy-style integer in [0, n)
 *   - c4a_aug_rng_beta           : Cheng's BB / BC for Beta(a, b)
 *   - c4a_aug_rng_permutation    : Fisher-Yates permutation of [0, n)
 *
 * Implemented in aug_rng_utils.c. Internal to the core; not part of the
 * exported public ABI surface.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_RNG_UTILS_H
#define CHEMOMETRICS4ALL_CORE_AUG_RNG_UTILS_H

#include <stddef.h>
#include <stdint.h>

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returns one uniform double in [low, high). Matches NumPy's
 * `Generator.uniform(low, high)` for a scalar size, i.e.
 *   low + (high - low) * next_double().
 */
double c4a_aug_rng_uniform(c4a_rng_pcg64* rng, double low, double high);

/* Fills out[0..n) with NumPy-style uniform[low, high) values. */
void c4a_aug_rng_uniform_fill(c4a_rng_pcg64* rng,
                              double low, double high,
                              double* out, size_t n);

/* Fills out[0..n) with `loc + scale * standard_normal` values, advancing
 * the RNG by n Ziggurat draws. Matches NumPy's
 * `Generator.normal(loc, scale, size=n)`. */
void c4a_aug_rng_normal_fill(c4a_rng_pcg64* rng,
                             double loc, double scale,
                             double* out, size_t n);

/* Returns one integer in [0, n) drawn via NumPy's masked-rejection
 * `Lemire`-style integer sampler used by `Generator.integers(0, n)`. */
uint64_t c4a_aug_rng_integers(c4a_rng_pcg64* rng, uint64_t n);

/* Returns one Beta(a, b) draw matching NumPy's `Generator.beta(a, b)`.
 * For a == b == 1 this is exactly `next_double()`. */
double c4a_aug_rng_beta(c4a_rng_pcg64* rng, double a, double b);

/* In-place Fisher-Yates permutation. `out[i] = i` initially, then the
 * algorithm swaps. Matches NumPy's
 * `Generator.permutation(n)` for the default `axis=0` shuffling. */
void c4a_aug_rng_permutation(c4a_rng_pcg64* rng, int64_t* out, int64_t n);

/* Returns a single standard_normal draw (Ziggurat). Equivalent to
 * c4a_pcg64_engine_standard_normal_fill with n=1. */
double c4a_aug_rng_standard_normal(c4a_rng_pcg64* rng);

/* Returns one uint64 draw — convenience wrapper. */
uint64_t c4a_aug_rng_uint64(c4a_rng_pcg64* rng);

/* Returns one [0, 1) double — convenience wrapper. */
double c4a_aug_rng_next_double(c4a_rng_pcg64* rng);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_RNG_UTILS_H */
