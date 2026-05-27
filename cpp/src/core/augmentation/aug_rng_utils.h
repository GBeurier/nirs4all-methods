/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared RNG primitives for the n4m_aug_* augmenters (Phases 15-18).
 *
 * The augmenter ABI requires the caller to supply a PCG64 handle that
 * advances per call. To implement the per-augmenter operations we expose:
 *
 *   - n4m_aug_rng_uniform        : NumPy-style uniform in [low, high)
 *   - n4m_aug_rng_uniform_fill   : fill an array
 *   - n4m_aug_rng_normal         : single normal draw (with scaling)
 *   - n4m_aug_rng_normal_fill    : Ziggurat-backed scaled fill
 *   - n4m_aug_rng_integers       : NumPy-style integer in [0, n)
 *   - n4m_aug_rng_beta           : Cheng's BB / BC for Beta(a, b)
 *   - n4m_aug_rng_permutation    : Fisher-Yates permutation of [0, n)
 *
 * Implemented in aug_rng_utils.c. Internal to the core; not part of the
 * exported public ABI surface.
 */
#ifndef N4M_CORE_AUG_RNG_UTILS_H
#define N4M_CORE_AUG_RNG_UTILS_H

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
double n4m_aug_rng_uniform(n4m_rng_pcg64* rng, double low, double high);

/* Fills out[0..n) with NumPy-style uniform[low, high) values. */
void n4m_aug_rng_uniform_fill(n4m_rng_pcg64* rng,
                              double low, double high,
                              double* out, size_t n);

/* Fills out[0..n) with `loc + scale * standard_normal` values, advancing
 * the RNG by n Ziggurat draws. Matches NumPy's
 * `Generator.normal(loc, scale, size=n)`. */
void n4m_aug_rng_normal_fill(n4m_rng_pcg64* rng,
                             double loc, double scale,
                             double* out, size_t n);

/* Returns one integer in [0, n), bit-for-bit matching NumPy's
 * `Generator.integers(0, n)` (it delegates to n4m_aug_rng_bounded, the buffered
 * uint32 Lemire sampler). Requires n - 1 < 2^32 (always true for spectra). */
uint64_t n4m_aug_rng_integers(n4m_rng_pcg64* rng, uint64_t n);

/* Returns one integer in [0, rng_incl] (inclusive), bit-for-bit matching
 * NumPy's bounded sampler for ranges that fit in 32 bits — the buffered
 * uint32 Lemire method (`(next_uint32() * (rng_incl + 1)) >> 32` with the
 * documented rejection). This is what `Generator.integers(0, rng_incl,
 * endpoint=True)` and `Generator.choice(..., replace=False)` actually use, so
 * it (not n4m_aug_rng_integers) is the parity-correct primitive. Requires
 * rng_incl < 2^32 (always true for spectral feature counts). */
uint64_t n4m_aug_rng_bounded(n4m_rng_pcg64* rng, uint64_t rng_incl);

/* Returns one Beta(a, b) draw matching NumPy's `Generator.beta(a, b)`.
 * The a <= 1 and b <= 1 path intentionally uses Joehnk rejection even for
 * a == b == 1 to match NumPy's RNG state advancement and output sequence. */
double n4m_aug_rng_beta(n4m_rng_pcg64* rng, double a, double b);

/* In-place uniform Fisher-Yates permutation of [0, n) over the parity-correct
 * bounded sampler (n4m_aug_rng_integers). NOTE: this is a correct uniform
 * shuffle but is NOT bit-identical to numpy.Generator.permutation, which uses a
 * different internal shuffle — do not rely on numpy cross-parity here. */
void n4m_aug_rng_permutation(n4m_rng_pcg64* rng, int64_t* out, int64_t n);

/* Draw `size` distinct indices in [0, pop_size) without replacement, writing
 * them to out[0..size). Bit-for-bit reproduces NumPy's
 * `Generator.choice(pop_size, size, replace=False, p=None)` (shuffle=True):
 * Floyd's open-addressing algorithm for small/medium populations and the
 * tail-shuffle path for pop_size > 10000 with size > pop_size/50, followed by
 * the same final partial Fisher-Yates. Requires 0 <= size <= pop_size.
 * Returns 0 on success, non-zero on bad arguments or allocation failure. */
int n4m_aug_rng_choice_no_replace(n4m_rng_pcg64* rng, int64_t pop_size,
                                  int64_t size, int64_t* out);

/* Returns a single standard_normal draw (Ziggurat). Equivalent to
 * n4m_pcg64_engine_standard_normal_fill with n=1. */
double n4m_aug_rng_standard_normal(n4m_rng_pcg64* rng);

/* Returns one uint64 draw — convenience wrapper. */
uint64_t n4m_aug_rng_uint64(n4m_rng_pcg64* rng);

/* Returns one [0, 1) double — convenience wrapper. */
double n4m_aug_rng_next_double(n4m_rng_pcg64* rng);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_RNG_UTILS_H */
