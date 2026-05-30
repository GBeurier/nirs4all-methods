/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal numpy-legacy-RandomState-compatible MT19937 RNG for n4m.
 *
 * Internal-only. Reproduces `numpy.random.RandomState(seed)` bit-for-bit — the
 * RNG that `sklearn.utils.check_random_state(int)` and hence the `auswahl`
 * selectors (VISSA / RandomFrog / IntervalRandomFrog / VIP_SPA) draw from.
 * This is DISTINCT from PCG64 (`default_rng`) and from base R's MT.
 *
 * Reproduces:
 *   RandomState(s)             ≡ n4m_rng_numpy_mt_seed(rng, s)   (init_by_array)
 *   rs.random_sample()         ≡ n4m_rng_numpy_mt_double()       (53-bit)
 *   rs.standard_normal()       ≡ n4m_rng_numpy_mt_normal()       (legacy_gauss,
 *                                                                 polar Box-Muller)
 *
 * MT19937 core = Matsumoto & Nishimura reference; seeding = numpy's
 * init_genrand(19650218) + init_by_array(key). Re-implemented from the public
 * algorithm.
 */
#ifndef N4M_CORE_COMMON_RNG_NUMPY_MT_H
#define N4M_CORE_COMMON_RNG_NUMPY_MT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_rng_numpy_mt {
    uint32_t mt[624];
    int      mti;
    int      has_gauss;     /* legacy_gauss one-value cache flag */
    double   gauss;         /* cached second polar Box-Muller value */
} n4m_rng_numpy_mt;

/* RandomState(seed): init_genrand(19650218) then init_by_array with the seed's
 * 32-bit little-endian words (a 32-bit seed → key length 1). Clears the gauss
 * cache, matching a fresh RandomState. */
void   n4m_rng_numpy_mt_seed(n4m_rng_numpy_mt* rng, uint64_t seed);

/* Raw tempered uint32 (Matsumoto genrand_int32). */
uint32_t n4m_rng_numpy_mt_next_uint32(n4m_rng_numpy_mt* rng);

/* rs.random_sample(): 53-bit double in [0,1) — a=genrand>>5, b=genrand>>6,
 * (a*2^26 + b) / 2^53. */
double n4m_rng_numpy_mt_double(n4m_rng_numpy_mt* rng);

/* rs.standard_normal(): numpy legacy_gauss — polar Box-Muller on two
 * random_sample draws, returns f*x2 and caches f*x1. */
double n4m_rng_numpy_mt_normal(n4m_rng_numpy_mt* rng);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_RNG_NUMPY_MT_H */
