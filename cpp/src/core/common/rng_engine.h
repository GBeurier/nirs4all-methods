/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Unified internal RNG dispatch for n4m's stochastic methods.
 *
 * Internal-only (never installed, never in n4m.h). It unifies the engines a
 * stochastic method may draw from, selected by `n4m_rng_kind`:
 *   - N4M_RNGK_SPLITMIX64 : n4m's own historical stream (the default; every
 *                           selector used a file-local splitmix64). The engine
 *                           reproduces that exact sequence so swapping a method
 *                           from its file-local helper to this dispatch is
 *                           byte-identical (frozen by test_rng_engine.cpp).
 *   - N4M_RNGK_PCG64      : ≡ numpy.random.default_rng (wraps rng_pcg64).
 *   - N4M_RNGK_MT_R       : ≡ base R set.seed/runif/rnorm (wraps rng_mt_r).
 *   - N4M_RNGK_NUMPY_MT   : ≡ numpy.random.RandomState (legacy MT19937), the
 *                           RNG sklearn check_random_state / auswahl use
 *                           (wraps rng_numpy_mt).
 *
 * The public, ABI-visible selector is `n4m_rng_kind_t` in n4m/pls.h + the
 * config setter `n4m_config_set_rng_kind`; this internal enum mirrors it.
 */
#ifndef N4M_CORE_COMMON_RNG_ENGINE_H
#define N4M_CORE_COMMON_RNG_ENGINE_H

#include <stdint.h>

#include "rng_pcg64.h"
#include "rng_mt_r.h"
#include "rng_numpy_mt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_rng_kind {
    N4M_RNGK_SPLITMIX64 = 0,
    N4M_RNGK_PCG64      = 1,
    N4M_RNGK_MT_R       = 2,
    N4M_RNGK_NUMPY_MT   = 3
} n4m_rng_kind;

/* All engine states embedded; only the one matching `kind` is live. The
 * footprint is dominated by mt_r/numpy_mt (~2.5 KB each); stochastic methods
 * allocate one of these per fit, so the size is irrelevant. */
typedef struct n4m_rng {
    n4m_rng_kind        kind;
    uint64_t            sm_state;   /* splitmix64 running state */
    n4m_rng_pcg64       pcg;
    n4m_rng_mt_r        mt_r;
    n4m_rng_numpy_mt    np_mt;
} n4m_rng;

/* Seed `rng` for the given engine. For SPLITMIX64 the seed is the initial
 * state (matching the kernels' `std::uint64_t state = seed`). */
void   n4m_rng_seed(n4m_rng* rng, n4m_rng_kind kind, uint64_t seed);

/* One uint64 from the active engine.
 *   SPLITMIX64 : canonical splitmix64 step (golden 0x9E3779B97F4A7C15).
 *   PCG64      : n4m_pcg64_engine_next_uint64.
 *   MT_R       : 64-bit assembled from two R MT uint32 (hi<<32|lo) — provided
 *                for completeness; selectors that target R use unif/normal.
 *   NUMPY_MT   : 64-bit from two RandomState uint32.
 */
uint64_t n4m_rng_next_u64(n4m_rng* rng);

/* Uniform double in [0,1). Per-engine the canonical mapping:
 *   SPLITMIX64 : (u64 >> 11) * 2^-53   (matches the kernels' kDoubleUnit).
 *   PCG64      : n4m_pcg64_engine_next_double.
 *   MT_R       : R unif_rand (1/2^32 scaled, fixup'd).
 *   NUMPY_MT   : RandomState random_sample (53-bit).
 */
double n4m_rng_next_double(n4m_rng* rng);

/* Standard normal from the active engine (Ziggurat for splitmix/pcg via the
 * pcg path is NOT used; MT_R uses R Inversion, NUMPY_MT uses RandomState
 * Box-Muller/gauss). SPLITMIX64/PCG64 use the existing ziggurat fill. */
double n4m_rng_next_normal(n4m_rng* rng);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_RNG_ENGINE_H */
