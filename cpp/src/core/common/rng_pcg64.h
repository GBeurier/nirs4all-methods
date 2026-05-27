/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal pure-C PCG64 random number generator for n4m.
 *
 * This header is internal-only — never installed and never included by
 * `n4m/n4m.h`. The public C ABI lives in `n4m.h` and is
 * implemented in `c_api/c_api_rng.cpp` which wraps this engine.
 *
 * Algorithm: PCG-XSL-RR 128/64, bit-exact against NumPy 1.26.4's
 * `numpy.random.PCG64`. The state struct shape (state-then-inc) and the
 * SeedSequence-based expansion of a single uint64 seed are designed to
 * match `np.random.default_rng(seed)`'s output stream exactly.
 *
 * Vendored / adapted from:
 *   numpy/random/src/pcg64/pcg64.{h,c}  (MIT — Melissa O'Neill, Robert Kern)
 *   numpy/random/bit_generator.pyx       (BSD-3-Clause — SeedSequence mixer)
 *
 * Both upstream licenses are compatible with CeCILL-2.1.
 */
#ifndef N4M_CORE_COMMON_RNG_PCG64_H
#define N4M_CORE_COMMON_RNG_PCG64_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * 128-bit unsigned integer abstraction.
 *
 * GCC/clang on LP64 platforms support __uint128_t directly. On MSVC we fall
 * back to a manual high/low pair with software-emulated arithmetic. The
 * arithmetic is wrapped in `n4m_pcg128_t` so the rest of the engine doesn't
 * care which path is active.
 * ------------------------------------------------------------------------- */
#if defined(__SIZEOF_INT128__) && !defined(N4M_PCG64_FORCE_EMULATED_128)
#  define N4M_PCG64_HAS_NATIVE_128 1
typedef __uint128_t n4m_pcg128_t;
#else
#  define N4M_PCG64_HAS_NATIVE_128 0
typedef struct n4m_pcg128_t {
    uint64_t high;
    uint64_t low;
} n4m_pcg128_t;
#endif

/* ---------------------------------------------------------------------------
 * PCG64 state.
 *
 * Mirrors NumPy's `pcg_state_setseq_128` (state-then-inc, both 128-bit) plus
 * the `has_uint32` / `uinteger` half-word cache. The cache is needed for
 * bit-exact parity with NumPy's bounded-integer sampler (`Generator.integers`
 * / `choice`), which draws 32-bit words via `next_uint32` (low half of a
 * uint64 first, high half cached for the next call). `next_uint64` /
 * `next_double` do NOT touch the cache, matching NumPy, so existing
 * double/uint64-based augmenters are unaffected.
 * ------------------------------------------------------------------------- */
typedef struct n4m_rng_pcg64 {
    n4m_pcg128_t state;
    n4m_pcg128_t inc;
    int          has_uint32;       /* 1 if `buffered_uint32` holds a draw */
    uint32_t     buffered_uint32;  /* cached high half of the last uint64 */
} n4m_rng_pcg64;

/* ---------------------------------------------------------------------------
 * Lifecycle / output / advance.
 *
 * All functions are internal (default `static` visibility in the shared lib).
 * They are declared here so the C TU that defines them and the C++ TU that
 * wraps them see the same prototypes.
 * ------------------------------------------------------------------------- */

/* Seed the engine from a single uint64 entropy source, expanded via NumPy's
 * SeedSequence (pool_size=4) into the 128-bit state and 128-bit inc. The
 * resulting stream is bit-equivalent to `np.random.default_rng(seed)`. */
void n4m_pcg64_engine_seed(n4m_rng_pcg64* rng, uint64_t seed);

/* Draw one uint64. Matches NumPy's `bit_generator.random_raw()`. Does not
 * touch the uint32 half-word cache. */
uint64_t n4m_pcg64_engine_next_uint64(n4m_rng_pcg64* rng);

/* Draw one uint32 via NumPy's PCG64 `next_uint32`: return the low 32 bits of a
 * fresh uint64 and cache the high 32 for the next call. Reseeding clears the
 * cache. */
uint32_t n4m_pcg64_engine_next_uint32(n4m_rng_pcg64* rng);

/* Draw a uniform double in [0, 1) — `(uint64 >> 11) * 2^-53`. Matches
 * NumPy's `next_double` callback for the PCG64 bit generator. */
double n4m_pcg64_engine_next_double(n4m_rng_pcg64* rng);

/* Advance the state by `delta` steps in O(log delta). The delta is split
 * into a (high, low) 128-bit value via `delta_high` / `delta_low`; the
 * 7-arg public ABI passes a single uint64 (high=0) for now. */
void n4m_pcg64_engine_advance(n4m_rng_pcg64* rng, uint64_t delta_high,
                            uint64_t delta_low);

/* Fill `out[0..n)` with standard-normal samples via the Ziggurat algorithm.
 * Implementation lives in `ziggurat.c`; declared here so both `ziggurat.c`
 * (the definer) and `c_api_rng.cpp` (the wrapper) agree on the prototype. */
void n4m_pcg64_engine_standard_normal_fill(n4m_rng_pcg64* rng, double* out,
                                         size_t n);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_RNG_PCG64_H */
