/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal pure-C PCG64 random number generator for chemometrics4all.
 *
 * This header is internal-only — never installed and never included by
 * `chemometrics4all/c4a.h`. The public C ABI lives in `c4a.h` and is
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
#ifndef CHEMOMETRICS4ALL_CORE_COMMON_RNG_PCG64_H
#define CHEMOMETRICS4ALL_CORE_COMMON_RNG_PCG64_H

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
 * arithmetic is wrapped in `c4a_pcg128_t` so the rest of the engine doesn't
 * care which path is active.
 * ------------------------------------------------------------------------- */
#if defined(__SIZEOF_INT128__) && !defined(C4A_PCG64_FORCE_EMULATED_128)
#  define C4A_PCG64_HAS_NATIVE_128 1
typedef __uint128_t c4a_pcg128_t;
#else
#  define C4A_PCG64_HAS_NATIVE_128 0
typedef struct c4a_pcg128_t {
    uint64_t high;
    uint64_t low;
} c4a_pcg128_t;
#endif

/* ---------------------------------------------------------------------------
 * PCG64 state.
 *
 * Mirrors NumPy's `pcg_state_setseq_128` (state-then-inc, both 128-bit). We
 * do NOT carry NumPy's `has_uint32` / `uinteger` half-word cache — our
 * Generator-level API never asks for 32-bit draws, so the cache would only
 * waste 8 bytes and add bookkeeping risk.
 * ------------------------------------------------------------------------- */
typedef struct c4a_rng_pcg64 {
    c4a_pcg128_t state;
    c4a_pcg128_t inc;
} c4a_rng_pcg64;

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
void c4a_pcg64_engine_seed(c4a_rng_pcg64* rng, uint64_t seed);

/* Draw one uint64. Matches NumPy's `bit_generator.random_raw()`. */
uint64_t c4a_pcg64_engine_next_uint64(c4a_rng_pcg64* rng);

/* Draw a uniform double in [0, 1) — `(uint64 >> 11) * 2^-53`. Matches
 * NumPy's `next_double` callback for the PCG64 bit generator. */
double c4a_pcg64_engine_next_double(c4a_rng_pcg64* rng);

/* Advance the state by `delta` steps in O(log delta). The delta is split
 * into a (high, low) 128-bit value via `delta_high` / `delta_low`; the
 * 7-arg public ABI passes a single uint64 (high=0) for now. */
void c4a_pcg64_engine_advance(c4a_rng_pcg64* rng, uint64_t delta_high,
                            uint64_t delta_low);

/* Fill `out[0..n)` with standard-normal samples via the Ziggurat algorithm.
 * Implementation lives in `ziggurat.c`; declared here so both `ziggurat.c`
 * (the definer) and `c_api_rng.cpp` (the wrapper) agree on the prototype. */
void c4a_pcg64_engine_standard_normal_fill(c4a_rng_pcg64* rng, double* out,
                                         size_t n);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_COMMON_RNG_PCG64_H */
