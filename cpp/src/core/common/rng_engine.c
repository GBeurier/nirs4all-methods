/* SPDX-License-Identifier: CECILL-2.1 */
/* Unified RNG dispatch. See rng_engine.h. */
#include "rng_engine.h"

#include <math.h>

/* Canonical splitmix64 — MUST match the file-local generators in the selector
 * kernels (golden 0x9E3779B97F4A7C15, Stafford mix13). Frozen by
 * test_rng_engine.cpp so swapping a kernel to this dispatch is byte-identical. */
static uint64_t sm_next(uint64_t* state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void n4m_rng_seed(n4m_rng* rng, n4m_rng_kind kind, uint64_t seed) {
    rng->kind = kind;
    switch (kind) {
        case N4M_RNGK_SPLITMIX64: rng->sm_state = seed; break;
        case N4M_RNGK_PCG64:      n4m_pcg64_engine_seed(&rng->pcg, seed); break;
        case N4M_RNGK_MT_R:       n4m_rng_mt_r_set_seed(&rng->mt_r, (uint32_t)seed); break;
        case N4M_RNGK_NUMPY_MT:   n4m_rng_numpy_mt_seed(&rng->np_mt, seed); break;
    }
}

uint64_t n4m_rng_next_u64(n4m_rng* rng) {
    switch (rng->kind) {
        case N4M_RNGK_SPLITMIX64: return sm_next(&rng->sm_state);
        case N4M_RNGK_PCG64:      return n4m_pcg64_engine_next_uint64(&rng->pcg);
        case N4M_RNGK_MT_R: {
            uint64_t hi = n4m_rng_mt_r_next_uint32(&rng->mt_r);
            uint64_t lo = n4m_rng_mt_r_next_uint32(&rng->mt_r);
            return (hi << 32) | lo;
        }
        case N4M_RNGK_NUMPY_MT: {
            uint64_t hi = n4m_rng_numpy_mt_next_uint32(&rng->np_mt);
            uint64_t lo = n4m_rng_numpy_mt_next_uint32(&rng->np_mt);
            return (hi << 32) | lo;
        }
    }
    return 0;
}

double n4m_rng_next_double(n4m_rng* rng) {
    switch (rng->kind) {
        case N4M_RNGK_SPLITMIX64:
            /* kDoubleUnit = 2^-53; matches the kernels' uniform mapping. */
            return (double)(sm_next(&rng->sm_state) >> 11) * (1.0 / 9007199254740992.0);
        case N4M_RNGK_PCG64:    return n4m_pcg64_engine_next_double(&rng->pcg);
        case N4M_RNGK_MT_R:     return n4m_rng_mt_r_unif(&rng->mt_r);
        case N4M_RNGK_NUMPY_MT: return n4m_rng_numpy_mt_double(&rng->np_mt);
    }
    return 0.0;
}

/* Polar Box-Muller on the engine's own [0,1) doubles — used for SPLITMIX64 and
 * PCG64 normals (R/numpy use their library-exact normal samplers instead). */
static double polar_box_muller(n4m_rng* rng) {
    double x1, x2, r2;
    do {
        x1 = 2.0 * n4m_rng_next_double(rng) - 1.0;
        x2 = 2.0 * n4m_rng_next_double(rng) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    return x2 * sqrt(-2.0 * log(r2) / r2);
}

double n4m_rng_next_normal(n4m_rng* rng) {
    switch (rng->kind) {
        case N4M_RNGK_MT_R:     return n4m_rng_mt_r_norm(&rng->mt_r);
        case N4M_RNGK_NUMPY_MT: return n4m_rng_numpy_mt_normal(&rng->np_mt);
        case N4M_RNGK_SPLITMIX64:
        case N4M_RNGK_PCG64:    return polar_box_muller(rng);
    }
    return 0.0;
}
