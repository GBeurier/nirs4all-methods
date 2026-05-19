// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the PCG64 random number generator. Every wrapper
// has a try/catch around its full body so no C++ exception ever crosses
// the boundary. The opaque handle `c4a_rng_pcg64_state_t` owns the
// internal C engine (`c4a_rng_pcg64` from core/common/rng_pcg64.h).
//
// Phase 1 only — all subsequent stochastic operators (augmenters,
// IsolationForest, KMeans, CARS, MCUVE) consume this handle to obtain
// bit-exact NumPy parity.

#include <stddef.h>
#include <stdint.h>

#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#include "rng_state_internal.hpp"

extern "C" {

C4A_API c4a_status_t c4a_rng_pcg64_create(uint64_t seed,
                                           c4a_rng_pcg64_state_t** out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        c4a_rng_pcg64_state_t* handle = new c4a_rng_pcg64_state_t{};
        c4a_pcg64_engine_seed(&handle->engine, seed);
        *out = handle;
        return C4A_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return C4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng) {
    try {
        delete rng;
    } catch (...) {
        // Best-effort cleanup — never propagate.
    }
}

C4A_API c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng,
                                             uint64_t seed) {
    if (rng == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        c4a_pcg64_engine_seed(&rng->engine, seed);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng,
                                           uint64_t* out) {
    if (rng == nullptr || out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out = c4a_pcg64_engine_next_uint64(&rng->engine);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng,
                                                uint64_t* out, size_t n) {
    if (rng == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n > 0 && out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        for (size_t i = 0; i < n; ++i) {
            out[i] = c4a_pcg64_engine_next_uint64(&rng->engine);
        }
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_rng_pcg64_standard_normal_fill(
    c4a_rng_pcg64_state_t* rng, double* out, size_t n) {
    if (rng == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n > 0 && out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        c4a_pcg64_engine_standard_normal_fill(&rng->engine, out, n);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng,
                                            uint64_t delta) {
    if (rng == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        c4a_pcg64_engine_advance(&rng->engine, 0u, delta);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
