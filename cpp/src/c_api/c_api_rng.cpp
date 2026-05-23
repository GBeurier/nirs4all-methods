// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the PCG64 random number generator. Every wrapper
// has a try/catch around its full body so no C++ exception ever crosses
// the boundary. The opaque handle `n4m_rng_pcg64_state_t` owns the
// internal C engine (`n4m_rng_pcg64` from core/common/rng_pcg64.h).
//
// Phase 1 only — all subsequent stochastic operators (augmenters,
// IsolationForest, KMeans, CARS, MCUVE) consume this handle to obtain
// bit-exact NumPy parity.

#include <stddef.h>
#include <stdint.h>

#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

#include "rng_state_internal.hpp"

extern "C" {

N4M_API n4m_status_t n4m_rng_pcg64_create(uint64_t seed,
                                           n4m_rng_pcg64_state_t** out) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        n4m_rng_pcg64_state_t* handle = new n4m_rng_pcg64_state_t{};
        n4m_pcg64_engine_seed(&handle->engine, seed);
        *out = handle;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_rng_pcg64_destroy(n4m_rng_pcg64_state_t* rng) {
    try {
        delete rng;
    } catch (...) {
        // Best-effort cleanup — never propagate.
    }
}

N4M_API n4m_status_t n4m_rng_pcg64_set_seed(n4m_rng_pcg64_state_t* rng,
                                             uint64_t seed) {
    if (rng == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        n4m_pcg64_engine_seed(&rng->engine, seed);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_rng_pcg64_uint64(n4m_rng_pcg64_state_t* rng,
                                           uint64_t* out) {
    if (rng == nullptr || out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out = n4m_pcg64_engine_next_uint64(&rng->engine);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_rng_pcg64_uint64_fill(n4m_rng_pcg64_state_t* rng,
                                                uint64_t* out, size_t n) {
    if (rng == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n > 0 && out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        for (size_t i = 0; i < n; ++i) {
            out[i] = n4m_pcg64_engine_next_uint64(&rng->engine);
        }
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_rng_pcg64_standard_normal_fill(
    n4m_rng_pcg64_state_t* rng, double* out, size_t n) {
    if (rng == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n > 0 && out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        n4m_pcg64_engine_standard_normal_fill(&rng->engine, out, n);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_rng_pcg64_advance(n4m_rng_pcg64_state_t* rng,
                                            uint64_t delta) {
    if (rng == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        n4m_pcg64_engine_advance(&rng->engine, 0u, delta);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
