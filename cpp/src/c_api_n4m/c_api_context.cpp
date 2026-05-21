// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_context_t. Every wrapper has a try/catch
// around its full body so no C++ exception ever crosses the boundary.
// Failed calls on a non-NULL context write a message to last_error.

#include <stddef.h>
#include <stdint.h>

#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/common/context.hpp"

namespace {

inline ::n4m::core::Context* as_core(n4m_context_t* ctx) noexcept {
    return static_cast<::n4m::core::Context*>(ctx);
}

inline const ::n4m::core::Context* as_core(const n4m_context_t* ctx) noexcept {
    return static_cast<const ::n4m::core::Context*>(ctx);
}

// Mutable cast used only to write an error message on a const-ish path
// (a getter that wants to report a null out-pointer).
inline ::n4m::core::Context* as_core_mut(const n4m_context_t* ctx) noexcept {
    return const_cast<::n4m::core::Context*>(as_core(ctx));
}

constexpr const char* kEmptyError = "";

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_context_create(n4m_context_t** out_ctx) {
    if (out_ctx == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_ctx = new n4m_context_s{};
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out_ctx = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_ctx = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_context_destroy(n4m_context_t* ctx) {
    try {
        delete ctx;
    } catch (...) {
        // Best-effort cleanup — never propagate.
    }
}

N4M_API n4m_status_t n4m_context_set_seed(n4m_context_t* ctx, uint64_t seed) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        as_core(ctx)->set_seed(seed);
        return N4M_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in n4m_context_set_seed");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_context_get_seed(const n4m_context_t* ctx,
                                           uint64_t* out_seed) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    if (out_seed == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in n4m_context_get_seed");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_seed = as_core(ctx)->seed();
        return N4M_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in n4m_context_get_seed");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_context_set_backend(n4m_context_t* ctx,
                                              n4m_backend_t backend) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        return as_core(ctx)->set_backend(backend);
    } catch (...) {
        as_core(ctx)->set_error("internal error in n4m_context_set_backend");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_context_get_backend(const n4m_context_t* ctx,
                                              n4m_backend_t* out_backend) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    if (out_backend == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in n4m_context_get_backend");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_backend = as_core(ctx)->backend();
        return N4M_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in n4m_context_get_backend");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_context_set_num_threads(n4m_context_t* ctx,
                                                  int32_t n_threads) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    // n_threads <= 0 is documented to mean "library default" — see n4m.h §6.
    // Any int32_t value is accepted.
    try {
        as_core(ctx)->set_num_threads(n_threads);
        return N4M_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in n4m_context_set_num_threads");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_context_get_num_threads(const n4m_context_t* ctx,
                                                  int32_t* out_threads) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    if (out_threads == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in n4m_context_get_num_threads");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_threads = as_core(ctx)->num_threads();
        return N4M_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in n4m_context_get_num_threads");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API const char* n4m_context_last_error(const n4m_context_t* ctx) {
    if (ctx == nullptr) return kEmptyError;
    try {
        return as_core(ctx)->last_error();
    } catch (...) {
        return kEmptyError;
    }
}

N4M_API void n4m_context_clear_error(n4m_context_t* ctx) {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->clear_error();
    } catch (...) {
        // Best-effort — clearing should never throw, but be safe.
    }
}

N4M_API n4m_status_t n4m_context_set_user_data(n4m_context_t* ctx, void* user) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        as_core(ctx)->set_user_data(user);
        return N4M_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in n4m_context_set_user_data");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void* n4m_context_get_user_data(const n4m_context_t* ctx) {
    if (ctx == nullptr) return nullptr;
    try {
        return as_core(ctx)->user_data();
    } catch (...) {
        return nullptr;
    }
}

}  // extern "C"
