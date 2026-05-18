// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for p4a_context_t. Every wrapper has a try/catch
// around its full body so no C++ exception ever crosses the boundary.
// Failed calls on a non-NULL context write a message to last_error.

#include <stddef.h>
#include <stdint.h>

#include <exception>
#include <new>

#include "pls4all/p4a.h"

#include "core/context.hpp"

namespace {

inline ::pls4all::core::Context* as_core(p4a_context_t* ctx) noexcept {
    return static_cast<::pls4all::core::Context*>(ctx);
}

inline const ::pls4all::core::Context* as_core(const p4a_context_t* ctx) noexcept {
    return static_cast<const ::pls4all::core::Context*>(ctx);
}

// Mutable cast used only to write an error message on a const-ish path
// (a getter that wants to report a null out-pointer).
inline ::pls4all::core::Context* as_core_mut(const p4a_context_t* ctx) noexcept {
    return const_cast<::pls4all::core::Context*>(as_core(ctx));
}

constexpr const char* kEmptyError = "";

}  // namespace

extern "C" {

P4A_API p4a_status_t p4a_context_create(p4a_context_t** out_ctx) {
    if (out_ctx == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out_ctx = new p4a_context_s{};
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        *out_ctx = nullptr;
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_ctx = nullptr;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_context_destroy(p4a_context_t* ctx) {
    try {
        delete ctx;
    } catch (...) {
        // Best-effort cleanup — never propagate.
    }
}

P4A_API p4a_status_t p4a_context_set_seed(p4a_context_t* ctx, uint64_t seed) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        as_core(ctx)->set_seed(seed);
        return P4A_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in p4a_context_set_seed");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_context_get_seed(const p4a_context_t* ctx,
                                           uint64_t* out_seed) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    if (out_seed == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in p4a_context_get_seed");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out_seed = as_core(ctx)->seed();
        return P4A_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in p4a_context_get_seed");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_context_set_backend(p4a_context_t* ctx,
                                              p4a_backend_t backend) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        return as_core(ctx)->set_backend(backend);
    } catch (...) {
        as_core(ctx)->set_error("internal error in p4a_context_set_backend");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_context_get_backend(const p4a_context_t* ctx,
                                              p4a_backend_t* out_backend) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    if (out_backend == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in p4a_context_get_backend");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out_backend = as_core(ctx)->backend();
        return P4A_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in p4a_context_get_backend");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_context_set_num_threads(p4a_context_t* ctx,
                                                  int32_t n_threads) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    // n_threads <= 0 is documented to mean "library default" — see p4a.h §6.
    // Any int32_t value is accepted.
    try {
        as_core(ctx)->set_num_threads(n_threads);
        return P4A_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in p4a_context_set_num_threads");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_context_get_num_threads(const p4a_context_t* ctx,
                                                  int32_t* out_threads) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    if (out_threads == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in p4a_context_get_num_threads");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        *out_threads = as_core(ctx)->num_threads();
        return P4A_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in p4a_context_get_num_threads");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API const char* p4a_context_last_error(const p4a_context_t* ctx) {
    if (ctx == nullptr) return kEmptyError;
    try {
        return as_core(ctx)->last_error();
    } catch (...) {
        return kEmptyError;
    }
}

P4A_API void p4a_context_clear_error(p4a_context_t* ctx) {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->clear_error();
    } catch (...) {
        // Best-effort — clearing should never throw, but be safe.
    }
}

P4A_API p4a_status_t p4a_context_set_user_data(p4a_context_t* ctx, void* user) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        as_core(ctx)->set_user_data(user);
        return P4A_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in p4a_context_set_user_data");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void* p4a_context_get_user_data(const p4a_context_t* ctx) {
    if (ctx == nullptr) return nullptr;
    try {
        return as_core(ctx)->user_data();
    } catch (...) {
        return nullptr;
    }
}

}  // extern "C"
