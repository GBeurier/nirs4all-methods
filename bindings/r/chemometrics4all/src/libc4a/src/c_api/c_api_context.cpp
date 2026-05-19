// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for c4a_context_t. Every wrapper has a try/catch
// around its full body so no C++ exception ever crosses the boundary.
// Failed calls on a non-NULL context write a message to last_error.

#include <stddef.h>
#include <stdint.h>

#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/context.hpp"

namespace {

inline ::chemometrics4all::core::Context* as_core(c4a_context_t* ctx) noexcept {
    return static_cast<::chemometrics4all::core::Context*>(ctx);
}

inline const ::chemometrics4all::core::Context* as_core(const c4a_context_t* ctx) noexcept {
    return static_cast<const ::chemometrics4all::core::Context*>(ctx);
}

// Mutable cast used only to write an error message on a const-ish path
// (a getter that wants to report a null out-pointer).
inline ::chemometrics4all::core::Context* as_core_mut(const c4a_context_t* ctx) noexcept {
    return const_cast<::chemometrics4all::core::Context*>(as_core(ctx));
}

constexpr const char* kEmptyError = "";

}  // namespace

extern "C" {

C4A_API c4a_status_t c4a_context_create(c4a_context_t** out_ctx) {
    if (out_ctx == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_ctx = new c4a_context_s{};
        return C4A_OK;
    } catch (const std::bad_alloc&) {
        *out_ctx = nullptr;
        return C4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_ctx = nullptr;
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_context_destroy(c4a_context_t* ctx) {
    try {
        delete ctx;
    } catch (...) {
        // Best-effort cleanup — never propagate.
    }
}

C4A_API c4a_status_t c4a_context_set_seed(c4a_context_t* ctx, uint64_t seed) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        as_core(ctx)->set_seed(seed);
        return C4A_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in c4a_context_set_seed");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_context_get_seed(const c4a_context_t* ctx,
                                           uint64_t* out_seed) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    if (out_seed == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in c4a_context_get_seed");
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_seed = as_core(ctx)->seed();
        return C4A_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in c4a_context_get_seed");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_context_set_backend(c4a_context_t* ctx,
                                              c4a_backend_t backend) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        return as_core(ctx)->set_backend(backend);
    } catch (...) {
        as_core(ctx)->set_error("internal error in c4a_context_set_backend");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_context_get_backend(const c4a_context_t* ctx,
                                              c4a_backend_t* out_backend) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    if (out_backend == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in c4a_context_get_backend");
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_backend = as_core(ctx)->backend();
        return C4A_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in c4a_context_get_backend");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_context_set_num_threads(c4a_context_t* ctx,
                                                  int32_t n_threads) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    // n_threads <= 0 is documented to mean "library default" — see c4a.h §6.
    // Any int32_t value is accepted.
    try {
        as_core(ctx)->set_num_threads(n_threads);
        return C4A_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in c4a_context_set_num_threads");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_context_get_num_threads(const c4a_context_t* ctx,
                                                  int32_t* out_threads) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    if (out_threads == nullptr) {
        as_core_mut(ctx)->set_error("null out pointer in c4a_context_get_num_threads");
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_threads = as_core(ctx)->num_threads();
        return C4A_OK;
    } catch (...) {
        as_core_mut(ctx)->set_error("internal error in c4a_context_get_num_threads");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API const char* c4a_context_last_error(const c4a_context_t* ctx) {
    if (ctx == nullptr) return kEmptyError;
    try {
        return as_core(ctx)->last_error();
    } catch (...) {
        return kEmptyError;
    }
}

C4A_API void c4a_context_clear_error(c4a_context_t* ctx) {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->clear_error();
    } catch (...) {
        // Best-effort — clearing should never throw, but be safe.
    }
}

C4A_API c4a_status_t c4a_context_set_user_data(c4a_context_t* ctx, void* user) {
    if (ctx == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        as_core(ctx)->set_user_data(user);
        return C4A_OK;
    } catch (...) {
        as_core(ctx)->set_error("internal error in c4a_context_set_user_data");
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void* c4a_context_get_user_data(const c4a_context_t* ctx) {
    if (ctx == nullptr) return nullptr;
    try {
        return as_core(ctx)->user_data();
    } catch (...) {
        return nullptr;
    }
}

}  // extern "C"
