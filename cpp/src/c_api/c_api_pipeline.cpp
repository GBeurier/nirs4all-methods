// SPDX-License-Identifier: CeCILL-2.1
//
// extern "C" wrappers for p4a_pipeline_t. Lifecycle + add_operator + size
// are fully implemented; fit / transform / transform_alloc are Phase 0
// stubs returning P4A_ERR_NOT_IMPLEMENTED.

#include <stdint.h>

#include <new>

#include "pls4all/p4a.h"

#include "core/pipeline.hpp"

namespace {

inline ::pls4all::core::Pipeline* as_core(p4a_pipeline_t* p) noexcept {
    return static_cast<::pls4all::core::Pipeline*>(p);
}
inline const ::pls4all::core::Pipeline* as_core(const p4a_pipeline_t* p) noexcept {
    return static_cast<const ::pls4all::core::Pipeline*>(p);
}

}  // namespace

extern "C" {

P4A_API p4a_status_t p4a_pipeline_create(p4a_pipeline_t** out) {
    if (out == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out = new p4a_pipeline_s{};
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_pipeline_destroy(p4a_pipeline_t* pipe) {
    try { delete pipe; } catch (...) {}
}

P4A_API p4a_status_t p4a_pipeline_clone(const p4a_pipeline_t* src,
                                         p4a_pipeline_t** out_dst) {
    if (src == nullptr || out_dst == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        auto* dst = new p4a_pipeline_s{};
        *static_cast<::pls4all::core::Pipeline*>(dst) = *as_core(src);
        *out_dst = dst;
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        *out_dst = nullptr;
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_dst = nullptr;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_pipeline_add_operator(p4a_pipeline_t* pipe,
                                                p4a_operator_kind_t kind,
                                                const double* params,
                                                int32_t n_params) {
    if (pipe == nullptr) return P4A_ERR_NULL_POINTER;
    if (!::pls4all::core::operator_kind_is_valid(kind)) return P4A_ERR_INVALID_ARGUMENT;
    if (n_params < 0) return P4A_ERR_INVALID_ARGUMENT;
    if (n_params > 0 && params == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        as_core(pipe)->add_operator(kind, params, n_params);
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_pipeline_size(const p4a_pipeline_t* pipe,
                                        int32_t* out_size) {
    if (pipe == nullptr || out_size == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out_size = as_core(pipe)->size();
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

/* --- Phase 0 stubs: fit / transform / transform_alloc --- */

P4A_API p4a_status_t p4a_pipeline_fit(p4a_context_t* /*ctx*/,
                                       p4a_pipeline_t* /*pipe*/,
                                       const p4a_matrix_view_t* /*X*/,
                                       const p4a_matrix_view_t* /*Y*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_pipeline_transform(p4a_context_t* /*ctx*/,
                                             const p4a_pipeline_t* /*pipe*/,
                                             const p4a_matrix_view_t* /*X*/,
                                             p4a_matrix_view_t* /*out*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_pipeline_transform_alloc(p4a_context_t* /*ctx*/,
                                                    const p4a_pipeline_t* /*pipe*/,
                                                    const p4a_matrix_view_t* /*X*/,
                                                    p4a_array_t** out) {
    if (out != nullptr) *out = nullptr;
    return P4A_ERR_NOT_IMPLEMENTED;
}

}  // extern "C"
