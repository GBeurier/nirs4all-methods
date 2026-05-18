// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for p4a_pipeline_t.

#include <stdint.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <new>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/model.hpp"
#include "core/pipeline.hpp"

namespace {

inline ::pls4all::core::Context* as_core(p4a_context_t* ctx) noexcept {
    return static_cast<::pls4all::core::Context*>(ctx);
}
inline ::pls4all::core::Pipeline* as_core(p4a_pipeline_t* p) noexcept {
    return static_cast<::pls4all::core::Pipeline*>(p);
}
inline const ::pls4all::core::Pipeline* as_core(const p4a_pipeline_t* p) noexcept {
    return static_cast<const ::pls4all::core::Pipeline*>(p);
}

void set_error(p4a_context_t* ctx, const char* message) noexcept {
    if (ctx != nullptr) {
        as_core(ctx)->set_error(message);
    }
}

[[nodiscard]] bool output_element_count(std::int64_t rows,
                                        std::int64_t cols,
                                        std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const std::uint64_t urows = static_cast<std::uint64_t>(rows);
    const std::uint64_t ucols = static_cast<std::uint64_t>(cols);
    if (ucols != 0U &&
        urows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / ucols) {
        return false;
    }
    out = static_cast<std::size_t>(urows * ucols);
    return true;
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

P4A_API p4a_status_t p4a_pipeline_fit(p4a_context_t* ctx,
                                       p4a_pipeline_t* pipe,
                                       const p4a_matrix_view_t* X,
                                       const p4a_matrix_view_t* Y) {
    if (ctx == nullptr || pipe == nullptr || X == nullptr) {
        set_error(ctx, "null pointer in p4a_pipeline_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        return as_core(pipe)->fit(*as_core(ctx), *X, Y);
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_pipeline_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_pipeline_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_pipeline_transform(p4a_context_t* ctx,
                                             const p4a_pipeline_t* pipe,
                                             const p4a_matrix_view_t* X,
                                             p4a_matrix_view_t* out) {
    if (ctx == nullptr || pipe == nullptr || X == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in p4a_pipeline_transform");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        return as_core(pipe)->transform(*as_core(ctx), *X, *out);
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_pipeline_transform");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_pipeline_transform");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_pipeline_transform_alloc(p4a_context_t* ctx,
                                                   const p4a_pipeline_t* pipe,
                                                   const p4a_matrix_view_t* X,
                                                   p4a_array_t** out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (ctx == nullptr || pipe == nullptr || X == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in p4a_pipeline_transform_alloc");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        auto arr = std::make_unique<p4a_array_s>();
        arr->dtype = P4A_DTYPE_F64;
        arr->rows = X->rows;
        arr->cols = X->cols;
        std::size_t n_values = 0;
        if (!output_element_count(arr->rows, arr->cols, n_values)) {
            set_error(ctx, "pipeline transform output shape is invalid or too large");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        arr->values.assign(n_values, 0.0);
        p4a_matrix_view_t view{};
        view.data = arr->values.empty() ? nullptr : arr->values.data();
        view.rows = arr->rows;
        view.cols = arr->cols;
        view.row_stride = arr->cols > 0 ? arr->cols : 1;
        view.col_stride = 1;
        view.dtype = P4A_DTYPE_F64;
        view.reserved0 = 0;
        const p4a_status_t status = as_core(pipe)->transform(*as_core(ctx), *X, view);
        if (status != P4A_OK) {
            return status;
        }
        *out = arr.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_pipeline_transform_alloc");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_pipeline_transform_alloc");
        return P4A_ERR_INTERNAL;
    }
}

}  // extern "C"
