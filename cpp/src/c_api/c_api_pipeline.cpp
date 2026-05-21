// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_pipeline_t.

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

inline ::n4m::core::Context* as_core(n4m_context_t* ctx) noexcept {
    return static_cast<::n4m::core::Context*>(ctx);
}
inline ::n4m::core::Pipeline* as_core(n4m_pipeline_t* p) noexcept {
    return static_cast<::n4m::core::Pipeline*>(p);
}
inline const ::n4m::core::Pipeline* as_core(const n4m_pipeline_t* p) noexcept {
    return static_cast<const ::n4m::core::Pipeline*>(p);
}

void set_error(n4m_context_t* ctx, const char* message) noexcept {
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

N4M_API n4m_status_t n4m_pipeline_create(n4m_pipeline_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out = new n4m_pipeline_s{};
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pipeline_destroy(n4m_pipeline_t* pipe) {
    try { delete pipe; } catch (...) {}
}

N4M_API n4m_status_t n4m_pipeline_clone(const n4m_pipeline_t* src,
                                         n4m_pipeline_t** out_dst) {
    if (src == nullptr || out_dst == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        auto* dst = new n4m_pipeline_s{};
        *static_cast<::n4m::core::Pipeline*>(dst) = *as_core(src);
        *out_dst = dst;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out_dst = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_dst = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pipeline_add_operator(n4m_pipeline_t* pipe,
                                                n4m_operator_kind_t kind,
                                                const double* params,
                                                int32_t n_params) {
    if (pipe == nullptr) return N4M_ERR_NULL_POINTER;
    if (!::n4m::core::operator_kind_is_valid(kind)) return N4M_ERR_INVALID_ARGUMENT;
    if (n_params < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (n_params > 0 && params == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        as_core(pipe)->add_operator(kind, params, n_params);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pipeline_size(const n4m_pipeline_t* pipe,
                                        int32_t* out_size) {
    if (pipe == nullptr || out_size == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_size = as_core(pipe)->size();
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pipeline_fit(n4m_context_t* ctx,
                                       n4m_pipeline_t* pipe,
                                       const n4m_matrix_view_t* X,
                                       const n4m_matrix_view_t* Y) {
    if (ctx == nullptr || pipe == nullptr || X == nullptr) {
        set_error(ctx, "null pointer in n4m_pipeline_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        return as_core(pipe)->fit(*as_core(ctx), *X, Y);
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pipeline_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pipeline_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pipeline_transform(n4m_context_t* ctx,
                                             const n4m_pipeline_t* pipe,
                                             const n4m_matrix_view_t* X,
                                             n4m_matrix_view_t* out) {
    if (ctx == nullptr || pipe == nullptr || X == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in n4m_pipeline_transform");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        return as_core(pipe)->transform(*as_core(ctx), *X, *out);
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pipeline_transform");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pipeline_transform");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pipeline_transform_alloc(n4m_context_t* ctx,
                                                   const n4m_pipeline_t* pipe,
                                                   const n4m_matrix_view_t* X,
                                                   n4m_array_t** out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (ctx == nullptr || pipe == nullptr || X == nullptr || out == nullptr) {
        set_error(ctx, "null pointer in n4m_pipeline_transform_alloc");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        auto arr = std::make_unique<n4m_array_s>();
        arr->dtype = N4M_DTYPE_F64;
        arr->rows = X->rows;
        arr->cols = X->cols;
        std::size_t n_values = 0;
        if (!output_element_count(arr->rows, arr->cols, n_values)) {
            set_error(ctx, "pipeline transform output shape is invalid or too large");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        arr->values.assign(n_values, 0.0);
        n4m_matrix_view_t view{};
        view.data = arr->values.empty() ? nullptr : arr->values.data();
        view.rows = arr->rows;
        view.cols = arr->cols;
        view.row_stride = arr->cols > 0 ? arr->cols : 1;
        view.col_stride = 1;
        view.dtype = N4M_DTYPE_F64;
        view.reserved0 = 0;
        const n4m_status_t status = as_core(pipe)->transform(*as_core(ctx), *X, view);
        if (status != N4M_OK) {
            return status;
        }
        *out = arr.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pipeline_transform_alloc");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pipeline_transform_alloc");
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
