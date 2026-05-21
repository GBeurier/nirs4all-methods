// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for AOM global and per-component (POP) selection. The
// kernel itself lives in cpp/src/core/aom_selection.cpp; this layer adapts
// the internal result struct into a public ABI result handle that owns its
// buffers and exposes them as raw pointers.

#include <stddef.h>
#include <stdint.h>

#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "pls4all/p4a.h"

#include "core/aom_selection.hpp"
#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/operator_bank.hpp"
#include "core/validation.hpp"

namespace {

inline ::n4m::core::Context* as_core(n4m_context_t* ctx) noexcept {
    return static_cast<::n4m::core::Context*>(ctx);
}
inline const ::n4m::core::Config* as_core(const n4m_config_t* cfg) noexcept {
    return static_cast<const ::n4m::core::Config*>(cfg);
}
inline const ::n4m::core::OperatorBank* as_core(const n4m_operator_bank_t* bank) noexcept {
    return static_cast<const ::n4m::core::OperatorBank*>(bank);
}
inline const ::n4m::core::ValidationPlan* as_core(const n4m_validation_plan_t* plan) noexcept {
    return static_cast<const ::n4m::core::ValidationPlan*>(plan);
}

void set_error(n4m_context_t* ctx, const char* msg) noexcept {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->set_error(msg);
    } catch (...) {
        // last_error is best-effort.
    }
}

[[nodiscard]] n4m_status_t validate_select_inputs(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    if (cfg == nullptr || bank == nullptr || X == nullptr || Y == nullptr || plan == nullptr) {
        set_error(ctx, "null pointer argument to AOM selection");
        return N4M_ERR_NULL_POINTER;
    }
    if (max_components <= 0) {
        set_error(ctx, "max_components must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

void populate_typed_operator_kinds(
    const std::vector<std::int64_t>& src,
    std::vector<n4m_operator_kind_t>& dst) {
    dst.clear();
    dst.reserve(src.size());
    for (const auto kind : src) {
        dst.push_back(static_cast<n4m_operator_kind_t>(kind));
    }
}

void populate_selected_indices_i32(
    const std::vector<std::int64_t>& src,
    std::vector<std::int32_t>& dst) {
    dst.clear();
    dst.reserve(src.size());
    for (const auto value : src) {
        dst.push_back(static_cast<std::int32_t>(value));
    }
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------ */
/*  AOM global selection                                              */
/* ------------------------------------------------------------------ */

N4M_API n4m_status_t n4m_aom_global_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components,
    n4m_aom_global_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;

    const n4m_status_t pre = validate_select_inputs(
        ctx, cfg, bank, X, Y, plan, max_components);
    if (pre != N4M_OK) return pre;

    try {
        auto handle = std::make_unique<n4m_aom_global_result_s>();
        const n4m_status_t status = ::n4m::core::select_aom_global(
            *as_core(ctx), *as_core(cfg), *as_core(bank),
            *X, *Y, *as_core(plan), max_components, handle->inner);
        if (status != N4M_OK) {
            return status;
        }
        handle->predictions_rows = X->rows;
        handle->predictions_cols = Y->cols;
        populate_typed_operator_kinds(
            handle->inner.operator_kinds, handle->operator_kinds_typed);
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_aom_global_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_aom_global_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aom_global_result_destroy(n4m_aom_global_result_t* result) {
    try {
        delete result;
    } catch (...) {
        // never propagate.
    }
}

N4M_API n4m_status_t n4m_aom_global_result_get_n_operators(
    const n4m_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.n_operators;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_max_components(
    const n4m_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.max_components;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_selected_operator_index(
    const n4m_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.selected_operator_index;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_selected_n_components(
    const n4m_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.selected_n_components;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_best_score(
    const n4m_aom_global_result_t* result, double* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.best_score;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_operator_kinds(
    const n4m_aom_global_result_t* result,
    const n4m_operator_kind_t** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->operator_kinds_typed.empty()
        ? nullptr : result->operator_kinds_typed.data();
    *out_size = static_cast<int32_t>(result->operator_kinds_typed.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_operator_scores(
    const n4m_aom_global_result_t* result,
    const double** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->inner.operator_scores.empty()
        ? nullptr : result->inner.operator_scores.data();
    *out_size = static_cast<int32_t>(result->inner.operator_scores.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_rmse_curves(
    const n4m_aom_global_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->inner.rmse_curves.empty()
        ? nullptr : result->inner.rmse_curves.data();
    *out_rows = result->inner.n_operators;
    *out_cols = result->inner.max_components;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_global_result_get_predictions(
    const n4m_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->inner.predictions.empty()
        ? nullptr : result->inner.predictions.data();
    *out_rows = result->predictions_rows;
    *out_cols = result->predictions_cols;
    return N4M_OK;
}

/* ------------------------------------------------------------------ */
/*  AOM per-component (POP) selection                                 */
/* ------------------------------------------------------------------ */

N4M_API n4m_status_t n4m_aom_per_component_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components,
    n4m_aom_per_component_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;

    const n4m_status_t pre = validate_select_inputs(
        ctx, cfg, bank, X, Y, plan, max_components);
    if (pre != N4M_OK) return pre;

    try {
        auto handle = std::make_unique<n4m_aom_per_component_result_s>();
        const n4m_status_t status = ::n4m::core::select_aom_per_component(
            *as_core(ctx), *as_core(cfg), *as_core(bank),
            *X, *Y, *as_core(plan), max_components, handle->inner);
        if (status != N4M_OK) {
            return status;
        }
        handle->predictions_rows = X->rows;
        handle->predictions_cols = Y->cols;
        populate_typed_operator_kinds(
            handle->inner.operator_kinds, handle->operator_kinds_typed);
        populate_selected_indices_i32(
            handle->inner.selected_operator_indices,
            handle->selected_operator_indices_i32);
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_aom_per_component_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_aom_per_component_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aom_per_component_result_destroy(
    n4m_aom_per_component_result_t* result) {
    try {
        delete result;
    } catch (...) {
    }
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_n_operators(
    const n4m_aom_per_component_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.n_operators;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_max_components(
    const n4m_aom_per_component_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.max_components;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_selected_n_components(
    const n4m_aom_per_component_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.selected_n_components;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_best_score(
    const n4m_aom_per_component_result_t* result, double* out) {
    if (result == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = result->inner.best_score;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_operator_kinds(
    const n4m_aom_per_component_result_t* result,
    const n4m_operator_kind_t** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->operator_kinds_typed.empty()
        ? nullptr : result->operator_kinds_typed.data();
    *out_size = static_cast<int32_t>(result->operator_kinds_typed.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_selected_operator_indices(
    const n4m_aom_per_component_result_t* result,
    const int32_t** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->selected_operator_indices_i32.empty()
        ? nullptr : result->selected_operator_indices_i32.data();
    *out_size = static_cast<int32_t>(result->selected_operator_indices_i32.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_component_scores(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->inner.component_scores.empty()
        ? nullptr : result->inner.component_scores.data();
    *out_rows = result->inner.max_components;
    *out_cols = result->inner.n_operators;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_prefix_scores(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->inner.prefix_scores.empty()
        ? nullptr : result->inner.prefix_scores.data();
    *out_size = static_cast<int32_t>(result->inner.prefix_scores.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_aom_per_component_result_get_predictions(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_data = result->inner.predictions.empty()
        ? nullptr : result->inner.predictions.data();
    *out_rows = result->predictions_rows;
    *out_cols = result->predictions_cols;
    return N4M_OK;
}

}  // extern "C"
