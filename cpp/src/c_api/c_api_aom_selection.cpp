// SPDX-License-Identifier: CeCILL-2.1
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
#include "core/context.hpp"
#include "core/operator_bank.hpp"
#include "core/validation.hpp"

namespace {

inline ::pls4all::core::Context* as_core(p4a_context_t* ctx) noexcept {
    return static_cast<::pls4all::core::Context*>(ctx);
}
inline const ::pls4all::core::Config* as_core(const p4a_config_t* cfg) noexcept {
    return static_cast<const ::pls4all::core::Config*>(cfg);
}
inline const ::pls4all::core::OperatorBank* as_core(const p4a_operator_bank_t* bank) noexcept {
    return static_cast<const ::pls4all::core::OperatorBank*>(bank);
}
inline const ::pls4all::core::ValidationPlan* as_core(const p4a_validation_plan_t* plan) noexcept {
    return static_cast<const ::pls4all::core::ValidationPlan*>(plan);
}

void set_error(p4a_context_t* ctx, const char* msg) noexcept {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->set_error(msg);
    } catch (...) {
        // last_error is best-effort.
    }
}

[[nodiscard]] p4a_status_t validate_select_inputs(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_operator_bank_t* bank,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t max_components) {
    if (ctx == nullptr) return P4A_ERR_NULL_POINTER;
    if (cfg == nullptr || bank == nullptr || X == nullptr || Y == nullptr || plan == nullptr) {
        set_error(ctx, "null pointer argument to AOM selection");
        return P4A_ERR_NULL_POINTER;
    }
    if (max_components <= 0) {
        set_error(ctx, "max_components must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

void populate_typed_operator_kinds(
    const std::vector<std::int64_t>& src,
    std::vector<p4a_operator_kind_t>& dst) {
    dst.clear();
    dst.reserve(src.size());
    for (const auto kind : src) {
        dst.push_back(static_cast<p4a_operator_kind_t>(kind));
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

P4A_API p4a_status_t p4a_aom_global_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_operator_bank_t* bank,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t max_components,
    p4a_aom_global_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;

    const p4a_status_t pre = validate_select_inputs(
        ctx, cfg, bank, X, Y, plan, max_components);
    if (pre != P4A_OK) return pre;

    try {
        auto handle = std::make_unique<p4a_aom_global_result_s>();
        const p4a_status_t status = ::pls4all::core::select_aom_global(
            *as_core(ctx), *as_core(cfg), *as_core(bank),
            *X, *Y, *as_core(plan), max_components, handle->inner);
        if (status != P4A_OK) {
            return status;
        }
        handle->predictions_rows = X->rows;
        handle->predictions_cols = Y->cols;
        populate_typed_operator_kinds(
            handle->inner.operator_kinds, handle->operator_kinds_typed);
        *out_result = handle.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_aom_global_select");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_aom_global_select");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_aom_global_result_destroy(p4a_aom_global_result_t* result) {
    try {
        delete result;
    } catch (...) {
        // never propagate.
    }
}

P4A_API p4a_status_t p4a_aom_global_result_get_n_operators(
    const p4a_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.n_operators;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_max_components(
    const p4a_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.max_components;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_selected_operator_index(
    const p4a_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.selected_operator_index;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_selected_n_components(
    const p4a_aom_global_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.selected_n_components;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_best_score(
    const p4a_aom_global_result_t* result, double* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.best_score;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_operator_kinds(
    const p4a_aom_global_result_t* result,
    const p4a_operator_kind_t** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->operator_kinds_typed.empty()
        ? nullptr : result->operator_kinds_typed.data();
    *out_size = static_cast<int32_t>(result->operator_kinds_typed.size());
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_operator_scores(
    const p4a_aom_global_result_t* result,
    const double** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->inner.operator_scores.empty()
        ? nullptr : result->inner.operator_scores.data();
    *out_size = static_cast<int32_t>(result->inner.operator_scores.size());
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_rmse_curves(
    const p4a_aom_global_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->inner.rmse_curves.empty()
        ? nullptr : result->inner.rmse_curves.data();
    *out_rows = result->inner.n_operators;
    *out_cols = result->inner.max_components;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_global_result_get_predictions(
    const p4a_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->inner.predictions.empty()
        ? nullptr : result->inner.predictions.data();
    *out_rows = result->predictions_rows;
    *out_cols = result->predictions_cols;
    return P4A_OK;
}

/* ------------------------------------------------------------------ */
/*  AOM per-component (POP) selection                                 */
/* ------------------------------------------------------------------ */

P4A_API p4a_status_t p4a_aom_per_component_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_operator_bank_t* bank,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t max_components,
    p4a_aom_per_component_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;

    const p4a_status_t pre = validate_select_inputs(
        ctx, cfg, bank, X, Y, plan, max_components);
    if (pre != P4A_OK) return pre;

    try {
        auto handle = std::make_unique<p4a_aom_per_component_result_s>();
        const p4a_status_t status = ::pls4all::core::select_aom_per_component(
            *as_core(ctx), *as_core(cfg), *as_core(bank),
            *X, *Y, *as_core(plan), max_components, handle->inner);
        if (status != P4A_OK) {
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_aom_per_component_select");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_aom_per_component_select");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_aom_per_component_result_destroy(
    p4a_aom_per_component_result_t* result) {
    try {
        delete result;
    } catch (...) {
    }
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_n_operators(
    const p4a_aom_per_component_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.n_operators;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_max_components(
    const p4a_aom_per_component_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.max_components;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_selected_n_components(
    const p4a_aom_per_component_result_t* result, int32_t* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.selected_n_components;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_best_score(
    const p4a_aom_per_component_result_t* result, double* out) {
    if (result == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    *out = result->inner.best_score;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_operator_kinds(
    const p4a_aom_per_component_result_t* result,
    const p4a_operator_kind_t** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->operator_kinds_typed.empty()
        ? nullptr : result->operator_kinds_typed.data();
    *out_size = static_cast<int32_t>(result->operator_kinds_typed.size());
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_selected_operator_indices(
    const p4a_aom_per_component_result_t* result,
    const int32_t** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->selected_operator_indices_i32.empty()
        ? nullptr : result->selected_operator_indices_i32.data();
    *out_size = static_cast<int32_t>(result->selected_operator_indices_i32.size());
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_component_scores(
    const p4a_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->inner.component_scores.empty()
        ? nullptr : result->inner.component_scores.data();
    *out_rows = result->inner.max_components;
    *out_cols = result->inner.n_operators;
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_prefix_scores(
    const p4a_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_size) {
    if (result == nullptr || out_data == nullptr || out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->inner.prefix_scores.empty()
        ? nullptr : result->inner.prefix_scores.data();
    *out_size = static_cast<int32_t>(result->inner.prefix_scores.size());
    return P4A_OK;
}

P4A_API p4a_status_t p4a_aom_per_component_result_get_predictions(
    const p4a_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols) {
    if (result == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out_data = result->inner.predictions.empty()
        ? nullptr : result->inner.predictions.data();
    *out_rows = result->predictions_rows;
    *out_cols = result->predictions_cols;
    return P4A_OK;
}

}  // extern "C"
