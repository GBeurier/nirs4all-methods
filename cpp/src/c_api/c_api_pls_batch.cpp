// SPDX-License-Identifier: CECILL-2.1
//
// C ABI surface for PLS-only cross-validation.
//
// This currently delegates to the exact PLS branch of n4m_sweep_run. That gives
// downstream code a stable, score-equivalent reference implementation while the
// future fused/grouped PLS grinder can replace the internals without changing
// the ABI.

#include <stdint.h>

#include "n4m/n4m.h"

#include "core/common/context.hpp"

namespace {

inline ::n4m::core::Context* as_core(n4m_context_t* ctx) noexcept {
    return static_cast<::n4m::core::Context*>(ctx);
}

void set_error(n4m_context_t* ctx, const char* msg) noexcept {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->set_error(msg);
    } catch (...) {
    }
}

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_pls_cross_validate(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int32_t* fold_ids,
    int64_t n_fold_ids,
    int32_t n_folds,
    const int32_t* component_grid,
    int64_t n_component_grid,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) {
        set_error(ctx, "null out_result in n4m_pls_cross_validate");
        return N4M_ERR_NULL_POINTER;
    }
    *out_result = nullptr;

    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_cross_validate");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_fold_ids < 0 || n_component_grid < 0) {
        set_error(ctx, "negative length/count in n4m_pls_cross_validate");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if ((fold_ids == nullptr) != (n_fold_ids == 0)) {
        set_error(ctx, "inconsistent fold_ids length in n4m_pls_cross_validate");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (component_grid == nullptr || n_component_grid == 0) {
        set_error(ctx, "empty component_grid in n4m_pls_cross_validate");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    return n4m_sweep_run(ctx, cfg, X, Y, n_folds, fold_ids, n_fold_ids,
                         nullptr, 0, component_grid, n_component_grid,
                         2, out_result);
}

}  // extern "C"
