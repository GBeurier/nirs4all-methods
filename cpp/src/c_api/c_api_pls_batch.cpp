// SPDX-License-Identifier: CECILL-2.1
//
// Reserved C ABI surface for the future fused/batched PLS CV executor.
// The executor itself is deliberately deferred; callers get an explicit,
// testable NOT_IMPLEMENTED status instead of accidentally hitting a slow or
// semantically different fallback.

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

    (void)cfg;
    (void)X;
    (void)Y;
    (void)fold_ids;
    (void)n_fold_ids;
    (void)n_folds;
    (void)component_grid;
    (void)n_component_grid;

    set_error(ctx,
              "n4m_pls_cross_validate is reserved for fused batched PLS "
              "cross-validation and is not implemented yet");
    return N4M_ERR_NOT_IMPLEMENTED;
}

}  // extern "C"
