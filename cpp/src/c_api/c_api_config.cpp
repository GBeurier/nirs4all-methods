// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_config_t. Every setter validates its input
// and leaves the config UNCHANGED on rejection (idempotent failure). Every
// wrapper has a try/catch around its body — see Direction_Technique §4 for
// the no-exception-across-ABI rule.

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/config.hpp"

namespace {

inline ::n4m::core::Config* as_core(n4m_config_t* cfg) noexcept {
    return static_cast<::n4m::core::Config*>(cfg);
}
inline const ::n4m::core::Config* as_core(const n4m_config_t* cfg) noexcept {
    return static_cast<const ::n4m::core::Config*>(cfg);
}

inline bool in_range_algorithm(n4m_algorithm_t a) noexcept {
    return a >= N4M_ALGO_PLS_REGRESSION && a <= N4M_ALGO_PCR;
}
inline bool in_range_solver(n4m_solver_t s) noexcept {
    return s >= N4M_SOLVER_NIPALS && s <= N4M_SOLVER_RANDOMIZED_SVD;
}
inline bool in_range_deflation(n4m_deflation_t d) noexcept {
    return d >= N4M_DEFLATION_REGRESSION && d <= N4M_DEFLATION_ORTHOGONAL;
}
inline bool in_range_dtype(n4m_dtype_t d) noexcept {
    return d == N4M_DTYPE_F64 || d == N4M_DTYPE_F32
        || d == N4M_DTYPE_I32 || d == N4M_DTYPE_I64;
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------------ */
/* Lifecycle                                                                */
/* ------------------------------------------------------------------------ */

N4M_API n4m_status_t n4m_config_create(n4m_config_t** out_cfg) {
    if (out_cfg == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_cfg = new n4m_config_s{};
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out_cfg = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_cfg = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_config_destroy(n4m_config_t* cfg) {
    try {
        delete cfg;
    } catch (...) {
        // Best-effort.
    }
}

N4M_API n4m_status_t n4m_config_clone(const n4m_config_t* src,
                                       n4m_config_t** out_dst) {
    if (src == nullptr || out_dst == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        auto* dst = new n4m_config_s{};
        *static_cast<::n4m::core::Config*>(dst) = *as_core(src);
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

/* ------------------------------------------------------------------------ */
/* Setters                                                                  */
/* ------------------------------------------------------------------------ */

#define N4M_CFG_TRY_BEGIN \
    try {
#define N4M_CFG_TRY_END \
    } catch (...) { return N4M_ERR_INTERNAL; }

N4M_API n4m_status_t n4m_config_set_algorithm(n4m_config_t* cfg, n4m_algorithm_t v) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (!in_range_algorithm(v)) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->algorithm = v;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_solver(n4m_config_t* cfg, n4m_solver_t v) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (!in_range_solver(v)) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->solver = v;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_deflation(n4m_config_t* cfg, n4m_deflation_t v) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (!in_range_deflation(v)) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->deflation = v;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_n_components(n4m_config_t* cfg, int32_t n) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    // Reject non-positive values. There is no upper cap at the C ABI level;
    // implementations may degrade performance gracefully for very large n,
    // and Phase 1+ will surface a N4M_ERR_INVALID_ARGUMENT at fit time if
    // n > rank(X).
    N4M_CFG_TRY_BEGIN
        if (n < 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->n_components = n;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_center_x(n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->center_x = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_scale_x(n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->scale_x = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_center_y(n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->center_y = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_scale_y(n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->scale_y = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_tol(n4m_config_t* cfg, double tol) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (std::isnan(tol) || std::isinf(tol) || tol <= 0.0) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        as_core(cfg)->tol = tol;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_max_iter(n4m_config_t* cfg, int32_t max_iter) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (max_iter < 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->max_iter = max_iter;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_store_scores(n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->store_scores = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_store_diagnostics(n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->store_diagnostics = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_sparse_simpls_legacy(
    n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->sparse_simpls_legacy = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_get_sparse_simpls_legacy(
    const n4m_config_t* cfg, int32_t* out_enabled) {
    if (cfg == nullptr || out_enabled == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        *out_enabled = as_core(cfg)->sparse_simpls_legacy;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_robust_pls_legacy(
    n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->robust_pls_legacy = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_get_robust_pls_legacy(
    const n4m_config_t* cfg, int32_t* out_enabled) {
    if (cfg == nullptr || out_enabled == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        *out_enabled = as_core(cfg)->robust_pls_legacy;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_approximate_press_legacy(
    n4m_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->approximate_press_legacy = enabled;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_get_approximate_press_legacy(
    const n4m_config_t* cfg, int32_t* out_enabled) {
    if (cfg == nullptr || out_enabled == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        *out_enabled = as_core(cfg)->approximate_press_legacy;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_dtype(n4m_config_t* cfg, n4m_dtype_t v) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        if (!in_range_dtype(v)) return N4M_ERR_INVALID_ARGUMENT;
        as_core(cfg)->dtype = v;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_pipeline(n4m_config_t* cfg,
                                              const n4m_pipeline_t* pipe) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        as_core(cfg)->pipeline = pipe;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_operator_bank(n4m_config_t* cfg,
                                                    const n4m_operator_bank_t* bank) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        as_core(cfg)->operator_bank = bank;
        return N4M_OK;
    N4M_CFG_TRY_END
}

N4M_API n4m_status_t n4m_config_set_gating_strategy(n4m_config_t* cfg,
                                                      const n4m_gating_strategy_t* gate) {
    if (cfg == nullptr) return N4M_ERR_NULL_POINTER;
    N4M_CFG_TRY_BEGIN
        as_core(cfg)->gating_strategy = gate;
        return N4M_OK;
    N4M_CFG_TRY_END
}

#undef N4M_CFG_TRY_BEGIN
#undef N4M_CFG_TRY_END

/* ------------------------------------------------------------------------ */
/* Getters                                                                  */
/* ------------------------------------------------------------------------ */

#define N4M_GETTER(NAME, T, FIELD)                                          \
    N4M_API n4m_status_t n4m_config_get_##NAME(const n4m_config_t* cfg,     \
                                                T* out) {                    \
        if (cfg == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;  \
        try {                                                                \
            *out = as_core(cfg)->FIELD;                                      \
            return N4M_OK;                                                   \
        } catch (...) {                                                      \
            return N4M_ERR_INTERNAL;                                         \
        }                                                                    \
    }

N4M_GETTER(algorithm,         n4m_algorithm_t, algorithm)
N4M_GETTER(solver,            n4m_solver_t,    solver)
N4M_GETTER(deflation,         n4m_deflation_t, deflation)
N4M_GETTER(n_components,      int32_t,         n_components)
N4M_GETTER(center_x,          int32_t,         center_x)
N4M_GETTER(scale_x,           int32_t,         scale_x)
N4M_GETTER(center_y,          int32_t,         center_y)
N4M_GETTER(scale_y,           int32_t,         scale_y)
N4M_GETTER(tol,               double,          tol)
N4M_GETTER(max_iter,          int32_t,         max_iter)
N4M_GETTER(store_scores,      int32_t,         store_scores)
N4M_GETTER(store_diagnostics, int32_t,         store_diagnostics)
N4M_GETTER(dtype,             n4m_dtype_t,     dtype)

#undef N4M_GETTER

N4M_API n4m_status_t n4m_config_get_pipeline(const n4m_config_t* cfg,
                                              const n4m_pipeline_t** out) {
    if (cfg == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out = as_core(cfg)->pipeline;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
N4M_API n4m_status_t n4m_config_get_operator_bank(const n4m_config_t* cfg,
                                                    const n4m_operator_bank_t** out) {
    if (cfg == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out = as_core(cfg)->operator_bank;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
N4M_API n4m_status_t n4m_config_get_gating_strategy(const n4m_config_t* cfg,
                                                      const n4m_gating_strategy_t** out) {
    if (cfg == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out = as_core(cfg)->gating_strategy;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
