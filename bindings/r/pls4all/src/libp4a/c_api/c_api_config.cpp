// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for p4a_config_t. Every setter validates its input
// and leaves the config UNCHANGED on rejection (idempotent failure). Every
// wrapper has a try/catch around its body — see Direction_Technique §4 for
// the no-exception-across-ABI rule.

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <exception>
#include <new>

#include "pls4all/p4a.h"

#include "core/config.hpp"

namespace {

inline ::pls4all::core::Config* as_core(p4a_config_t* cfg) noexcept {
    return static_cast<::pls4all::core::Config*>(cfg);
}
inline const ::pls4all::core::Config* as_core(const p4a_config_t* cfg) noexcept {
    return static_cast<const ::pls4all::core::Config*>(cfg);
}

inline bool in_range_algorithm(p4a_algorithm_t a) noexcept {
    return a >= P4A_ALGO_PLS_REGRESSION && a <= P4A_ALGO_PCR;
}
inline bool in_range_solver(p4a_solver_t s) noexcept {
    return s >= P4A_SOLVER_NIPALS && s <= P4A_SOLVER_RANDOMIZED_SVD;
}
inline bool in_range_deflation(p4a_deflation_t d) noexcept {
    return d >= P4A_DEFLATION_REGRESSION && d <= P4A_DEFLATION_ORTHOGONAL;
}
inline bool in_range_dtype(p4a_dtype_t d) noexcept {
    return d == P4A_DTYPE_F64 || d == P4A_DTYPE_F32
        || d == P4A_DTYPE_I32 || d == P4A_DTYPE_I64;
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------------ */
/* Lifecycle                                                                */
/* ------------------------------------------------------------------------ */

P4A_API p4a_status_t p4a_config_create(p4a_config_t** out_cfg) {
    if (out_cfg == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out_cfg = new p4a_config_s{};
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        *out_cfg = nullptr;
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out_cfg = nullptr;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_config_destroy(p4a_config_t* cfg) {
    try {
        delete cfg;
    } catch (...) {
        // Best-effort.
    }
}

P4A_API p4a_status_t p4a_config_clone(const p4a_config_t* src,
                                       p4a_config_t** out_dst) {
    if (src == nullptr || out_dst == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        auto* dst = new p4a_config_s{};
        *static_cast<::pls4all::core::Config*>(dst) = *as_core(src);
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

/* ------------------------------------------------------------------------ */
/* Setters                                                                  */
/* ------------------------------------------------------------------------ */

#define P4A_CFG_TRY_BEGIN \
    try {
#define P4A_CFG_TRY_END \
    } catch (...) { return P4A_ERR_INTERNAL; }

P4A_API p4a_status_t p4a_config_set_algorithm(p4a_config_t* cfg, p4a_algorithm_t v) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (!in_range_algorithm(v)) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->algorithm = v;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_solver(p4a_config_t* cfg, p4a_solver_t v) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (!in_range_solver(v)) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->solver = v;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_deflation(p4a_config_t* cfg, p4a_deflation_t v) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (!in_range_deflation(v)) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->deflation = v;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_n_components(p4a_config_t* cfg, int32_t n) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    // Reject non-positive values. There is no upper cap at the C ABI level;
    // implementations may degrade performance gracefully for very large n,
    // and Phase 1+ will surface a P4A_ERR_INVALID_ARGUMENT at fit time if
    // n > rank(X).
    P4A_CFG_TRY_BEGIN
        if (n < 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->n_components = n;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_center_x(p4a_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->center_x = enabled;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_scale_x(p4a_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->scale_x = enabled;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_center_y(p4a_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->center_y = enabled;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_scale_y(p4a_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->scale_y = enabled;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_tol(p4a_config_t* cfg, double tol) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (std::isnan(tol) || std::isinf(tol) || tol <= 0.0) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        as_core(cfg)->tol = tol;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_max_iter(p4a_config_t* cfg, int32_t max_iter) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (max_iter < 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->max_iter = max_iter;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_store_scores(p4a_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->store_scores = enabled;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_store_diagnostics(p4a_config_t* cfg, int32_t enabled) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (enabled != 0 && enabled != 1) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->store_diagnostics = enabled;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_dtype(p4a_config_t* cfg, p4a_dtype_t v) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        if (!in_range_dtype(v)) return P4A_ERR_INVALID_ARGUMENT;
        as_core(cfg)->dtype = v;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_pipeline(p4a_config_t* cfg,
                                              const p4a_pipeline_t* pipe) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        as_core(cfg)->pipeline = pipe;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_operator_bank(p4a_config_t* cfg,
                                                    const p4a_operator_bank_t* bank) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        as_core(cfg)->operator_bank = bank;
        return P4A_OK;
    P4A_CFG_TRY_END
}

P4A_API p4a_status_t p4a_config_set_gating_strategy(p4a_config_t* cfg,
                                                      const p4a_gating_strategy_t* gate) {
    if (cfg == nullptr) return P4A_ERR_NULL_POINTER;
    P4A_CFG_TRY_BEGIN
        as_core(cfg)->gating_strategy = gate;
        return P4A_OK;
    P4A_CFG_TRY_END
}

#undef P4A_CFG_TRY_BEGIN
#undef P4A_CFG_TRY_END

/* ------------------------------------------------------------------------ */
/* Getters                                                                  */
/* ------------------------------------------------------------------------ */

#define P4A_GETTER(NAME, T, FIELD)                                          \
    P4A_API p4a_status_t p4a_config_get_##NAME(const p4a_config_t* cfg,     \
                                                T* out) {                    \
        if (cfg == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;  \
        try {                                                                \
            *out = as_core(cfg)->FIELD;                                      \
            return P4A_OK;                                                   \
        } catch (...) {                                                      \
            return P4A_ERR_INTERNAL;                                         \
        }                                                                    \
    }

P4A_GETTER(algorithm,         p4a_algorithm_t, algorithm)
P4A_GETTER(solver,            p4a_solver_t,    solver)
P4A_GETTER(deflation,         p4a_deflation_t, deflation)
P4A_GETTER(n_components,      int32_t,         n_components)
P4A_GETTER(center_x,          int32_t,         center_x)
P4A_GETTER(scale_x,           int32_t,         scale_x)
P4A_GETTER(center_y,          int32_t,         center_y)
P4A_GETTER(scale_y,           int32_t,         scale_y)
P4A_GETTER(tol,               double,          tol)
P4A_GETTER(max_iter,          int32_t,         max_iter)
P4A_GETTER(store_scores,      int32_t,         store_scores)
P4A_GETTER(store_diagnostics, int32_t,         store_diagnostics)
P4A_GETTER(dtype,             p4a_dtype_t,     dtype)

#undef P4A_GETTER

P4A_API p4a_status_t p4a_config_get_pipeline(const p4a_config_t* cfg,
                                              const p4a_pipeline_t** out) {
    if (cfg == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out = as_core(cfg)->pipeline;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}
P4A_API p4a_status_t p4a_config_get_operator_bank(const p4a_config_t* cfg,
                                                    const p4a_operator_bank_t** out) {
    if (cfg == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out = as_core(cfg)->operator_bank;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}
P4A_API p4a_status_t p4a_config_get_gating_strategy(const p4a_config_t* cfg,
                                                      const p4a_gating_strategy_t** out) {
    if (cfg == nullptr || out == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out = as_core(cfg)->gating_strategy;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

}  // extern "C"
