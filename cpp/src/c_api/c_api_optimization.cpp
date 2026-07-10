// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the native ask/tell hyperparameter optimizer
// (optimization role header, ABI 2.1). Translates opaque handles to the
// internal core classes; catches C++ exceptions and maps them to
// n4m_status_t. No numerical logic lives here — the search algorithm is in
// core/optimization/optimizer.cpp and the CV objective is the existing
// cross_validate_regression.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "n4m/n4m.h"
#include "n4m/optimization.h"

#include "core/common/context.hpp"
#include "core/config.hpp"
#include "core/cross_validation.hpp"
#include "core/method_result.hpp"
#include "core/metrics.hpp"
#include "core/optimization/optimizer.hpp"
#include "core/validation.hpp"

namespace {

namespace opt = ::n4m::core::opt;

inline ::n4m::core::Context* as_core(n4m_context_t* ctx) noexcept {
    return static_cast<::n4m::core::Context*>(ctx);
}
inline const ::n4m::core::ValidationPlan* as_core(const n4m_validation_plan_t* plan) noexcept {
    return static_cast<const ::n4m::core::ValidationPlan*>(plan);
}
inline const opt::Trial* as_trial(const n4m_trial_t* t) noexcept {
    return static_cast<const opt::Trial*>(t);
}

void set_error(n4m_context_t* ctx, const char* msg) noexcept {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->set_error(msg);
    } catch (...) {
    }
}

// Map an n4m_metric_t onto the regression CV metrics. Classification metrics
// are not available from the regression CV path in F0.
bool regression_metric_value(n4m_metric_t metric,
                             const ::n4m::core::RegressionMetrics& m,
                             double* out) {
    switch (metric) {
        case N4M_METRIC_RMSE: *out = m.rmse; return true;
        case N4M_METRIC_MSE:  *out = m.rmse * m.rmse; return true;
        case N4M_METRIC_MAE:  *out = m.mae; return true;
        case N4M_METRIC_R2:   *out = m.r2; return true;
        default: return false;  // classification metrics: NOT_IMPLEMENTED in F0
    }
}

// Pack the per-trial trace (ids, scores, status, rung, duration) into a result
// bag. Shared by n4m_optimizer_get_trials and n4m_finetune_estimator so both
// expose the documented trace consistently.
void pack_trial_trace(const std::vector<std::unique_ptr<n4m_trial_s>>& trials, int64_t since_id,
                      ::n4m::core::MethodResult& out) {
    std::vector<double> ids, scores, statuses, rungs, durations;
    for (const auto& up : trials) {
        if (up->id < since_id) continue;
        ids.push_back(static_cast<double>(up->id));
        scores.push_back(up->has_score ? up->score : std::nan(""));
        statuses.push_back(static_cast<double>(up->status));
        rungs.push_back(static_cast<double>(up->rung));
        durations.push_back(up->duration_seconds);
    }
    const std::int64_t n = static_cast<std::int64_t>(ids.size());
    out.set_double_matrix("trial_ids", std::move(ids), 1, n);
    out.set_double_matrix("trial_scores", std::move(scores), 1, n);
    out.set_double_matrix("trial_status", std::move(statuses), 1, n);
    out.set_double_matrix("trial_rung", std::move(rungs), 1, n);
    out.set_double_matrix("trial_duration", std::move(durations), 1, n);
    out.set_scalar("n_trials", static_cast<double>(n));
}

}  // namespace

extern "C" {

// ==== options ============================================================

N4M_API void n4m_optimizer_options_init(n4m_optimizer_options_t* opts) {
    if (opts == nullptr) return;
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size      = sizeof(n4m_optimizer_options_t);
    opts->sampler          = N4M_SAMPLER_RANDOM;
    opts->pruner           = N4M_PRUNER_NONE;
    opts->direction        = N4M_OPT_AUTO;  // derive from metric unless overridden
    opts->eval_mode        = N4M_EVAL_MEAN;
    opts->metric           = N4M_METRIC_RMSE;
    opts->liar             = N4M_LIAR_NONE;
    opts->n_startup_trials = 10;
    opts->seed             = 0;
    opts->timeout_seconds  = 0.0;
}

// ==== search space =======================================================

N4M_API n4m_status_t n4m_search_space_create(n4m_search_space_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    try {
        *out = new n4m_search_space_s();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_search_space_destroy(n4m_search_space_t* space) {
    try {
        delete space;
    } catch (...) {
    }
}

N4M_API n4m_status_t n4m_search_space_add_int(n4m_search_space_t* space, const char* name,
                                              int64_t low, int64_t high, int64_t step,
                                              int32_t log) {
    if (space == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    if (high < low || step < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (log && low <= 0) return N4M_ERR_INVALID_ARGUMENT;  // log needs positive bounds
    try {
        opt::ParamSpec p;
        p.name = name;
        p.kind = log ? N4M_PARAM_LOG_INT : N4M_PARAM_INT;
        p.low = static_cast<double>(low);
        p.high = static_cast<double>(high);
        p.step = static_cast<double>(step);
        p.is_int = true;
        p.is_log = log != 0;
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_float(n4m_search_space_t* space, const char* name,
                                                double low, double high, double step,
                                                int32_t log) {
    if (space == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    if (!std::isfinite(low) || !std::isfinite(high) || !std::isfinite(step)) {
        return N4M_ERR_INVALID_ARGUMENT;  // reject NaN / Inf bounds
    }
    if (high < low) return N4M_ERR_INVALID_ARGUMENT;
    if (log && (low <= 0.0 || high <= 0.0)) return N4M_ERR_INVALID_ARGUMENT;  // log needs positive bounds
    try {
        opt::ParamSpec p;
        p.name = name;
        p.kind = log ? N4M_PARAM_LOG_FLOAT : N4M_PARAM_FLOAT;
        p.low = low;
        p.high = high;
        p.step = step;
        p.is_int = false;
        p.is_log = log != 0;
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_categorical(n4m_search_space_t* space, const char* name,
                                                      n4m_cat_type_t type, const void* values,
                                                      int32_t n_values) {
    if (space == nullptr || name == nullptr || values == nullptr) return N4M_ERR_NULL_POINTER;
    if (n_values <= 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        opt::ParamSpec p;
        p.name = name;
        p.kind = N4M_PARAM_CATEGORICAL;
        p.cat_type = type;
        for (int32_t i = 0; i < n_values; ++i) {
            switch (type) {
                case N4M_CAT_STR: {
                    const char* const* arr = static_cast<const char* const*>(values);
                    p.labels.emplace_back(arr[i] != nullptr ? arr[i] : "");
                    p.num_values.push_back(static_cast<double>(i));
                    break;
                }
                case N4M_CAT_INT: {
                    const int64_t* arr = static_cast<const int64_t*>(values);
                    p.num_values.push_back(static_cast<double>(arr[i]));
                    p.labels.push_back(std::to_string(arr[i]));
                    break;
                }
                case N4M_CAT_FLOAT: {
                    const double* arr = static_cast<const double*>(values);
                    p.num_values.push_back(arr[i]);
                    p.labels.push_back(std::to_string(arr[i]));
                    break;
                }
                case N4M_CAT_BOOL: {
                    const int32_t* arr = static_cast<const int32_t*>(values);
                    p.num_values.push_back(arr[i] ? 1.0 : 0.0);
                    p.labels.emplace_back(arr[i] ? "true" : "false");
                    break;
                }
                default:
                    return N4M_ERR_INVALID_ARGUMENT;
            }
        }
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_ordinal(n4m_search_space_t* space, const char* name,
                                                  const double* values, int32_t n_values) {
    if (space == nullptr || name == nullptr || values == nullptr) return N4M_ERR_NULL_POINTER;
    if (n_values <= 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        opt::ParamSpec p;
        p.name = name;
        p.kind = N4M_PARAM_ORDINAL;
        for (int32_t i = 0; i < n_values; ++i) {
            p.num_values.push_back(values[i]);
            p.labels.push_back(std::to_string(values[i]));
        }
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_sorted_tuple(n4m_search_space_t* space, const char* name,
                                                       int32_t length, double low, double high,
                                                       int32_t element_is_int) {
    if (space == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    if (length <= 0 || high < low) return N4M_ERR_INVALID_ARGUMENT;
    if (!std::isfinite(low) || !std::isfinite(high)) return N4M_ERR_INVALID_ARGUMENT;
    try {
        opt::ParamSpec p;
        p.name = name;
        p.kind = N4M_PARAM_SORTED_TUPLE;
        p.low = low;
        p.high = high;
        p.tuple_length = length;
        p.tuple_element_is_int = element_is_int != 0;
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_constraint(n4m_search_space_t* space,
                                                     n4m_constraint_kind_t kind,
                                                     const char* const* param_refs,
                                                     const char* const* label_refs,
                                                     int32_t n_refs) {
    if (space == nullptr || param_refs == nullptr) return N4M_ERR_NULL_POINTER;
    if (n_refs <= 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        opt::Constraint c;
        c.kind = kind;
        for (int32_t i = 0; i < n_refs; ++i) {
            c.param_refs.emplace_back(param_refs[i] != nullptr ? param_refs[i] : "");
            c.label_refs.emplace_back(
                (label_refs != nullptr && label_refs[i] != nullptr) ? label_refs[i] : "");
        }

        if (kind == N4M_CONSTRAINT_CONDITION_IN || kind == N4M_CONSTRAINT_CONDITION_NOT_IN) {
            // F0: exactly {child, parent}; activation applied per child ParamSpec.
            if (n_refs != 2) return N4M_ERR_UNSUPPORTED;
            opt::ParamSpec* child = space->find(c.param_refs[0]);
            if (child == nullptr) return N4M_ERR_INVALID_ARGUMENT;  // unknown child param
            const bool is_in = (kind == N4M_CONSTRAINT_CONDITION_IN);
            const std::string& parent = c.param_refs[1];
            const std::string& label = c.label_refs[1];
            if (!child->cond_parent.empty() &&
                (child->cond_parent != parent || child->cond_is_in != is_in)) {
                return N4M_ERR_UNSUPPORTED;  // conflicting condition on the same child
            }
            child->cond_parent = parent;
            child->cond_is_in = is_in;
            if (!label.empty()) child->cond_labels.push_back(label);
        }

        space->constraints.push_back(std::move(c));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_num_params(const n4m_search_space_t* space, int32_t* out_n) {
    if (space == nullptr || out_n == nullptr) return N4M_ERR_NULL_POINTER;
    *out_n = static_cast<int32_t>(space->params.size());
    return N4M_OK;
}

// ==== optimizer ==========================================================

N4M_API n4m_status_t n4m_optimizer_create(n4m_context_t* ctx, const n4m_search_space_t* space,
                                          const n4m_optimizer_options_t* opts,
                                          n4m_optimizer_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (ctx == nullptr || space == nullptr || opts == nullptr) {
        set_error(ctx, "null pointer in n4m_optimizer_create");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        if (opts->struct_size < sizeof(uint64_t)) {
            set_error(ctx, "n4m_optimizer_options_t.struct_size unset (call n4m_optimizer_options_init)");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        n4m_optimizer_options_t o;
        n4m_optimizer_options_init(&o);  // fill defaults, then overlay the caller's bytes
        const std::size_t copy = std::min(static_cast<std::size_t>(opts->struct_size), sizeof(o));
        std::memcpy(&o, opts, copy);
        o.struct_size = sizeof(o);

        n4m_status_t st = N4M_OK;
        auto impl = opt::make_optimizer(*space, o, &st);
        if (impl == nullptr) {
            set_error(ctx, "unsupported sampler/pruner (reserved for a later phase)");
            return st;
        }
        auto handle = std::make_unique<n4m_optimizer_s>();
        handle->impl = std::move(impl);
        *out = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_optimizer_create");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_optimizer_create");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_optimizer_destroy(n4m_optimizer_t* opt) {
    try {
        delete opt;
    } catch (...) {
    }
}

N4M_API n4m_status_t n4m_optimizer_enqueue(n4m_optimizer_t* opt, const char* const* names,
                                           const double* values, int32_t n) {
    if (opt == nullptr || names == nullptr || values == nullptr) return N4M_ERR_NULL_POINTER;
    if (n < 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        std::vector<std::pair<std::string, double>> params;
        params.reserve(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            params.emplace_back(names[i] != nullptr ? names[i] : "", values[i]);
        }
        return opt->impl->enqueue(std::move(params));
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_ask(n4m_optimizer_t* opt, n4m_trial_t** out_trial) {
    if (opt == nullptr || out_trial == nullptr) return N4M_ERR_NULL_POINTER;
    *out_trial = nullptr;
    try {
        n4m_trial_s* t = nullptr;
        const n4m_status_t st = opt->impl->ask(&t);
        if (st != N4M_OK) return st;
        *out_trial = t;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_ask_batch(n4m_optimizer_t* opt, int32_t n,
                                             n4m_trial_t** out_trials, int32_t* out_count) {
    if (opt == nullptr || out_trials == nullptr || out_count == nullptr) return N4M_ERR_NULL_POINTER;
    *out_count = 0;
    if (n < 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        for (int32_t i = 0; i < n; ++i) {
            n4m_trial_s* t = nullptr;
            const n4m_status_t st = opt->impl->ask(&t);
            if (st != N4M_OK) return st;
            out_trials[i] = t;
            ++(*out_count);
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_tell(n4m_optimizer_t* opt, int64_t trial_id, double score) {
    if (opt == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        return opt->impl->tell_result(trial_id, N4M_TRIAL_COMPLETED, score);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_tell_result(n4m_optimizer_t* opt, int64_t trial_id,
                                               n4m_trial_status_t status, double score,
                                               const char* error) {
    (void)error;  // stored per-trial in a later phase; F0 records status + score
    if (opt == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        return opt->impl->tell_result(trial_id, status, score);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_tell_intermediate(n4m_optimizer_t* opt, int64_t trial_id,
                                                     int32_t step, double score,
                                                     int32_t* out_should_prune) {
    if (opt == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        return opt->impl->tell_intermediate(trial_id, step, score, out_should_prune);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_best(const n4m_optimizer_t* opt, n4m_trial_t** out_best,
                                        double* out_score) {
    if (opt == nullptr || out_best == nullptr) return N4M_ERR_NULL_POINTER;
    *out_best = nullptr;
    try {
        double score = 0.0;
        n4m_trial_s* b = opt->impl->best(&score);
        if (b == nullptr) return N4M_ERR_NOT_FITTED;  // no completed trial yet
        *out_best = b;
        if (out_score != nullptr) *out_score = score;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_get_trials(const n4m_optimizer_t* opt, int64_t since_id,
                                              n4m_method_result_t** out) {
    if (opt == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    try {
        auto handle = std::make_unique<n4m_method_result_s>();
        pack_trial_trace(opt->impl->trials(), since_id, *handle);
        *out = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_save(const n4m_optimizer_t* opt, n4m_array_t** out_blob) {
    (void)opt;
    if (out_blob != nullptr) *out_blob = nullptr;
    return N4M_ERR_NOT_IMPLEMENTED;  // resume lands in F7
}

N4M_API n4m_status_t n4m_optimizer_load(n4m_context_t* ctx, const uint8_t* blob, uint64_t n,
                                        n4m_optimizer_t** out) {
    (void)ctx;
    (void)blob;
    (void)n;
    if (out != nullptr) *out = nullptr;
    return N4M_ERR_NOT_IMPLEMENTED;  // resume lands in F7
}

// ==== trial accessors ====================================================

N4M_API n4m_status_t n4m_trial_get_id(const n4m_trial_t* trial, int64_t* out) {
    if (trial == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = as_trial(trial)->id;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_trial_get_int(const n4m_trial_t* trial, const char* name, int64_t* out) {
    if (trial == nullptr || name == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const opt::TrialParam* tp = as_trial(trial)->find(name);
        if (tp == nullptr) return N4M_ERR_INVALID_ARGUMENT;
        *out = static_cast<int64_t>(std::llround(tp->value));
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_trial_get_float(const n4m_trial_t* trial, const char* name, double* out) {
    if (trial == nullptr || name == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const opt::TrialParam* tp = as_trial(trial)->find(name);
        if (tp == nullptr) return N4M_ERR_INVALID_ARGUMENT;
        *out = tp->value;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_trial_get_category(const n4m_trial_t* trial, const char* name,
                                            int32_t* out_index, const char** out_label) {
    if (trial == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const opt::TrialParam* tp = as_trial(trial)->find(name);
        if (tp == nullptr) return N4M_ERR_INVALID_ARGUMENT;
        if (out_index != nullptr) *out_index = tp->cat_index;
        if (out_label != nullptr) *out_label = tp->cat_label.c_str();  // core-owned
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_trial_is_active(const n4m_trial_t* trial, const char* name, int32_t* out) {
    if (trial == nullptr || name == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const opt::TrialParam* tp = as_trial(trial)->find(name);
        if (tp == nullptr) return N4M_ERR_INVALID_ARGUMENT;
        *out = tp->active ? 1 : 0;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_trial_get_rung(const n4m_trial_t* trial, int32_t* out) {
    if (trial == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = as_trial(trial)->rung;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_trial_get_status(const n4m_trial_t* trial, n4m_trial_status_t* out) {
    if (trial == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = as_trial(trial)->status;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_trial_get_duration(const n4m_trial_t* trial, double* out_seconds) {
    if (trial == nullptr || out_seconds == nullptr) return N4M_ERR_NULL_POINTER;
    *out_seconds = as_trial(trial)->duration_seconds;
    return N4M_OK;
}

// ==== pure-native single-level driver ====================================

N4M_API n4m_status_t n4m_finetune_estimator(n4m_context_t* ctx, n4m_algorithm_t estimator,
                                            const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
                                            const n4m_validation_plan_t* plan,
                                            const n4m_search_space_t* space,
                                            const n4m_optimizer_options_t* opts, int32_t n_trials,
                                            n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || X == nullptr || Y == nullptr || plan == nullptr || space == nullptr ||
        opts == nullptr) {
        set_error(ctx, "null pointer in n4m_finetune_estimator");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_trials <= 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        if (opts->struct_size < sizeof(uint64_t)) {
            set_error(ctx, "n4m_optimizer_options_t.struct_size unset (call n4m_optimizer_options_init)");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        n4m_optimizer_options_t o;
        n4m_optimizer_options_init(&o);  // fill defaults, then overlay the caller's bytes
        const std::size_t copy = std::min(static_cast<std::size_t>(opts->struct_size), sizeof(o));
        std::memcpy(&o, opts, copy);
        o.struct_size = sizeof(o);

        n4m_status_t st = N4M_OK;
        auto optimizer = opt::make_optimizer(*space, o, &st);
        if (optimizer == nullptr) {
            set_error(ctx, "unsupported sampler/pruner in n4m_finetune_estimator");
            return st;
        }

        // F0 tunes only the estimator's n_components; reject any other declared
        // hyperparameter rather than silently ignoring it (later phases widen this).
        for (const auto& p : space->params) {
            if (p.name != "n_components") {
                set_error(ctx, "n4m_finetune_estimator (F0) only tunes 'n_components'");
                return N4M_ERR_UNSUPPORTED;
            }
        }

        ::n4m::core::Context& core_ctx = *as_core(ctx);
        const ::n4m::core::ValidationPlan& vp = *as_core(plan);

        for (int32_t i = 0; i < n_trials; ++i) {
            n4m_trial_s* t = nullptr;
            const n4m_status_t ask_st = optimizer->ask(&t);
            if (ask_st != N4M_OK) return ask_st;

            ::n4m::core::Config cfg;
            cfg.algorithm = estimator;
            const opt::TrialParam* nc = t->find("n_components");
            if (nc != nullptr) {
                int comp = static_cast<int>(std::llround(nc->value));
                if (comp < 1) comp = 1;
                cfg.n_components = comp;
            }

            ::n4m::core::CrossValidationResult cv;
            const n4m_status_t cv_st = ::n4m::core::cross_validate_regression(
                core_ctx, cfg, *X, *Y, vp, cv);
            if (cv_st != N4M_OK) {
                (void)optimizer->tell_result(t->id, N4M_TRIAL_FAILED, 0.0);
                continue;
            }
            double score = 0.0;
            if (!regression_metric_value(o.metric, cv.metrics, &score)) {
                set_error(ctx, "metric not available from the regression CV path in F0");
                return N4M_ERR_NOT_IMPLEMENTED;
            }
            (void)optimizer->tell_result(t->id, N4M_TRIAL_COMPLETED, score);
        }

        double best_score = 0.0;
        n4m_trial_s* best = optimizer->best(&best_score);
        if (best == nullptr) {
            set_error(ctx, "no trial completed successfully in n4m_finetune_estimator");
            return N4M_ERR_NOT_FITTED;
        }

        auto handle = std::make_unique<n4m_method_result_s>();
        for (const auto& kv : best->params) {
            handle->set_scalar(std::string("best.") + kv.first, kv.second.value);
        }
        handle->set_scalar("best_score", best_score);
        pack_trial_trace(optimizer->trials(), 0, *handle);  // ids/scores/status/rung/duration + n_trials
        handle->set_scalar("metric", static_cast<double>(o.metric));
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_finetune_estimator");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_finetune_estimator");
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
