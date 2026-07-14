// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the native ask/tell hyperparameter optimizer
// (optimization role header, ABI 2.2). Translates opaque handles to the
// internal core classes; catches C++ exceptions and maps them to
// n4m_status_t. No numerical logic lives here — the search algorithm is in
// core/optimization/optimizer.cpp and the CV objective is the existing
// cross_validate_regression.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "n4m/n4m.h"
#include "n4m/optimization.h"

#include "core/common/context.hpp"
#include "core/config.hpp"
#include "core/cross_validation.hpp"
#include "core/method_result.hpp"
#include "core/metrics.hpp"
#include "core/model.hpp"
#include "core/optimization/optimizer.hpp"
#include "core/validation.hpp"

namespace {

namespace opt = ::n4m::core::opt;

constexpr int64_t kMaxConsecutiveBinary64Integer = 9007199254740992LL;  // 2^53

bool is_consecutive_binary64_integer(int64_t value) noexcept {
    return value >= -kMaxConsecutiveBinary64Integer &&
           value <= kMaxConsecutiveBinary64Integer;
}

bool is_supported_integer_tuple_bound(double value) noexcept {
    return std::isfinite(value) && std::trunc(value) == value &&
           std::abs(value) <= static_cast<double>(kMaxConsecutiveBinary64Integer);
}

// Strict RFC 3629 UTF-8 validation for every text value that can be persisted
// in the owning HPO trace. Reject overlong encodings, surrogate code points and
// values above U+10FFFF at the C boundary so a successful tell/build can always
// be decoded by every binding later.
bool valid_utf8(std::string_view text) noexcept {
    const auto continuation = [](unsigned char byte) {
        return byte >= 0x80U && byte <= 0xBFU;
    };
    std::size_t index = 0;
    while (index < text.size()) {
        const auto first = static_cast<unsigned char>(text[index]);
        if (first <= 0x7FU) {
            ++index;
            continue;
        }
        if (first >= 0xC2U && first <= 0xDFU) {
            if (index + 1 >= text.size() ||
                !continuation(static_cast<unsigned char>(text[index + 1]))) {
                return false;
            }
            index += 2;
            continue;
        }
        if (first >= 0xE0U && first <= 0xEFU) {
            if (index + 2 >= text.size()) return false;
            const auto second = static_cast<unsigned char>(text[index + 1]);
            const auto third = static_cast<unsigned char>(text[index + 2]);
            if (!continuation(third) ||
                (first == 0xE0U && (second < 0xA0U || second > 0xBFU)) ||
                (first == 0xEDU && (second < 0x80U || second > 0x9FU)) ||
                ((first != 0xE0U && first != 0xEDU) && !continuation(second))) {
                return false;
            }
            index += 3;
            continue;
        }
        if (first >= 0xF0U && first <= 0xF4U) {
            if (index + 3 >= text.size()) return false;
            const auto second = static_cast<unsigned char>(text[index + 1]);
            const auto third = static_cast<unsigned char>(text[index + 2]);
            const auto fourth = static_cast<unsigned char>(text[index + 3]);
            if (!continuation(third) || !continuation(fourth) ||
                (first == 0xF0U && (second < 0x90U || second > 0xBFU)) ||
                (first == 0xF4U && (second < 0x80U || second > 0x8FU)) ||
                ((first != 0xF0U && first != 0xF4U) && !continuation(second))) {
                return false;
            }
            index += 4;
            continue;
        }
        return false;
    }
    return true;
}

std::string round_trip_double_label(double value) {
    char buffer[64];
    const int written =
        std::snprintf(buffer, sizeof(buffer), "%.*g",
                      std::numeric_limits<double>::max_digits10, value);
    if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(buffer)) return {};
    return std::string(buffer, static_cast<std::size_t>(written));
}

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

// Normalize the caller-owned, forward-compatible options prefix into the current
// ABI layout. ABI 2.1 is the first published layout, so there is no valid shorter
// prefix to preserve; larger future layouts are accepted and their tail ignored.
n4m_status_t normalize_optimizer_options(n4m_context_t* ctx,
                                         const n4m_optimizer_options_t* opts,
                                         n4m_optimizer_options_t* out) noexcept {
    if (opts->struct_size < sizeof(n4m_optimizer_options_t)) {
        set_error(ctx,
                  "n4m_optimizer_options_t.struct_size is smaller than the ABI 2.1 layout");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (opts->struct_size > std::numeric_limits<std::size_t>::max()) {
        set_error(ctx,
                  "n4m_optimizer_options_t.struct_size exceeds the addressable size_t range");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_optimizer_options_init(out);
    // ABI 2.1 has no valid shorter prefix. The guard above therefore proves
    // that the complete current layout is readable; copy it directly instead
    // of narrowing the caller's uint64_t before taking a minimum on ILP32.
    std::memcpy(out, opts, sizeof(*out));
    out->struct_size = sizeof(*out);
    return N4M_OK;
}

bool is_regression_metric(n4m_metric_t metric) noexcept {
    return metric == N4M_METRIC_RMSE || metric == N4M_METRIC_MSE ||
           metric == N4M_METRIC_MAE || metric == N4M_METRIC_R2;
}

bool is_candidate_specific_cv_failure(n4m_status_t status) noexcept {
    return status == N4M_ERR_INVALID_ARGUMENT ||
           status == N4M_ERR_NUMERICAL_FAILURE ||
           status == N4M_ERR_CONVERGENCE_FAILED;
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

// ==== MT8 finetune-estimator registry ====================================
// Closed, internal description of which native algorithms n4m_finetune_estimator
// can drive as a GENERIC regression estimator: exactly one core::fit_model route
// scored by a plain regression cross-validation. Each entry carries only
// eligibility + task (regression) + the tunable-axis schema. The numerical
// solver/deflation recipe is NOT stored here — it stays the sole authority of
// core::fit_model and is applied through core::canonical_regression_routing. No
// callbacks, plugins, nested CV or second DAG. Any algorithm absent from the
// table (classification chassis PLS_DA / OPLS_DA, non-generic MB_PLS / LW_PLS /
// AOM_PLS, and every invalid enum) is rejected up front with N4M_ERR_UNSUPPORTED.
enum class FtAxis { Forbidden, Optional, Required };

struct FinetuneEntry {
    n4m_algorithm_t algorithm;
    const char*     name;             // provenance label for diagnostics
    FtAxis          n_components;      // INT (non-log) axis, domain [1, INT32_MAX]
    FtAxis          sparsity_lambda;   // FLOAT / safe LOG_FLOAT axis, domain [0, 1)
    bool            require_any;       // at least one declared axis must be present
};

constexpr FinetuneEntry kFinetuneRegistry[] = {
    {N4M_ALGO_PLS_REGRESSION, "PLS_REGRESSION", FtAxis::Required, FtAxis::Forbidden, false},
    {N4M_ALGO_PLS_CANONICAL,  "PLS_CANONICAL",  FtAxis::Required, FtAxis::Forbidden, false},
    {N4M_ALGO_PLS_SVD,        "PLS_SVD",        FtAxis::Required, FtAxis::Forbidden, false},
    {N4M_ALGO_OPLS,           "OPLS",           FtAxis::Required, FtAxis::Forbidden, false},
    {N4M_ALGO_PCR,            "PCR",            FtAxis::Required, FtAxis::Forbidden, false},
    {N4M_ALGO_SPARSE_PLS,     "SPARSE_PLS",     FtAxis::Optional, FtAxis::Optional,  true},
};

const FinetuneEntry* find_finetune_entry(n4m_algorithm_t algorithm) noexcept {
    for (const FinetuneEntry& entry : kFinetuneRegistry) {
        if (entry.algorithm == algorithm) return &entry;
    }
    return nullptr;
}

// Preflight the ordered search space against a registered axis schema. Every
// structural violation (unknown axis, duplicate axis, wrong kind/log, out-of-
// domain bound, bad step, missing required axis) is a deterministic, stable
// N4M_ERR_UNSUPPORTED returned before any study is created — the same status and
// precedence the historical n_components-only path used. Never mutates or samples.
n4m_status_t validate_finetune_space(n4m_context_t* ctx, const FinetuneEntry& entry,
                                     const opt::SearchSpace& space) noexcept {
    for (const opt::Constraint& constraint : space.constraints) {
        if (constraint.kind == N4M_CONSTRAINT_CONDITION_IN ||
            constraint.kind == N4M_CONSTRAINT_CONDITION_NOT_IN) {
            set_error(ctx,
                      "conditional axes are unsupported by n4m_finetune_estimator; "
                      "its registered axes are numeric and cannot be condition parents");
            return N4M_ERR_UNSUPPORTED;
        }
    }
    bool has_n_components = false;
    bool has_sparsity_lambda = false;
    for (const opt::ParamSpec& p : space.params) {
        if (p.name == "n_components") {
            if (entry.n_components == FtAxis::Forbidden) {
                set_error(ctx, "finetune estimator does not tune 'n_components'");
                return N4M_ERR_UNSUPPORTED;
            }
            if (has_n_components) {
                set_error(ctx, "finetune search space repeats axis 'n_components'");
                return N4M_ERR_UNSUPPORTED;
            }
            has_n_components = true;
            if (p.kind != N4M_PARAM_INT || !p.is_int || p.is_log ||
                !std::isfinite(p.low) || !std::isfinite(p.high) ||
                !std::isfinite(p.step) || std::trunc(p.low) != p.low ||
                std::trunc(p.high) != p.high || std::trunc(p.step) != p.step ||
                p.low < 1.0 || p.low > p.high ||
                p.high > static_cast<double>(std::numeric_limits<int32_t>::max()) ||
                p.step < 1.0) {
                set_error(ctx,
                          "finetune axis 'n_components' must be a non-log INT axis with "
                          "low >= 1, high <= INT32_MAX and step >= 1");
                return N4M_ERR_UNSUPPORTED;
            }
        } else if (p.name == "sparsity_lambda") {
            if (entry.sparsity_lambda == FtAxis::Forbidden) {
                set_error(ctx, "finetune estimator does not tune 'sparsity_lambda'");
                return N4M_ERR_UNSUPPORTED;
            }
            if (has_sparsity_lambda) {
                set_error(ctx, "finetune search space repeats axis 'sparsity_lambda'");
                return N4M_ERR_UNSUPPORTED;
            }
            has_sparsity_lambda = true;
            const bool linear = p.kind == N4M_PARAM_FLOAT;
            const bool logarithmic = p.kind == N4M_PARAM_LOG_FLOAT;
            const bool invalid_linear_step =
                linear &&
                (p.step < 0.0 ||
                 (p.step > 0.0 &&
                  (!std::isfinite((p.high - p.low) / p.step) ||
                   (p.high > p.low && p.low + p.step == p.low))));
            const bool invalid_log_step = logarithmic && p.step != 0.0;
            if ((!linear && !logarithmic) || !std::isfinite(p.low) ||
                !std::isfinite(p.high) || !std::isfinite(p.step) || p.is_int ||
                p.is_log != logarithmic || p.low < 0.0 || p.low > p.high ||
                p.high >= 1.0 || (logarithmic && p.low <= 0.0) ||
                invalid_linear_step || invalid_log_step) {
                set_error(ctx,
                          "finetune axis 'sparsity_lambda' must be a FLOAT axis with "
                          "0 <= low <= high < 1; FLOAT requires step >= 0 and a "
                          "representable grid, LOG_FLOAT requires low > 0 and step = 0");
                return N4M_ERR_UNSUPPORTED;
            }
        } else {
            set_error(ctx, "finetune search space declares an unknown axis");
            return N4M_ERR_UNSUPPORTED;
        }
    }
    if (entry.n_components == FtAxis::Required && !has_n_components) {
        set_error(ctx, "finetune estimator requires an 'n_components' axis");
        return N4M_ERR_UNSUPPORTED;
    }
    if (entry.sparsity_lambda == FtAxis::Required && !has_sparsity_lambda) {
        set_error(ctx, "finetune estimator requires a 'sparsity_lambda' axis");
        return N4M_ERR_UNSUPPORTED;
    }
    if (entry.require_any && !has_n_components && !has_sparsity_lambda) {
        set_error(ctx,
                  "finetune estimator requires at least one of 'n_components' or "
                  "'sparsity_lambda'");
        return N4M_ERR_UNSUPPORTED;
    }
    return N4M_OK;
}

// Build the per-trial Config for a registered candidate: algorithm + canonical
// routing (via the fit_model authority) + the tuned axes read from the trial.
// Omitted or optional-inactive axes keep their documented Config defaults
// (n_components = 2, sparsity_lambda = 0.0). Returns N4M_ERR_INVALID_ARGUMENT when
// a REQUIRED axis is inactive or an ACTIVE tuned value is not finite / of the
// exact type/domain — the driver turns that into an INVALID_CANDIDATE FAILED
// terminal — and N4M_ERR_INTERNAL only if a non-routable algorithm ever slips in.
n4m_status_t build_finetune_config(const FinetuneEntry& entry, const n4m_trial_s& trial,
                                   ::n4m::core::Config& cfg) noexcept {
    cfg.algorithm = entry.algorithm;
    n4m_solver_t solver{};
    n4m_deflation_t deflation{};
    if (!::n4m::core::canonical_regression_routing(entry.algorithm, solver, deflation)) {
        return N4M_ERR_INTERNAL;  // registry lists only routable algorithms
    }
    cfg.solver = solver;
    cfg.deflation = deflation;

    if (entry.n_components != FtAxis::Forbidden) {
        const opt::TrialParam* tp = trial.find("n_components");
        if (tp != nullptr && tp->active) {
            if (!std::isfinite(tp->value) || std::trunc(tp->value) != tp->value ||
                tp->value < 1.0 ||
                tp->value > static_cast<double>(std::numeric_limits<int32_t>::max())) {
                return N4M_ERR_INVALID_ARGUMENT;
            }
            cfg.n_components = static_cast<std::int32_t>(tp->value);
        } else if (entry.n_components == FtAxis::Required) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    if (entry.sparsity_lambda != FtAxis::Forbidden) {
        const opt::TrialParam* tp = trial.find("sparsity_lambda");
        if (tp != nullptr && tp->active) {
            if (!std::isfinite(tp->value) || tp->value < 0.0 || tp->value >= 1.0) {
                return N4M_ERR_INVALID_ARGUMENT;
            }
            cfg.sparsity_lambda = tp->value;
        } else if (entry.sparsity_lambda == FtAxis::Required) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

constexpr std::string_view kTrialErrorPrefix = N4M_TRIAL_ERROR_WIRE_PREFIX;
constexpr std::string_view kAnyTrialErrorPrefix = "n4m.error.v";

bool valid_trial_error_code(std::string_view code) noexcept {
    if (code.size() < 2 || code.size() > 64 || code.front() < 'A' || code.front() > 'Z') {
        return false;
    }
    for (const char ch : code) {
        const bool valid = (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
        if (!valid) return false;
    }
    return true;
}

n4m_status_t decode_trial_error(n4m_trial_status_t status, const char* raw,
                                opt::TrialError& out, bool& has_error) {
    has_error = false;
    const bool error_status =
        status == N4M_TRIAL_FAILED || status == N4M_TRIAL_CANCELLED;
    const std::string_view text = raw == nullptr ? std::string_view{} : std::string_view(raw);
    if (!valid_utf8(text)) return N4M_ERR_INVALID_ARGUMENT;
    if (!error_status) {
        return text.empty() ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
    }

    has_error = true;
    if (text.empty()) {
        out = status == N4M_TRIAL_FAILED
                  ? opt::TrialError{"OBJECTIVE_ERROR", "trial failed", false}
                  : opt::TrialError{"BUDGET_CANCELLED", "trial cancelled", false};
        return N4M_OK;
    }
    if (text.rfind(kAnyTrialErrorPrefix, 0) == 0 && text.rfind(kTrialErrorPrefix, 0) != 0) {
        return N4M_ERR_INVALID_ARGUMENT;  // unknown structured-error version
    }
    if (text.rfind(kTrialErrorPrefix, 0) != 0) {
        out = {status == N4M_TRIAL_FAILED ? "OBJECTIVE_ERROR" : "BUDGET_CANCELLED",
               std::string(text), false};
        return N4M_OK;
    }

    const std::string_view payload = text.substr(kTrialErrorPrefix.size());
    const std::size_t code_end = payload.find('|');
    if (code_end == std::string_view::npos) return N4M_ERR_INVALID_ARGUMENT;
    const std::size_t retry_end = payload.find('|', code_end + 1);
    if (retry_end == std::string_view::npos) return N4M_ERR_INVALID_ARGUMENT;
    const std::string_view code = payload.substr(0, code_end);
    const std::string_view retryable =
        payload.substr(code_end + 1, retry_end - code_end - 1);
    const std::string_view message = payload.substr(retry_end + 1);
    if (!valid_trial_error_code(code) || (retryable != "0" && retryable != "1") ||
        message.empty()) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out = {std::string(code), std::string(message), retryable == "1"};
    return N4M_OK;
}

void append_utf8(const std::string& value, std::vector<std::int32_t>& bytes) {
    for (const char ch : value) {
        const auto byte = static_cast<unsigned char>(ch);
        bytes.push_back(static_cast<std::int32_t>(byte));
    }
}

// Pack an owning trace snapshot. The original scalar/matrix keys remain for
// compatibility; trace v1 adds lossless ids and flattened variable-length
// records whose offsets are int64 vectors.
void pack_trial_trace(const std::vector<std::unique_ptr<n4m_trial_s>>& trials,
                      const opt::SearchSpace& space, int64_t since_id,
                      ::n4m::core::MethodResult& out) {
    std::vector<double> ids, scores, statuses, rungs, durations;
    std::vector<std::int64_t> ids_i64;
    std::vector<std::int64_t> ask_sequences;
    std::vector<std::int64_t> terminal_sequences;
    std::vector<const n4m_trial_s*> selected;
    std::int64_t n_terminal_events = 0;
    for (const auto& up : trials) {
        if (up->id < since_id) continue;
        selected.push_back(up.get());
        ids.push_back(static_cast<double>(up->id));
        ids_i64.push_back(up->id);
        ask_sequences.push_back(up->ask_sequence);
        terminal_sequences.push_back(up->terminal_sequence);
        if (up->terminal_sequence >= 0) ++n_terminal_events;
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
    out.set_int64_vector("trial_ids_i64", std::move(ids_i64));
    out.set_int64_vector("trial_ask_sequence", std::move(ask_sequences));
    out.set_int64_vector("trial_terminal_sequence", std::move(terminal_sequences));
    out.set_scalar("n_trials", static_cast<double>(n));
    out.set_scalar("trace_format_version",
                   static_cast<double>(N4M_TRIAL_TRACE_FORMAT_VERSION));

    std::vector<std::string> param_names;
    std::vector<std::int32_t> param_kinds;
    std::vector<std::int32_t> param_category_types;
    std::vector<std::int32_t> param_integer;
    for (const opt::ParamSpec& param : space.params) {
        const std::int32_t width =
            param.kind == N4M_PARAM_SORTED_TUPLE ? param.tuple_length : 1;
        for (std::int32_t index = 0; index < width; ++index) {
            param_names.push_back(
                param.kind == N4M_PARAM_SORTED_TUPLE
                    ? param.name + "#" + std::to_string(index)
                    : param.name);
            param_kinds.push_back(static_cast<std::int32_t>(param.kind));
            param_category_types.push_back(
                param.kind == N4M_PARAM_CATEGORICAL
                    ? static_cast<std::int32_t>(param.cat_type)
                    : -1);
            param_integer.push_back(
                (param.is_int ||
                 (param.kind == N4M_PARAM_SORTED_TUPLE && param.tuple_element_is_int))
                    ? 1
                    : 0);
        }
    }
    const std::size_t n_params = param_names.size();
    std::vector<std::int32_t> param_name_bytes;
    std::vector<std::int64_t> param_name_offsets{0};
    for (const std::string& name : param_names) {
        append_utf8(name, param_name_bytes);
        param_name_offsets.push_back(static_cast<std::int64_t>(param_name_bytes.size()));
    }

    std::vector<double> param_values;
    std::vector<std::int32_t> param_category_index;
    std::vector<std::int32_t> param_active;
    std::vector<std::int32_t> param_label_bytes;
    std::vector<std::int64_t> param_label_offsets{0};
    for (const n4m_trial_s* trial : selected) {
        if (trial->params.size() != n_params) {
            throw std::logic_error("optimizer trace parameter width changed");
        }
        for (std::size_t index = 0; index < n_params; ++index) {
            const auto& item = trial->params[index];
            if (item.first != param_names[index]) {
                throw std::logic_error("optimizer trace parameter order changed");
            }
            param_values.push_back(item.second.value);
            param_category_index.push_back(item.second.cat_index);
            param_active.push_back(item.second.active ? 1 : 0);
            append_utf8(item.second.cat_label, param_label_bytes);
            param_label_offsets.push_back(static_cast<std::int64_t>(param_label_bytes.size()));
        }
    }
    out.set_double_matrix("trial_param_values", std::move(param_values), n,
                          static_cast<std::int64_t>(n_params));
    out.set_int_vector("trial_param_category_index", std::move(param_category_index));
    out.set_int_vector("trial_param_active", std::move(param_active));
    out.set_int_vector("trial_param_kind", std::move(param_kinds));
    out.set_int_vector("trial_param_category_type", std::move(param_category_types));
    out.set_int_vector("trial_param_integer", std::move(param_integer));
    out.set_int_vector("trial_param_name_utf8", std::move(param_name_bytes));
    out.set_int64_vector("trial_param_name_offsets", std::move(param_name_offsets));
    out.set_int_vector("trial_param_label_utf8", std::move(param_label_bytes));
    out.set_int64_vector("trial_param_label_offsets", std::move(param_label_offsets));
    out.set_scalar("n_params", static_cast<double>(n_params));

    std::vector<std::int64_t> intermediate_offsets{0};
    std::vector<std::int64_t> intermediate_sequences;
    std::vector<std::int32_t> intermediate_steps;
    std::vector<double> intermediate_scores;
    std::vector<std::int32_t> intermediate_should_prune;
    for (const n4m_trial_s* trial : selected) {
        if (trial->intermediate_sequences.size() != trial->intermediates.size()) {
            throw std::logic_error("optimizer trace intermediate sequence width changed");
        }
        for (std::size_t index = 0; index < trial->intermediates.size(); ++index) {
            const auto& item = trial->intermediates[index];
            intermediate_sequences.push_back(trial->intermediate_sequences[index]);
            intermediate_steps.push_back(item.first);
            intermediate_scores.push_back(item.second);
            const bool is_policy_prune = trial->pruned_by_policy &&
                                         index + 1 == trial->intermediates.size();
            intermediate_should_prune.push_back(is_policy_prune ? 1 : 0);
        }
        intermediate_offsets.push_back(
            static_cast<std::int64_t>(intermediate_steps.size()));
    }
    const std::int64_t n_intermediates =
        static_cast<std::int64_t>(intermediate_scores.size());
    out.set_int64_vector("trial_intermediate_offsets", std::move(intermediate_offsets));
    out.set_int64_vector("trial_intermediate_sequence", std::move(intermediate_sequences));
    out.set_int_vector("trial_intermediate_steps", std::move(intermediate_steps));
    out.set_double_matrix("trial_intermediate_scores", std::move(intermediate_scores), 1,
                          n_intermediates);
    out.set_int_vector("trial_intermediate_should_prune",
                       std::move(intermediate_should_prune));
    out.set_scalar("n_intermediates", static_cast<double>(n_intermediates));
    out.set_scalar("n_events",
                   static_cast<double>(n + n_intermediates + n_terminal_events));

    std::vector<std::int32_t> error_code_bytes;
    std::vector<std::int64_t> error_code_offsets{0};
    std::vector<std::int32_t> error_message_bytes;
    std::vector<std::int64_t> error_message_offsets{0};
    std::vector<std::int32_t> error_retryable;
    for (const n4m_trial_s* trial : selected) {
        if (trial->has_error) {
            append_utf8(trial->error.code, error_code_bytes);
            append_utf8(trial->error.message, error_message_bytes);
        }
        error_code_offsets.push_back(static_cast<std::int64_t>(error_code_bytes.size()));
        error_message_offsets.push_back(static_cast<std::int64_t>(error_message_bytes.size()));
        error_retryable.push_back(trial->has_error && trial->error.retryable ? 1 : 0);
    }
    out.set_int_vector("trial_error_code_utf8", std::move(error_code_bytes));
    out.set_int64_vector("trial_error_code_offsets", std::move(error_code_offsets));
    out.set_int_vector("trial_error_message_utf8", std::move(error_message_bytes));
    out.set_int64_vector("trial_error_message_offsets", std::move(error_message_offsets));
    out.set_int_vector("trial_error_retryable", std::move(error_retryable));
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
    if (!valid_utf8(name)) return N4M_ERR_INVALID_ARGUMENT;
    if (high < low || step < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (log && low <= 0) return N4M_ERR_INVALID_ARGUMENT;  // log needs positive bounds
    if (!is_consecutive_binary64_integer(low) ||
        !is_consecutive_binary64_integer(high) ||
        !is_consecutive_binary64_integer(step)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
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
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_float(n4m_search_space_t* space, const char* name,
                                                double low, double high, double step,
                                                int32_t log) {
    if (space == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    if (!valid_utf8(name)) return N4M_ERR_INVALID_ARGUMENT;
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
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_categorical(n4m_search_space_t* space, const char* name,
                                                      n4m_cat_type_t type, const void* values,
                                                      int32_t n_values) {
    if (space == nullptr || name == nullptr || values == nullptr) return N4M_ERR_NULL_POINTER;
    if (!valid_utf8(name)) return N4M_ERR_INVALID_ARGUMENT;
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
                    if (arr[i] != nullptr && !valid_utf8(arr[i])) {
                        return N4M_ERR_INVALID_ARGUMENT;
                    }
                    p.labels.emplace_back(arr[i] != nullptr ? arr[i] : "");
                    p.num_values.push_back(static_cast<double>(i));
                    break;
                }
                case N4M_CAT_INT: {
                    const int64_t* arr = static_cast<const int64_t*>(values);
                    if (!is_consecutive_binary64_integer(arr[i])) {
                        return N4M_ERR_INVALID_ARGUMENT;
                    }
                    p.num_values.push_back(static_cast<double>(arr[i]));
                    p.labels.push_back(std::to_string(arr[i]));
                    break;
                }
                case N4M_CAT_FLOAT: {
                    const double* arr = static_cast<const double*>(values);
                    p.num_values.push_back(arr[i]);
                    p.labels.push_back(round_trip_double_label(arr[i]));
                    break;
                }
                case N4M_CAT_BOOL: {
                    const int32_t* arr = static_cast<const int32_t*>(values);
                    // Preserve invalid payloads so the single final validator can
                    // reject values other than the ABI-defined 0/1 pair.
                    p.num_values.push_back(static_cast<double>(arr[i]));
                    if (arr[i] == 0) p.labels.emplace_back("false");
                    else if (arr[i] == 1) p.labels.emplace_back("true");
                    else p.labels.emplace_back("");
                    break;
                }
                default:
                    return N4M_ERR_INVALID_ARGUMENT;
            }
        }
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_ordinal(n4m_search_space_t* space, const char* name,
                                                  const double* values, int32_t n_values) {
    if (space == nullptr || name == nullptr || values == nullptr) return N4M_ERR_NULL_POINTER;
    if (!valid_utf8(name)) return N4M_ERR_INVALID_ARGUMENT;
    if (n_values <= 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        opt::ParamSpec p;
        p.name = name;
        p.kind = N4M_PARAM_ORDINAL;
        for (int32_t i = 0; i < n_values; ++i) {
            p.num_values.push_back(values[i]);
            p.labels.push_back(round_trip_double_label(values[i]));
        }
        space->params.push_back(std::move(p));
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_search_space_add_sorted_tuple(n4m_search_space_t* space, const char* name,
                                                       int32_t length, double low, double high,
                                                       int32_t element_is_int) {
    if (space == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    if (!valid_utf8(name)) return N4M_ERR_INVALID_ARGUMENT;
    if (length <= 0 || high < low) return N4M_ERR_INVALID_ARGUMENT;
    if (!std::isfinite(low) || !std::isfinite(high)) return N4M_ERR_INVALID_ARGUMENT;
    if (element_is_int != 0 &&
        (!is_supported_integer_tuple_bound(low) ||
         !is_supported_integer_tuple_bound(high))) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
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
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
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
    for (int32_t i = 0; i < n_refs; ++i) {
        if ((param_refs[i] != nullptr && !valid_utf8(param_refs[i])) ||
            (label_refs != nullptr && label_refs[i] != nullptr &&
             !valid_utf8(label_refs[i]))) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    try {
        opt::Constraint c;
        c.kind = kind;
        for (int32_t i = 0; i < n_refs; ++i) {
            c.param_refs.emplace_back(param_refs[i] != nullptr ? param_refs[i] : "");
            c.label_refs.emplace_back(
                (label_refs != nullptr && label_refs[i] != nullptr) ? label_refs[i] : "");
        }

        if ((kind == N4M_CONSTRAINT_CONDITION_IN ||
             kind == N4M_CONSTRAINT_CONDITION_NOT_IN) &&
            n_refs == 2) {
            // Compile valid-looking {child,parent} records for sampling. Final
            // arity/ref/label/cycle checks remain centralized in SearchSpace::validate.
            opt::ParamSpec* child = space->find(c.param_refs[0]);
            // Conditions bind metadata to their child immediately. Requiring the
            // child now avoids accepting a later-valid graph that sampling ignores;
            // the parent may still be declared after this call.
            if (child == nullptr) return N4M_ERR_INVALID_ARGUMENT;
            const bool is_in = (kind == N4M_CONSTRAINT_CONDITION_IN);
            const std::string& parent = c.param_refs[1];
            const std::string& label = c.label_refs[1];
            if (child->cond_parent.empty() ||
                (child->cond_parent == parent && child->cond_is_in == is_in)) {
                child->cond_parent = parent;
                child->cond_is_in = is_in;
                if (!label.empty()) child->cond_labels.push_back(label);
            }
        }

        space->constraints.push_back(std::move(c));
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
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
        n4m_optimizer_options_t o;
        const n4m_status_t normalize_status = normalize_optimizer_options(ctx, opts, &o);
        if (normalize_status != N4M_OK) return normalize_status;

        n4m_status_t st = N4M_OK;
        std::string optimizer_error;
        auto impl = opt::make_optimizer(*space, o, &st, &optimizer_error);
        if (impl == nullptr) {
            set_error(ctx, optimizer_error.empty()
                               ? "unsupported sampler/pruner or invalid search space"
                               : optimizer_error.c_str());
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
            if (names[i] != nullptr && !valid_utf8(names[i])) {
                return N4M_ERR_INVALID_ARGUMENT;
            }
            params.emplace_back(names[i] != nullptr ? names[i] : "", values[i]);
        }
        return opt->impl->enqueue(std::move(params));
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
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
    // out_count is required; set it to 0 as early as possible (even when a later
    // precedence check fails) so a caller always reads a defined committed count.
    if (out_count == nullptr) return N4M_ERR_NULL_POINTER;
    *out_count = 0;
    if (opt == nullptr) return N4M_ERR_NULL_POINTER;
    if (n < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (n == 0) return N4M_OK;  // nothing to dispense; out_trials may be NULL
    if (out_trials == nullptr) return N4M_ERR_NULL_POINTER;  // n > 0 requires storage
    try {
        for (int32_t i = 0; i < n; ++i) out_trials[i] = nullptr;  // initialise every slot
        return opt->impl->ask_batch(n, out_trials, out_count);
    } catch (const std::bad_alloc&) {
        // A commit that threw mid-batch left [0,*out_count) valid and the rest
        // NULL (ask() is transactional on throw), matching the partial-fatal
        // ownership contract.
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
    if (opt == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        opt::TrialError structured_error;
        bool has_error = false;
        const n4m_status_t decode_status =
            decode_trial_error(status, error, structured_error, has_error);
        if (decode_status != N4M_OK) return decode_status;
        return opt->impl->tell_result(trial_id, status, score,
                                      has_error ? &structured_error : nullptr);
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
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
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
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
    if (since_id < 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        auto handle = std::make_unique<n4m_method_result_s>();
        pack_trial_trace(opt->impl->trials(), opt->impl->search_space(), since_id, *handle);
        *out = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_save(const n4m_optimizer_t* opt, n4m_array_t** out_blob) {
    if (out_blob == nullptr) return N4M_ERR_NULL_POINTER;
    *out_blob = nullptr;
    if (opt == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        std::vector<std::uint8_t> bytes;
        const n4m_status_t status = opt::save_optimizer_checkpoint(*opt->impl, bytes);
        if (status != N4M_OK) return status;
        // n4m_array_t predates byte dtypes. The reserved persistence ABI therefore
        // owns an I64 word array whose backing bytes are the checkpoint verbatim;
        // N4MOPT guarantees an 8-byte-aligned total size. Consumers use
        // n4m_array_view(...).data as bytes and pass rows*cols*8 to load().
        if (bytes.empty() || bytes.size() % sizeof(std::uint64_t) != 0U ||
            bytes.size() / sizeof(std::uint64_t) >
                static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            return N4M_ERR_INTERNAL;
        }
        auto array = std::make_unique<n4m_array_s>();
        array->dtype = N4M_DTYPE_I64;
        array->rows = 1;
        array->cols = static_cast<std::int64_t>(bytes.size() / sizeof(std::uint64_t));
        array->values.resize(static_cast<std::size_t>(array->cols));
        std::memcpy(array->values.data(), bytes.data(), bytes.size());
        *out_blob = array.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_optimizer_load(n4m_context_t* ctx, const uint8_t* blob, uint64_t n,
                                        n4m_optimizer_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (ctx == nullptr || blob == nullptr) {
        set_error(ctx, "null pointer in n4m_optimizer_load");
        return N4M_ERR_NULL_POINTER;
    }
    if (n > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        set_error(ctx, "optimizer checkpoint size exceeds this host");
        return N4M_ERR_CORRUPT_BUFFER;
    }
    try {
        std::unique_ptr<opt::Optimizer> impl;
        std::string checkpoint_error;
        const n4m_status_t status = opt::load_optimizer_checkpoint(
            blob, static_cast<std::size_t>(n), impl, &checkpoint_error);
        if (status != N4M_OK) {
            set_error(ctx, checkpoint_error.empty() ? "invalid optimizer checkpoint"
                                                    : checkpoint_error.c_str());
            return status;
        }
        auto handle = std::make_unique<n4m_optimizer_s>();
        handle->impl = std::move(impl);
        *out = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_optimizer_load");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_optimizer_load");
        return N4M_ERR_INTERNAL;
    }
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
        n4m_optimizer_options_t o;
        const n4m_status_t normalize_status = normalize_optimizer_options(ctx, opts, &o);
        if (normalize_status != N4M_OK) return normalize_status;

        n4m_status_t st = N4M_OK;
        std::string optimizer_error;
        st = opt::validate_optimizer_options(o, &optimizer_error);
        if (st != N4M_OK) {
            set_error(ctx, optimizer_error.c_str());
            return st;
        }
        const FinetuneEntry* entry = find_finetune_entry(estimator);
        if (entry == nullptr) {
            set_error(ctx,
                      "n4m_finetune_estimator supports only the generic native "
                      "regression estimators PLS_REGRESSION, PLS_CANONICAL, PLS_SVD, "
                      "OPLS, SPARSE_PLS and PCR");
            return N4M_ERR_UNSUPPORTED;
        }
        if (o.pruner != N4M_PRUNER_NONE) {
            set_error(ctx,
                      "n4m_finetune_estimator does not yet support pruners because it "
                      "reports no intermediate scores");
            return N4M_ERR_UNSUPPORTED;
        }
        if (!is_regression_metric(o.metric)) {
            set_error(ctx,
                      "metric is not implemented by the regression-only "
                      "n4m_finetune_estimator");
            return N4M_ERR_NOT_IMPLEMENTED;
        }
        const n4m_status_t schema_status = validate_finetune_space(ctx, *entry, *space);
        if (schema_status != N4M_OK) return schema_status;

        ::n4m::core::Context& core_ctx = *as_core(ctx);
        const ::n4m::core::ValidationPlan& vp = *as_core(plan);
        const n4m_status_t input_status =
            ::n4m::core::validate_regression_cv_inputs(core_ctx, *X, *Y, vp);
        if (input_status != N4M_OK) return input_status;

        auto optimizer = opt::make_optimizer(*space, o, &st, &optimizer_error);
        if (optimizer == nullptr) {
            set_error(ctx, optimizer_error.empty()
                               ? "unsupported sampler/pruner or invalid search space in "
                                 "n4m_finetune_estimator"
                               : optimizer_error.c_str());
            return st;
        }

        n4m_status_t first_trial_failure = N4M_OK;
        std::string first_trial_error;
        bool timed_out = false;

        const auto finish_failed_trial = [&](n4m_trial_s* trial,
                                             n4m_status_t cause) -> n4m_status_t {
            const std::string cause_message = core_ctx.last_error();
            const char* code = "CV_ERROR";
            if (cause == N4M_ERR_INVALID_ARGUMENT) code = "INVALID_CANDIDATE";
            else if (cause == N4M_ERR_NUMERICAL_FAILURE) code = "NUMERICAL_FAILURE";
            else if (cause == N4M_ERR_CONVERGENCE_FAILED) code = "CONVERGENCE_FAILED";
            const opt::TrialError trial_error{
                code,
                cause_message.empty() ? "native cross-validation failed" : cause_message,
                false,
            };
            const n4m_status_t tell_status =
                optimizer->tell_result(trial->id, N4M_TRIAL_FAILED, 0.0, &trial_error);
            if (tell_status != N4M_OK) {
                set_error(ctx, "optimizer rejected FAILED terminal status");
                return tell_status;
            }
            if (first_trial_failure == N4M_OK) {
                first_trial_failure = cause;
                first_trial_error = cause_message;
            }
            return N4M_OK;
        };

        for (int32_t i = 0; i < n_trials; ++i) {
            n4m_trial_s* t = nullptr;
            const n4m_status_t ask_st = optimizer->ask(&t);
            if (ask_st == N4M_ERR_CANCELLED) {
                double partial_best_score = 0.0;
                if (optimizer->best(&partial_best_score) != nullptr) {
                    timed_out = true;
                    break;
                }
                set_error(ctx, "finetune timeout elapsed before any trial completed");
                return N4M_ERR_CANCELLED;
            }
            if (ask_st != N4M_OK) return ask_st;

            ::n4m::core::Config cfg;
            const n4m_status_t build_status = build_finetune_config(*entry, *t, cfg);
            if (build_status != N4M_OK) {
                set_error(ctx, build_status == N4M_ERR_INVALID_ARGUMENT
                                   ? "native sampler produced an invalid tuned candidate"
                                   : "internal error building finetune candidate config");
                const n4m_status_t tell_status = finish_failed_trial(t, build_status);
                if (tell_status != N4M_OK) return tell_status;
                if (!is_candidate_specific_cv_failure(build_status)) return build_status;
                continue;
            }

            ::n4m::core::CrossValidationResult cv;
            const n4m_status_t cv_st = ::n4m::core::cross_validate_regression(
                core_ctx, cfg, *X, *Y, vp, cv);
            if (cv_st != N4M_OK) {
                const n4m_status_t tell_status = finish_failed_trial(t, cv_st);
                if (tell_status != N4M_OK) return tell_status;
                if (!is_candidate_specific_cv_failure(cv_st)) return cv_st;
                continue;
            }
            double score = 0.0;
            if (!regression_metric_value(o.metric, cv.metrics, &score)) {
                set_error(ctx, "regression metric dispatch failed after preflight validation");
                const n4m_status_t tell_status =
                    finish_failed_trial(t, N4M_ERR_NOT_IMPLEMENTED);
                return tell_status == N4M_OK ? N4M_ERR_NOT_IMPLEMENTED : tell_status;
            }
            if (!std::isfinite(score)) {
                set_error(ctx, "cross-validation produced a non-finite objective score");
                const n4m_status_t tell_status =
                    finish_failed_trial(t, N4M_ERR_NUMERICAL_FAILURE);
                if (tell_status != N4M_OK) return tell_status;
                continue;
            }
            const n4m_status_t tell_status =
                optimizer->tell_result(t->id, N4M_TRIAL_COMPLETED, score);
            if (tell_status != N4M_OK) {
                set_error(ctx, "optimizer rejected COMPLETED terminal status");
                return tell_status;
            }
        }

        double best_score = 0.0;
        n4m_trial_s* best = optimizer->best(&best_score);
        if (best == nullptr) {
            if (first_trial_failure != N4M_OK) {
                set_error(ctx, first_trial_error.empty()
                                   ? "all finetuning trials failed"
                                   : first_trial_error.c_str());
                return first_trial_failure;
            }
            set_error(ctx, "no trial completed successfully in n4m_finetune_estimator");
            return N4M_ERR_NOT_FITTED;
        }

        auto handle = std::make_unique<n4m_method_result_s>();
        for (const auto& kv : best->params) {
            if (!kv.second.active) continue;
            handle->set_scalar(std::string("best.") + kv.first, kv.second.value);
        }
        handle->set_scalar("best_score", best_score);
        pack_trial_trace(optimizer->trials(), optimizer->search_space(), 0,
                         *handle);  // compatibility columns + trace v1
        handle->set_scalar("metric", static_cast<double>(o.metric));
        handle->set_scalar("estimator", static_cast<double>(entry->algorithm));
        handle->set_scalar("timed_out", timed_out ? 1.0 : 0.0);
        handle->set_scalar("requested_trials", static_cast<double>(n_trials));
        *out_result = handle.release();
        core_ctx.clear_error();
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
