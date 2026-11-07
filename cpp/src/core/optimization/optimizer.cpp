// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Ask/tell optimizer core (F0): SearchSpace + Trial helpers + the base
// Optimizer (random sampler, none pruner) + the sampler factory.

#include "core/optimization/optimizer.hpp"
#include "core/common/double_double.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace n4m::core::opt {

// ---- SearchSpace / Trial lookups ----------------------------------------

ParamSpec* SearchSpace::find(std::string_view name) {
    for (auto& p : params) {
        if (std::string_view(p.name) == name) return &p;
    }
    return nullptr;
}
const ParamSpec* SearchSpace::find(std::string_view name) const {
    for (const auto& p : params) {
        if (std::string_view(p.name) == name) return &p;
    }
    return nullptr;
}

n4m_status_t SearchSpace::validate(n4m_sampler_kind_t sampler, std::string* error) const {
    if (error != nullptr) error->clear();
    auto fail = [error](n4m_status_t status, std::string message) {
        if (error != nullptr) *error = std::move(message);
        return status;
    };
    if (params.empty()) {
        return fail(N4M_ERR_INVALID_ARGUMENT,
                    "search space requires at least one parameter");
    }

    std::unordered_set<std::string> names;
    names.reserve(params.size());
    for (const auto& p : params) {
        if (p.name.empty()) {
            return fail(N4M_ERR_INVALID_ARGUMENT, "search-space parameter name is empty");
        }
        if (p.name.find('#') != std::string::npos) {
            return fail(N4M_ERR_INVALID_ARGUMENT,
                        "search-space parameter name contains reserved '#' separator: " + p.name);
        }
        if (!names.insert(p.name).second) {
            return fail(N4M_ERR_INVALID_ARGUMENT,
                        "duplicate search-space parameter name: " + p.name);
        }

        const auto finite_domain = [&p]() {
            return std::isfinite(p.low) && std::isfinite(p.high) && std::isfinite(p.step) &&
                   p.low <= p.high;
        };
        constexpr double max_consecutive_binary64_integer = 9007199254740992.0;  // 2^53
        const auto exact_integer = [](double value) {
            return std::isfinite(value) && std::trunc(value) == value &&
                   std::abs(value) <= max_consecutive_binary64_integer;
        };
        switch (p.kind) {
            case N4M_PARAM_INT:
                if (!finite_domain() || !exact_integer(p.low) || !exact_integer(p.high) ||
                    !exact_integer(p.step) || !p.is_int || p.is_log || p.step <= 0.0) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed integer search-space domain: " + p.name);
                }
                break;
            case N4M_PARAM_LOG_INT:
                if (!finite_domain() || !exact_integer(p.low) || !exact_integer(p.high) ||
                    !exact_integer(p.step) || !p.is_int || !p.is_log || p.low <= 0.0 ||
                    p.step != 1.0) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed log-integer search-space domain: " + p.name);
                }
                break;
            case N4M_PARAM_FLOAT:
                if (!finite_domain() || p.is_int || p.is_log || p.step < 0.0) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed floating search-space domain: " + p.name);
                }
                if (p.step > 0.0 &&
                    (!std::isfinite((p.high - p.low) / p.step) ||
                     (p.high > p.low && p.low + p.step == p.low))) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "floating search-space grid is not representable: " + p.name);
                }
                if (!std::isfinite(p.high - p.low)) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "floating search-space span overflows binary64: " + p.name);
                }
                break;
            case N4M_PARAM_LOG_FLOAT:
                if (!finite_domain() || p.is_int || !p.is_log || p.low <= 0.0 ||
                    p.step != 0.0) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed log-floating search-space domain: " + p.name);
                }
                break;
            case N4M_PARAM_CATEGORICAL: {
                if (p.cat_type < N4M_CAT_STR || p.cat_type > N4M_CAT_BOOL || p.labels.empty() ||
                    p.labels.size() != p.num_values.size()) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed categorical search-space domain: " + p.name);
                }
                std::unordered_set<std::string> labels;
                labels.reserve(p.labels.size());
                for (std::size_t i = 0; i < p.labels.size(); ++i) {
                    if (p.labels[i].empty() || !labels.insert(p.labels[i]).second ||
                        !std::isfinite(p.num_values[i])) {
                        return fail(N4M_ERR_INVALID_ARGUMENT,
                                    "empty, duplicate, or non-finite categorical choice: " +
                                        p.name);
                    }
                    if (p.cat_type == N4M_CAT_BOOL && p.num_values[i] != 0.0 &&
                        p.num_values[i] != 1.0) {
                        return fail(N4M_ERR_INVALID_ARGUMENT,
                                    "boolean categorical choices must be exactly 0 or 1: " +
                                        p.name);
                    }
                    if (p.cat_type == N4M_CAT_INT && !exact_integer(p.num_values[i])) {
                        return fail(
                            N4M_ERR_INVALID_ARGUMENT,
                            "integer categorical choice exceeds exact binary64 range: " +
                                p.name);
                    }
                    for (std::size_t j = 0; j < i; ++j) {
                        if (p.num_values[j] == p.num_values[i]) {
                            return fail(N4M_ERR_INVALID_ARGUMENT,
                                        "duplicate categorical choice: " + p.name);
                        }
                    }
                }
                break;
            }
            case N4M_PARAM_ORDINAL:
                if (p.num_values.empty() || p.labels.size() != p.num_values.size()) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed ordinal search-space domain: " + p.name);
                }
                for (std::size_t i = 0; i < p.num_values.size(); ++i) {
                    if (!std::isfinite(p.num_values[i]) || p.labels[i].empty()) {
                        return fail(N4M_ERR_INVALID_ARGUMENT,
                                    "empty or non-finite ordinal choice: " + p.name);
                    }
                    for (std::size_t j = 0; j < i; ++j) {
                        if (p.num_values[j] == p.num_values[i] || p.labels[j] == p.labels[i]) {
                            return fail(N4M_ERR_INVALID_ARGUMENT,
                                        "duplicate ordinal choice: " + p.name);
                        }
                    }
                }
                break;
            case N4M_PARAM_SORTED_TUPLE:
                if (!finite_domain() || p.tuple_length <= 0 ||
                    (p.tuple_element_is_int &&
                     (!exact_integer(p.low) || !exact_integer(p.high)))) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "malformed sorted-tuple search-space domain: " + p.name);
                }
                if (!std::isfinite(p.high - p.low)) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "sorted-tuple search-space span overflows binary64: " + p.name);
                }
                break;
            default:
                return fail(N4M_ERR_INVALID_ARGUMENT,
                            "unknown search-space parameter kind: " + p.name);
        }
    }

    struct Activation {
        std::string                     parent;
        n4m_constraint_kind_t           kind{N4M_CONSTRAINT_CONDITION_IN};
        std::unordered_set<std::string> labels;
    };
    std::unordered_map<std::string, Activation> activations;
    bool has_hard_constraints = false;

    for (const auto& c : constraints) {
        if (c.param_refs.size() != c.label_refs.size()) {
            return fail(N4M_ERR_INVALID_ARGUMENT,
                        "constraint parameter/label arrays have different lengths");
        }
        std::size_t required_refs = 0;
        bool        hard_constraint = false;
        switch (c.kind) {
            case N4M_CONSTRAINT_MUTEX_GROUP:
                if (c.param_refs.size() < 2) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "mutex constraint requires at least two references");
                }
                has_hard_constraints = true;
                hard_constraint = true;
                break;
            case N4M_CONSTRAINT_REQUIRES:
            case N4M_CONSTRAINT_EXCLUDE:
                required_refs = 2;
                has_hard_constraints = true;
                hard_constraint = true;
                break;
            case N4M_CONSTRAINT_CONDITION_IN:
            case N4M_CONSTRAINT_CONDITION_NOT_IN:
                required_refs = 2;
                break;
            default:
                return fail(N4M_ERR_INVALID_ARGUMENT, "unknown search-space constraint kind");
        }
        if (required_refs != 0 && c.param_refs.size() != required_refs) {
            return fail(N4M_ERR_INVALID_ARGUMENT,
                        "constraint requires exactly two parameter references");
        }

        std::vector<std::pair<std::string, std::string>> constraint_refs;
        for (std::size_t i = 0; i < c.param_refs.size(); ++i) {
            const std::string& ref = c.param_refs[i];
            const std::string& label = c.label_refs[i];
            const auto ref_identity = std::make_pair(ref, label);
            if (ref.empty() ||
                std::find(constraint_refs.begin(), constraint_refs.end(), ref_identity) !=
                    constraint_refs.end()) {
                return fail(N4M_ERR_INVALID_ARGUMENT,
                            "constraint contains an empty or duplicate (parameter, label) "
                            "reference");
            }
            constraint_refs.push_back(ref_identity);
            const ParamSpec* p = find(ref);
            if (p == nullptr) {
                return fail(N4M_ERR_INVALID_ARGUMENT,
                            "constraint references unknown parameter: " + ref);
            }
            if (!label.empty()) {
                if (p->kind != N4M_PARAM_CATEGORICAL && p->kind != N4M_PARAM_ORDINAL) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                "constraint label targets a non-categorical parameter: " + ref);
                }
                if (std::find(p->labels.begin(), p->labels.end(), label) == p->labels.end()) {
                    return fail(N4M_ERR_INVALID_ARGUMENT,
                                    "constraint references unknown choice label for parameter: " + ref);
                }
            }
            if (hard_constraint && p->kind == N4M_PARAM_SORTED_TUPLE) {
                return fail(N4M_ERR_UNSUPPORTED,
                            "hard constraints cannot reference sorted_tuple parameter: " + ref);
            }
        }

        if (c.kind == N4M_CONSTRAINT_CONDITION_IN ||
            c.kind == N4M_CONSTRAINT_CONDITION_NOT_IN) {
            if (!c.label_refs[0].empty() || c.label_refs[1].empty()) {
                return fail(N4M_ERR_INVALID_ARGUMENT,
                            "condition requires an unlabeled child and a labeled parent");
            }
            const std::string& child = c.param_refs[0];
            const std::string& parent = c.param_refs[1];
            auto [it, inserted] = activations.emplace(
                child, Activation{parent, c.kind, std::unordered_set<std::string>{}});
            if (!inserted && (it->second.parent != parent || it->second.kind != c.kind)) {
                return fail(N4M_ERR_UNSUPPORTED,
                            "multiple condition parents or predicates for parameter: " + child);
            }
            if (!it->second.labels.insert(c.label_refs[1]).second) {
                return fail(N4M_ERR_INVALID_ARGUMENT,
                            "duplicate condition choice label for parameter: " + child);
            }
        }
    }

    std::unordered_map<std::string, int> state;
    state.reserve(activations.size());
    std::function<bool(const std::string&)> has_cycle = [&](const std::string& child) {
        const auto state_it = state.find(child);
        if (state_it != state.end()) return state_it->second == 1;
        state[child] = 1;
        const auto activation_it = activations.find(child);
        if (activation_it != activations.end() &&
            has_cycle(activation_it->second.parent)) {
            return true;
        }
        state[child] = 2;
        return false;
    };
    for (const auto& kv : activations) {
        if (has_cycle(kv.first)) {
            return fail(N4M_ERR_INVALID_ARGUMENT, "conditional activation graph contains a cycle");
        }
    }

    if (has_hard_constraints &&
        (sampler == N4M_SAMPLER_SOBOL || sampler == N4M_SAMPLER_GA ||
         sampler == N4M_SAMPLER_PSO ||
         sampler == N4M_SAMPLER_CMAES || sampler == N4M_SAMPLER_GP_EI)) {
        const char* sampler_name = sampler == N4M_SAMPLER_SOBOL ? "sobol"
                                   : sampler == N4M_SAMPLER_GA  ? "ga"
                                   : sampler == N4M_SAMPLER_PSO ? "pso"
                                   : sampler == N4M_SAMPLER_CMAES ? "cmaes"
                                                                  : "gp_ei";
        return fail(N4M_ERR_UNSUPPORTED,
                    std::string("hard constraints are unsupported by sampler ") + sampler_name);
    }
    return N4M_OK;
}

n4m_status_t validate_optimizer_options(const n4m_optimizer_options_t& opts,
                                        std::string* error) {
    if (error != nullptr) error->clear();
    auto fail = [error](n4m_status_t status, const char* message) {
        if (error != nullptr) *error = message;
        return status;
    };
    auto enum_bits = [](const auto& value) {
        std::int32_t raw = 0;
        static_assert(sizeof(value) == sizeof(raw));
        std::memcpy(&raw, &value, sizeof(raw));
        return raw;
    };

    if (opts.struct_size < sizeof(n4m_optimizer_options_t)) {
        return fail(N4M_ERR_INVALID_ARGUMENT,
                    "n4m_optimizer_options_t.struct_size is smaller than the ABI 2.1 layout");
    }
    const std::int32_t sampler = enum_bits(opts.sampler);
    const std::int32_t pruner = enum_bits(opts.pruner);
    const std::int32_t direction = enum_bits(opts.direction);
    const std::int32_t eval_mode = enum_bits(opts.eval_mode);
    const std::int32_t metric = enum_bits(opts.metric);
    const std::int32_t liar = enum_bits(opts.liar);
    if (sampler < N4M_SAMPLER_RANDOM || sampler > N4M_SAMPLER_GP_EI) {
        return fail(N4M_ERR_NOT_IMPLEMENTED, "unknown or unimplemented sampler kind");
    }
    if (pruner < N4M_PRUNER_NONE || pruner > N4M_PRUNER_RACING) {
        return fail(N4M_ERR_NOT_IMPLEMENTED, "unknown or unimplemented pruner kind");
    }
    if (direction < N4M_OPT_AUTO || direction > N4M_OPT_MAXIMIZE) {
        return fail(N4M_ERR_INVALID_ARGUMENT, "optimizer option direction is invalid");
    }
    if (eval_mode < N4M_EVAL_BEST || eval_mode > N4M_EVAL_ROBUST_BEST) {
        return fail(N4M_ERR_INVALID_ARGUMENT, "optimizer option eval_mode is invalid");
    }
    if (eval_mode != N4M_EVAL_MEAN) {
        return fail(N4M_ERR_NOT_IMPLEMENTED,
                    "optimizer option eval_mode is not implemented unless it is mean");
    }
    switch (metric) {
        case N4M_METRIC_RMSE:
        case N4M_METRIC_MSE:
        case N4M_METRIC_MAE:
        case N4M_METRIC_R2:
        case N4M_METRIC_ACCURACY:
        case N4M_METRIC_BALANCED_ACCURACY:
        case N4M_METRIC_F1:
        case N4M_METRIC_LOGLOSS:
            break;
        default:
            return fail(N4M_ERR_INVALID_ARGUMENT, "optimizer option metric is invalid");
    }
    if (liar < N4M_LIAR_NONE || liar > N4M_LIAR_MAX) {
        return fail(N4M_ERR_INVALID_ARGUMENT, "optimizer option liar is invalid");
    }
    if (liar != N4M_LIAR_NONE) {
        return fail(N4M_ERR_NOT_IMPLEMENTED,
                    "optimizer option liar is not implemented unless it is none");
    }
    if (opts.n_startup_trials < 0) {
        return fail(N4M_ERR_INVALID_ARGUMENT, "n_startup_trials must be >= 0");
    }
    if (opts.n_startup_trials == 0 &&
        (sampler == N4M_SAMPLER_TPE || sampler == N4M_SAMPLER_GP_EI ||
         pruner == N4M_PRUNER_MEDIAN)) {
        return fail(N4M_ERR_INVALID_ARGUMENT,
                    "n_startup_trials must be > 0 for tpe, gp_ei, and median");
    }
    if (!std::isfinite(opts.timeout_seconds) || opts.timeout_seconds < 0.0) {
        return fail(N4M_ERR_INVALID_ARGUMENT, "timeout_seconds must be finite and >= 0");
    }
    if (opts.max_resource < 0) {
        return fail(N4M_ERR_INVALID_ARGUMENT, "max_resource must be >= 0");
    }
    if (pruner == N4M_PRUNER_HYPERBAND) {
        if (opts.max_resource <= 0) {
            return fail(N4M_ERR_INVALID_ARGUMENT,
                        "hyperband requires max_resource > 0");
        }
    } else if (opts.max_resource != 0) {
        return fail(N4M_ERR_INVALID_ARGUMENT,
                    "max_resource must be 0 unless pruner is hyperband");
    }
    if (opts.reduction_factor < 0 || opts.reduction_factor == 1) {
        return fail(N4M_ERR_INVALID_ARGUMENT,
                    "reduction_factor must be 0 or >= 2");
    }
    if (pruner != N4M_PRUNER_ASHA && pruner != N4M_PRUNER_HYPERBAND &&
        opts.reduction_factor != 0) {
        return fail(N4M_ERR_INVALID_ARGUMENT,
                    "reduction_factor must be 0 unless pruner is asha or hyperband");
    }
    for (const std::uint8_t byte : opts.reserved) {
        if (byte != 0) {
            return fail(N4M_ERR_INVALID_ARGUMENT,
                        "n4m_optimizer_options_t.reserved must be zero");
        }
    }
    return N4M_OK;
}

const TrialParam* Trial::find(std::string_view name) const {
    for (const auto& kv : params) {
        if (std::string_view(kv.first) == name) return &kv.second;
    }
    return nullptr;
}

// ---- direction ----------------------------------------------------------

n4m_opt_direction_t direction_for_metric(n4m_metric_t metric) {
    switch (metric) {
        case N4M_METRIC_R2:
        case N4M_METRIC_ACCURACY:
        case N4M_METRIC_BALANCED_ACCURACY:
        case N4M_METRIC_F1:
            return N4M_OPT_MAXIMIZE;
        default:  // RMSE / MSE / MAE / LOGLOSS
            return N4M_OPT_MINIMIZE;
    }
}

// ---- Optimizer ----------------------------------------------------------

Optimizer::Optimizer(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : space_(space), opts_(opts) {
    dir_ = (opts.direction == N4M_OPT_AUTO) ? direction_for_metric(opts.metric)
                                            : opts.direction;
    n4m_rng_seed(&rng_, N4M_RNGK_SPLITMIX64, opts.seed);
    start_time_ = std::chrono::steady_clock::now();
    pruner_ = make_pruner(opts, nullptr);  // make_optimizer has already validated the kind
}

bool Optimizer::better(double candidate, double incumbent) const {
    return dir_ == N4M_OPT_MAXIMIZE ? candidate > incumbent : candidate < incumbent;
}

double Optimizer::sample_numeric(const ParamSpec& p) {
    return numeric_from_unit(p, n4m_rng_next_double(&rng_));
}

namespace {

constexpr double kMaxExactInteger = 9007199254740992.0;  // 2^53

bool exact_integer_grid(const ParamSpec& p) {
    const auto exact = [](double v) {
        return std::trunc(v) == v && std::abs(v) <= kMaxExactInteger;
    };
    return p.step > 0.0 && !p.is_log && exact(p.low) && exact(p.high) && exact(p.step);
}

std::int64_t integer_grid_max_index(const ParamSpec& p) {
    return (static_cast<std::int64_t>(p.high) - static_cast<std::int64_t>(p.low)) /
           static_cast<std::int64_t>(p.step);
}

// Exact floor(u * count) for finite u in [0,1) and count <= 2^54 + 1.
std::uint64_t exact_unit_index(double u, std::uint64_t count) {
    int exponent = 0;
    const double fraction = std::frexp(u, &exponent);
    const auto significand = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
    const int shift = 53 - exponent;
    const std::uint64_t a0 = significand & 0xffffffffU, a1 = significand >> 32;
    const std::uint64_t b0 = count & 0xffffffffU, b1 = count >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffU) + (p10 & 0xffffffffU);
    const std::uint64_t low = (mid << 32) | (p00 & 0xffffffffU);
    const std::uint64_t high = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    if (shift >= 128) return 0;
    if (shift >= 64) return high >> (shift - 64);
    return (high << (64 - shift)) | (low >> shift);
}

double integer_grid_value(const ParamSpec& p, double u) {
    if (std::isnan(u)) return u;  // strict trial validation still refuses it
    const std::int64_t q = integer_grid_max_index(p);
    const std::int64_t index =
        q < INT64_C(9007199254740992)
            ? static_cast<std::int64_t>(std::floor(u * (static_cast<double>(q) + 1.0)))
            : static_cast<std::int64_t>(exact_unit_index(u, static_cast<std::uint64_t>(q) + 1U));
    return static_cast<double>(static_cast<std::int64_t>(p.low) +
                               index * static_cast<std::int64_t>(p.step));
}

// high-low is split exactly, so a rounded next cannot pass as an aligned bound.
double grid_excess(const ParamSpec& p, double index) {
    const DD span = dd_two_sum(p.high, -p.low);
    return std::fma(index, p.step, -span.hi) - span.lo;
}

double ulp_size(double value) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    const double toward_positive = std::nextafter(value, std::numeric_limits<double>::infinity());
    const double toward_negative = std::nextafter(value, -std::numeric_limits<double>::infinity());
    return std::max(std::abs(toward_positive - value),
                    std::abs(value - toward_negative));
}

// Below this spacing, two roundings of low + index*step can exceed the tolerance.
bool near_ulp_grid(const ParamSpec& p) {
    return p.step < 8.0 * std::max(ulp_size(p.low), ulp_size(p.high));
}

double grid_index_below(double index) {
    return index > kMaxExactInteger
               ? std::nextafter(index, -std::numeric_limits<double>::infinity())
               : index - 1.0;
}

double grid_value_tolerance(const ParamSpec& p, double value, double snapped) {
    const double ulp = std::max({ulp_size(value), ulp_size(snapped), ulp_size(p.low),
                                 ulp_size(p.high), ulp_size(p.step)});
    // Never allow numerical tolerance to bridge a material fraction of a grid
    // cell. In particular, do not use an absolute scale floor of 1.0: that
    // accepts arbitrary off-grid values in sub-unit and subnormal domains.
    return std::min(std::abs(p.step) * 0.25, ulp * 16.0);
}

double grid_max_index(const ParamSpec& p) {
    if (exact_integer_grid(p)) return static_cast<double>(integer_grid_max_index(p));
    const double ratio = (p.high - p.low) / p.step;
    double index = std::floor(std::nextafter(ratio, std::numeric_limits<double>::infinity()));
    if (!std::isfinite(index) || index < 0.0) return 0.0;

    // Correct the at-most-one-cell ambiguity introduced by binary64 division
    // at an aligned high bound. The same ULP-capped rule is used by enqueue().
    while (index > 0.0) {
        const double snapped = p.low + index * p.step;
        if (snapped <= p.high + grid_value_tolerance(p, p.high, snapped) &&
            grid_excess(p, index) <= p.step * 0.25) {
            break;
        }
        index = grid_index_below(index);
    }
    const double next = p.low + (index + 1.0) * p.step;
    if (std::isfinite(next) &&
        next <= p.high + grid_value_tolerance(p, p.high, next) &&
        grid_excess(p, index + 1.0) <= p.step * 0.25) {
        index += 1.0;
    }
    return index;
}

double grid_value(const ParamSpec& p, double index) {
    const double value = near_ulp_grid(p) ? std::fma(index, p.step, p.low)
                                          : p.low + index * p.step;
    if (value > p.high &&
        value <= p.high + grid_value_tolerance(p, p.high, value)) {
        return p.high;
    }
    return value;
}

bool snap_to_grid(const ParamSpec& p, double value, double* snapped_out) {
    const double max_index = grid_max_index(p);
    const double approx = std::round((value - p.low) / p.step);
    const DD offset = dd_two_sum(value, -p.low);
    const double residual = std::fma(approx, p.step, -offset.hi) - offset.lo;
    const double exact = approx - std::round(residual / p.step);
    for (const double index : {exact, approx, std::min(exact, max_index)}) {
        if (index < 0.0 || index > max_index) continue;
        const double snapped = grid_value(p, index);
        const double tolerance = grid_value_tolerance(p, value, snapped);
        if (std::abs(value - snapped) <= tolerance && snapped <= p.high + tolerance) {
            *snapped_out = snapped;
            return true;
        }
    }
    return false;
}

// value must already satisfy the axis range and integer checks.
bool snap_scalar_grid_value(const ParamSpec& p, double value, double* snapped) {
    if (p.step <= 0.0 || p.is_log) {
        *snapped = value;
        return true;
    }
    if (exact_integer_grid(p) && std::trunc(value) == value) {
        const auto integer = static_cast<std::int64_t>(value);
        if ((integer - static_cast<std::int64_t>(p.low)) %
                static_cast<std::int64_t>(p.step) != 0) {
            return false;
        }
        *snapped = static_cast<double>(integer);
        return true;
    }
    return !p.is_int && snap_to_grid(p, value, snapped);
}

double sorted_tuple_element(const ParamSpec& p, double u) {
    if (p.tuple_element_is_int) return p.low + std::floor(u * (p.high - p.low + 1.0));
    return p.low + u * (p.high - p.low);
}

}  // namespace

bool scalar_grid_contains(const ParamSpec& p, double value) {
    double snapped = 0.0;
    return snap_scalar_grid_value(p, value, &snapped) && snapped == value;
}

double Optimizer::numeric_from_unit(const ParamSpec& p, double u) const {
    if (u < 0.0) u = 0.0;
    if (u >= 1.0) u = std::nextafter(1.0, 0.0);
    if (exact_integer_grid(p)) return integer_grid_value(p, u);
    if (p.step > 0.0 && !p.is_log) {
        const double max_index = grid_max_index(p);
        double index = std::floor(u * (max_index + 1.0));
        if (index > max_index) index = max_index;
        return grid_value(p, index);
    }
    double v;
    if (p.is_log && p.low > 0.0 && p.high > 0.0) {
        const double lo = std::log(p.low);
        const double hi = std::log(p.high);
        v = std::exp(lo + u * (hi - lo));
        if (p.is_int) v = std::round(v);
        if (v < p.low) {
            v = p.low;
        } else if (v > p.high) {
            v = p.high;
        }
    } else if (p.is_int) {
        v = std::floor(p.low + u * (p.high - p.low + 1.0));
        if (v > p.high) v = p.high;
    } else {
        v = p.low + u * (p.high - p.low);
    }
    return v;
}

double Optimizer::unit_from_numeric(const ParamSpec& p, double value) const {
    if (p.high <= p.low) return 0.0;
    double log_width = 0.0;
    if (p.is_log && p.low > 0.0 && p.high > 0.0 && value > 0.0) {
        log_width = std::log(p.high) - std::log(p.low);
    }
    double u;
    if (log_width > 0.0) {
        u = (std::log(value) - std::log(p.low)) / log_width;
    } else {
        u = (value - p.low) / (p.high - p.low);
    }
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    return u;
}

void Optimizer::set_trial_value(::n4m_trial_s& t, const ParamSpec& p, double forced) const {
    TrialParam tp;
    if (p.kind == N4M_PARAM_CATEGORICAL || p.kind == N4M_PARAM_ORDINAL) {
        int idx = static_cast<int>(std::llround(forced));
        const int n = static_cast<int>(std::max(p.labels.size(), p.num_values.size()));
        if (idx < 0) idx = 0;
        if (n > 0 && idx >= n) idx = n - 1;
        tp.cat_index = idx;
        tp.cat_label = (idx < static_cast<int>(p.labels.size()))
                           ? p.labels[static_cast<std::size_t>(idx)]
                           : "";
        tp.value = (idx < static_cast<int>(p.num_values.size()))
                       ? p.num_values[static_cast<std::size_t>(idx)]
                       : static_cast<double>(idx);
    } else {
        tp.value = forced;
    }
    t.params.emplace_back(p.name, tp);
}

void Optimizer::decode_candidate(const std::vector<double>& u, ::n4m_trial_s& t,
                                 const std::vector<std::pair<std::string, double>>* forced) {
    auto forced_of = [&](const std::string& name, double* out) -> bool {
        if (forced == nullptr) return false;
        for (const auto& f : *forced) {
            if (f.first == name) { *out = f.second; return true; }
        }
        return false;
    };
    t.params.clear();
    for (std::size_t j = 0; j < space_.params.size(); ++j) {
        const ParamSpec& p = space_.params[j];
        double fv = 0.0;
        if (forced_of(p.name, &fv)) {
            set_trial_value(t, p, fv);
            continue;
        }
        if (p.kind == N4M_PARAM_SORTED_TUPLE) {  // not encoded in the candidate vector
            std::vector<double> vals;
            vals.reserve(static_cast<std::size_t>(p.tuple_length));
            for (std::int32_t i = 0; i < p.tuple_length; ++i) {
                const double uu = n4m_rng_next_double(&rng_);
                vals.push_back(sorted_tuple_element(p, uu));
            }
            std::sort(vals.begin(), vals.end());
            for (std::int32_t i = 0; i < p.tuple_length; ++i) {
                TrialParam sub;
                sub.value = vals[static_cast<std::size_t>(i)];
                t.params.emplace_back(p.name + "#" + std::to_string(i), sub);
            }
            continue;
        }
        double uj = j < u.size() ? u[j] : 0.5;
        if (uj < 0.0) uj = 0.0;
        else if (uj >= 1.0) uj = std::nextafter(1.0, 0.0);
        TrialParam tp;
        if (p.kind == N4M_PARAM_CATEGORICAL || p.kind == N4M_PARAM_ORDINAL) {
            const int n = static_cast<int>(std::max(p.labels.size(), p.num_values.size()));
            int idx = n > 0 ? static_cast<int>(uj * n) : 0;
            if (idx >= n) idx = n - 1;
            if (idx < 0) idx = 0;
            tp.cat_index = idx;
            tp.cat_label = (idx < static_cast<int>(p.labels.size()))
                               ? p.labels[static_cast<std::size_t>(idx)]
                               : "";
            tp.value = (idx < static_cast<int>(p.num_values.size()))
                           ? p.num_values[static_cast<std::size_t>(idx)]
                           : static_cast<double>(idx);
        } else {
            tp.value = numeric_from_unit(p, uj);
        }
        t.params.emplace_back(p.name, tp);
    }
    apply_conditions(t);
}

bool Optimizer::sample(::n4m_trial_s& t,
                       const std::vector<std::pair<std::string, double>>* forced) {
    auto forced_of = [&](const std::string& name, double* out) -> bool {
        if (forced == nullptr) return false;
        for (const auto& f : *forced) {
            if (f.first == name) { *out = f.second; return true; }
        }
        return false;
    };
    constexpr int kMaxTries = 200;
    for (int attempt = 0; attempt < kMaxTries; ++attempt) {
        t.params.clear();
        for (const auto& p : space_.params) {
            if (p.kind == N4M_PARAM_SORTED_TUPLE) {  // not forceable in F0
                std::vector<double> vals;
                vals.reserve(static_cast<std::size_t>(p.tuple_length));
                for (std::int32_t i = 0; i < p.tuple_length; ++i) {
                    const double u = n4m_rng_next_double(&rng_);
                    vals.push_back(sorted_tuple_element(p, u));
                }
                std::sort(vals.begin(), vals.end());
                for (std::int32_t i = 0; i < p.tuple_length; ++i) {
                    TrialParam sub;
                    sub.value = vals[static_cast<std::size_t>(i)];
                    t.params.emplace_back(p.name + "#" + std::to_string(i), sub);
                }
                continue;
            }

            double fv = 0.0;
            if (forced_of(p.name, &fv)) {
                set_trial_value(t, p, fv);
                continue;
            }

            TrialParam tp;
            switch (p.kind) {
                case N4M_PARAM_CATEGORICAL: {
                    const int n = static_cast<int>(p.labels.size());
                    std::int32_t ov = 0;
                    int idx = override_categorical(p, &ov)
                                  ? static_cast<int>(ov)
                                  : (n > 0 ? static_cast<int>(n4m_rng_next_double(&rng_) * n) : 0);
                    if (idx >= n) idx = n - 1;
                    if (idx < 0) idx = 0;
                    tp.cat_index = idx;
                    tp.cat_label = (idx < n) ? p.labels[static_cast<std::size_t>(idx)] : "";
                    tp.value = (idx < static_cast<int>(p.num_values.size()))
                                   ? p.num_values[static_cast<std::size_t>(idx)]
                                   : static_cast<double>(idx);
                    break;
                }
                case N4M_PARAM_ORDINAL: {
                    const int n = static_cast<int>(p.num_values.size());
                    std::int32_t ov = 0;
                    int idx = override_categorical(p, &ov)
                                  ? static_cast<int>(ov)
                                  : (n > 0 ? static_cast<int>(n4m_rng_next_double(&rng_) * n) : 0);
                    if (idx >= n) idx = n - 1;
                    if (idx < 0) idx = 0;
                    tp.cat_index = idx;
                    tp.value = (n > 0) ? p.num_values[static_cast<std::size_t>(idx)] : 0.0;
                    tp.cat_label = (idx < static_cast<int>(p.labels.size()))
                                       ? p.labels[static_cast<std::size_t>(idx)]
                                       : "";
                    break;
                }
                default: {  // INT / FLOAT / LOG_INT / LOG_FLOAT
                    double ov = 0.0;
                    tp.value = override_numeric(p, &ov) ? ov : sample_numeric(p);
                    break;
                }
            }
            t.params.emplace_back(p.name, tp);
        }
        apply_conditions(t);
        if (constraints_ok(t)) return true;
    }
    return false;  // constraints unsatisfiable within the retry budget
}

void Optimizer::apply_conditions(::n4m_trial_s& t) const {
    // A parameter is active iff its own condition holds AND its conditional parent
    // is itself active — resolved recursively up the conditional forest (E2). A
    // single label-only pass wrongly reactivates a grandchild whose branch is dead
    // (the parent's stale label still matches) — that breaks nested sub-pipeline
    // search, where an operator's attributes must vanish when an ancestor slot
    // does not select that operator. Memoised; SearchSpace::validate guarantees
    // that the forest is acyclic and each child has at most one parent/predicate.
    std::unordered_map<std::string, int> memo;  // name -> 0 inactive / 1 active
    std::function<bool(const ParamSpec&)> resolve = [&](const ParamSpec& p) -> bool {
        auto it = memo.find(p.name);
        if (it != memo.end()) return it->second == 1;
        memo[p.name] = 0;  // cycle guard (defensive; the forest is acyclic)
        bool active = true;
        if (!p.cond_parent.empty()) {
            const TrialParam* parent_val = t.find(p.cond_parent);
            bool label_in = false;
            if (parent_val != nullptr) {
                for (const auto& l : p.cond_labels) {
                    if (l == parent_val->cat_label) { label_in = true; break; }
                }
            }
            const bool cond_ok = p.cond_is_in ? label_in : !label_in;
            const ParamSpec* parent_spec = space_.find(p.cond_parent);
            // A missing parent cannot be active, so its child cannot be either.
            const bool parent_active = parent_spec != nullptr && resolve(*parent_spec);
            active = cond_ok && parent_active;
        }
        memo[p.name] = active ? 1 : 0;
        return active;
    };
    // Params start active; conditions only ever DEACTIVATE (never re-activate), so
    // the '#' cascade of a structural ancestor can no longer overwrite a
    // deactivated child back to active regardless of declaration order. Skipping
    // unconditional/active params also keeps flat spaces allocation-cheap.
    for (const auto& p : space_.params) {
        if (p.cond_parent.empty()) continue;  // unconditional → active; no-op
        if (resolve(p)) continue;             // active → nothing to deactivate
        const std::string prefix = p.name + "#";  // deactivate the param + its structural sub-params
        for (auto& kv : t.params) {
            if (kv.first == p.name || kv.first.rfind(prefix, 0) == 0) {
                kv.second.active = false;
            }
        }
    }
}

bool Optimizer::ref_present(const ::n4m_trial_s& t, const std::string& param,
                            const std::string& label) const {
    const TrialParam* tp = t.find(param);
    if (tp == nullptr || !tp->active) return false;
    if (label.empty()) return true;         // bare presence
    return tp->cat_label == label;          // categorical label match
}

bool Optimizer::constraints_ok(const ::n4m_trial_s& t) const {
    auto label_at = [](const Constraint& c, std::size_t i) -> std::string {
        return i < c.label_refs.size() ? c.label_refs[i] : std::string();
    };
    for (const auto& c : space_.constraints) {
        switch (c.kind) {
            case N4M_CONSTRAINT_MUTEX_GROUP: {
                // Only the all-present combination is forbidden (nirs4all `_mutex_`
                // issubset rule); proper subsets are allowed.
                if (c.param_refs.empty()) break;
                bool all = true;
                for (std::size_t i = 0; i < c.param_refs.size(); ++i) {
                    if (!ref_present(t, c.param_refs[i], label_at(c, i))) { all = false; break; }
                }
                if (all) return false;
                break;
            }
            case N4M_CONSTRAINT_REQUIRES: {
                if (c.param_refs.size() < 2) break;
                const bool a = ref_present(t, c.param_refs[0], label_at(c, 0));
                const bool b = ref_present(t, c.param_refs[1], label_at(c, 1));
                if (a && !b) return false;
                break;
            }
            case N4M_CONSTRAINT_EXCLUDE: {
                if (c.param_refs.size() < 2) break;
                const bool a = ref_present(t, c.param_refs[0], label_at(c, 0));
                const bool b = ref_present(t, c.param_refs[1], label_at(c, 1));
                if (a && b) return false;
                break;
            }
            default:  // CONDITION_* handled by apply_conditions (activation)
                break;
        }
    }
    return true;
}

::n4m_trial_s* Optimizer::find(std::int64_t id) const {
    for (const auto& up : trials_) {
        if (up->id == id) return up.get();
    }
    return nullptr;
}

n4m_status_t Optimizer::ask(::n4m_trial_s** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    if (next_id_ == std::numeric_limits<std::int64_t>::max() ||
        next_event_sequence_ == std::numeric_limits<std::int64_t>::max()) {
        return N4M_ERR_INTERNAL;
    }
    if (opts_.timeout_seconds > 0.0) {
        const double elapsed =
            std::chrono::duration<double>(now() - start_time_).count();
        if (elapsed > opts_.timeout_seconds) return N4M_ERR_CANCELLED;
    }

    const std::vector<std::pair<std::string, double>>* forced = nullptr;
    if (!enqueued_.empty()) {
        // Keep the warm-start queued until the trial is fully committed. An
        // allocation/sampler exception must not silently consume it: callers
        // may safely retry a failed ask().
        forced = &enqueued_.front();
    }

    auto t = std::make_unique<::n4m_trial_s>();
    // Make the final ownership transfer non-allocating before sampling mutates
    // RNG/adaptive-sampler state.
    trials_.reserve(trials_.size() + 1);
    t->id = next_id_;
    t->ask_time = std::chrono::steady_clock::now();
    if (!sample(*t, forced)) {
        if (forced != nullptr) enqueued_.pop_front();
        return N4M_ERR_INVALID_ARGUMENT;  // constraints unsatisfiable (incl. an invalid warm-start)
    }
    t->ask_sequence = next_event_sequence_;
    trials_.push_back(std::move(t));
    if (forced != nullptr) enqueued_.pop_front();
    *out = trials_.back().get();
    ++next_id_;
    ++next_event_sequence_;
    return N4M_OK;
}

n4m_status_t Optimizer::ask_batch(std::int32_t n, ::n4m_trial_s** out,
                                  std::int32_t* out_count) {
    // Preconditions guaranteed by the C wrapper: out != nullptr with n
    // pre-nulled slots, out_count != nullptr with *out_count == 0, and n > 0.
    for (std::int32_t i = 0; i < n; ++i) {
        // Capture whether the upcoming ask is refused purely by a population
        // generation boundary BEFORE it consumes id `next_id_`. This is the
        // single seam that keeps a benign batch boundary from being confused
        // with a genuine unsatisfiable-constraint / invalid-warm-start refusal
        // (both surface as N4M_ERR_INVALID_ARGUMENT out of ask()).
        const bool boundary = at_generation_boundary(next_id_);
        ::n4m_trial_s* trial = nullptr;
        const n4m_status_t st = ask(&trial);
        if (st == N4M_OK) {
            out[i] = trial;
            ++(*out_count);
            continue;
        }
        // Benign stop once >=1 trial is committed: hand back a partial OK batch
        // and let the caller score/advance before asking again. Hit at zero
        // progress, the same condition surfaces its stable status so the caller
        // can tell "await scores" (INVALID_ARGUMENT) from "deadline reached"
        // (CANCELLED).
        if (st == N4M_ERR_CANCELLED) {  // ask() deadline
            return (*out_count > 0) ? N4M_OK : N4M_ERR_CANCELLED;
        }
        if (st == N4M_ERR_INVALID_ARGUMENT && boundary) {  // generation boundary
            return (*out_count > 0) ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
        }
        // Any other failure is genuine (unsatisfiable constraints, invalid
        // queued warm-start, internal): return its exact status. The committed
        // [0,*out_count) trials stay borrowed/valid and are never rolled back;
        // the remaining slots are still nullptr.
        return st;
    }
    return N4M_OK;
}

std::int32_t Optimizer::resolved_in_range(std::int64_t base, std::int32_t size) const {
    std::int32_t resolved = 0;
    for (const auto& up : trials_) {
        if (up->id >= base && up->id < base + size && up->status != N4M_TRIAL_RUNNING) ++resolved;
    }
    return resolved;
}

n4m_status_t Optimizer::enqueue(std::vector<std::pair<std::string, double>> params) {
    if (!allow_enqueue()) return N4M_ERR_UNSUPPORTED;  // e.g. population samplers
    std::unordered_set<std::string> names;
    names.reserve(params.size());
    for (auto& kv : params) {
        if (!names.insert(kv.first).second || !std::isfinite(kv.second)) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const ParamSpec* p = space_.find(kv.first);
        if (p == nullptr) return N4M_ERR_INVALID_ARGUMENT;  // unknown param
        switch (p->kind) {
            case N4M_PARAM_INT:
            case N4M_PARAM_LOG_INT: {
                if (kv.second < p->low || kv.second > p->high ||
                    std::trunc(kv.second) != kv.second) {
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                double snapped = 0.0;
                if (!snap_scalar_grid_value(*p, kv.second, &snapped)) {
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                break;
            }
            case N4M_PARAM_FLOAT:
            case N4M_PARAM_LOG_FLOAT: {
                if (kv.second < p->low || kv.second > p->high) {
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                double snapped = 0.0;
                if (!snap_scalar_grid_value(*p, kv.second, &snapped)) {
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                kv.second = snapped;
                break;
            }
            case N4M_PARAM_CATEGORICAL:
            case N4M_PARAM_ORDINAL: {
                const int n = static_cast<int>(std::max(p->labels.size(), p->num_values.size()));
                if (std::trunc(kv.second) != kv.second || kv.second < 0.0 ||
                    kv.second >= static_cast<double>(n)) {
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                break;
            }
            default:  // SORTED_TUPLE not warm-startable via a single value
                return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    enqueued_.push_back(std::move(params));
    return N4M_OK;
}

n4m_status_t Optimizer::tell_result(std::int64_t id, n4m_trial_status_t status, double score,
                                    const TrialError* error) {
    if (status != N4M_TRIAL_COMPLETED && status != N4M_TRIAL_PRUNED &&
        status != N4M_TRIAL_FAILED && status != N4M_TRIAL_CANCELLED) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    ::n4m_trial_s* t = find(id);
    if (t == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    if (status == N4M_TRIAL_COMPLETED && !std::isfinite(score)) return N4M_ERR_INVALID_ARGUMENT;

    const bool error_status =
        status == N4M_TRIAL_FAILED || status == N4M_TRIAL_CANCELLED;
    if (!error_status && error != nullptr) return N4M_ERR_INVALID_ARGUMENT;
    TrialError effective_error;
    if (error_status) {
        if (error == nullptr) return N4M_ERR_INVALID_ARGUMENT;
        effective_error = *error;
        if (effective_error.code.empty() || effective_error.message.empty()) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }

    if (t->status != N4M_TRIAL_RUNNING) {
        // Terminal state is immutable. Replays are accepted only when the full
        // terminal payload matches, so retries cannot silently rewrite history.
        if (t->status != status) return N4M_ERR_INVALID_ARGUMENT;
        if (status == N4M_TRIAL_COMPLETED) {
            return t->has_score && t->score == score ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
        }
        if (error_status) {
            return t->has_error && t->error == effective_error ? N4M_OK
                                                                : N4M_ERR_INVALID_ARGUMENT;
        }
        return N4M_OK;  // PRUNED has no caller-supplied terminal payload.
    }
    if (next_event_sequence_ == std::numeric_limits<std::int64_t>::max()) {
        return N4M_ERR_INTERNAL;
    }
    t->status = status;
    if (status == N4M_TRIAL_COMPLETED) {
        t->score = score;
        t->has_score = true;
    }
    if (error_status) {
        t->error = std::move(effective_error);
        t->has_error = true;
    }
    t->terminal_sequence = next_event_sequence_++;
    t->duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t->ask_time).count();
    return N4M_OK;
}

n4m_status_t Optimizer::tell_intermediate(std::int64_t id, std::int32_t step, double score,
                                          std::int32_t* out_should_prune) {
    if (out_should_prune != nullptr) *out_should_prune = 0;
    if (step < 0) return N4M_ERR_INVALID_ARGUMENT;
    ::n4m_trial_s* t = find(id);
    if (t == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    if (!std::isfinite(score)) return N4M_ERR_INVALID_ARGUMENT;
    // A pruning decision terminalizes the trial inside this call. Preserve
    // at-least-once delivery semantics for an exact retry of that final report.
    if (t->status == N4M_TRIAL_PRUNED && t->pruned_by_policy &&
        !t->intermediates.empty() && t->intermediates.back().first == step &&
        t->intermediates.back().second == score) {
        if (out_should_prune != nullptr) *out_should_prune = 1;
        return N4M_OK;
    }
    if (t->status != N4M_TRIAL_RUNNING) {
        return N4M_ERR_INVALID_ARGUMENT;  // terminal => no more rungs
    }

    // The trace is append-only and ordered. An exact replay of the latest
    // non-pruning value is idempotent; rewrites and out-of-order steps fail.
    if (!t->intermediates.empty()) {
        const auto& latest = t->intermediates.back();
        if (step < latest.first) return N4M_ERR_INVALID_ARGUMENT;
        if (step == latest.first) {
            return latest.second == score ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
        }
    }
    if (next_event_sequence_ > std::numeric_limits<std::int64_t>::max() - 2) {
        return N4M_ERR_INTERNAL;
    }
    // Reserve both append-only streams before changing either one. If an
    // allocation fails, the C wrapper reports OOM and the trace stays intact.
    t->intermediates.reserve(t->intermediates.size() + 1);
    t->intermediate_sequences.reserve(t->intermediate_sequences.size() + 1);
    t->intermediates.emplace_back(step, score);
    t->intermediate_sequences.push_back(next_event_sequence_);
    bool prune = false;
    try {
        if (pruner_ != nullptr) {
            prune = pruner_->should_prune(*t, step, score, trials_, dir_);
        }
    } catch (...) {
        // Pruners may allocate peer-score scratch space. Roll back the staged
        // report if they fail so an OOM/exception cannot commit an event while
        // returning an error, or turn a retry into a false non-prune replay.
        t->intermediate_sequences.pop_back();
        t->intermediates.pop_back();
        throw;
    }
    ++next_event_sequence_;
    t->rung = step;
    if (prune) {
        t->status = N4M_TRIAL_PRUNED;
        t->pruned_by_policy = true;
        t->terminal_sequence = next_event_sequence_++;
        t->duration_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t->ask_time).count();
    }
    if (out_should_prune != nullptr) *out_should_prune = prune ? 1 : 0;
    return N4M_OK;
}

::n4m_trial_s* Optimizer::best(double* out_score) const {
    ::n4m_trial_s* best_trial = nullptr;
    double best_score = 0.0;
    for (const auto& up : trials_) {
        if (up->status != N4M_TRIAL_COMPLETED || !up->has_score) continue;
        if (best_trial == nullptr || better(up->score, best_score)) {
            best_trial = up.get();
            best_score = up->score;
        }
    }
    if (best_trial != nullptr && out_score != nullptr) *out_score = best_score;
    return best_trial;
}

std::unique_ptr<Optimizer> make_optimizer(const SearchSpace& space,
                                          const n4m_optimizer_options_t& opts,
                                          n4m_status_t* status,
                                          std::string* error) {
    const n4m_status_t options_status = validate_optimizer_options(opts, error);
    if (options_status != N4M_OK) {
        if (status != nullptr) *status = options_status;
        return nullptr;
    }
    const n4m_status_t validation_status = space.validate(opts.sampler, error);
    if (validation_status != N4M_OK) {
        if (status != nullptr) *status = validation_status;
        return nullptr;
    }
    {
        // Construction remains centralized even though semantic option validation
        // already rejected unknown kinds and invalid Hyperband resources.
        n4m_status_t pruner_status = N4M_OK;
        (void)make_pruner(opts, &pruner_status);
        if (pruner_status != N4M_OK) {
            if (status != nullptr) *status = pruner_status;
            return nullptr;
        }
    }
    switch (opts.sampler) {
        case N4M_SAMPLER_RANDOM:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<Optimizer>(space, opts);
        case N4M_SAMPLER_SOBOL:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<SobolSampler>(space, opts);
        case N4M_SAMPLER_TERNARY:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<TernarySampler>(space, opts);
        case N4M_SAMPLER_LHS:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<LhsSampler>(space, opts);
        case N4M_SAMPLER_GA:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<GaSampler>(space, opts);
        case N4M_SAMPLER_PSO:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<PsoSampler>(space, opts);
        case N4M_SAMPLER_CMAES:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<CmaEsSampler>(space, opts);
        case N4M_SAMPLER_TPE:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<TpeSampler>(space, opts);
        case N4M_SAMPLER_GP_EI:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<GpEiSampler>(space, opts);
        default:  // all sampler kinds implemented
            if (error != nullptr) *error = "unknown or unimplemented sampler kind";
            if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;
            return nullptr;
    }
}

}  // namespace n4m::core::opt
