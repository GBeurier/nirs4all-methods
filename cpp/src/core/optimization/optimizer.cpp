// SPDX-License-Identifier: CECILL-2.1
//
// Ask/tell optimizer core (F0): SearchSpace + Trial helpers + the base
// Optimizer (random sampler, none pruner) + the sampler factory.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <string_view>

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

double Optimizer::numeric_from_unit(const ParamSpec& p, double u) const {
    if (u < 0.0) u = 0.0;
    if (u >= 1.0) u = std::nextafter(1.0, 0.0);
    double v;
    if (p.is_log && p.low > 0.0 && p.high > 0.0) {
        const double lo = std::log(p.low);
        const double hi = std::log(p.high);
        v = std::exp(lo + u * (hi - lo));
        if (p.is_int) v = std::round(v);
    } else if (p.is_int) {
        v = std::floor(p.low + u * (p.high - p.low + 1.0));
        if (v > p.high) v = p.high;
    } else {
        v = p.low + u * (p.high - p.low);
    }
    if (p.step > 0.0 && !p.is_log) {
        const double k = std::round((v - p.low) / p.step);
        v = p.low + k * p.step;
        if (v > p.high) v = p.high;
        if (v < p.low) v = p.low;
    }
    return v;
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
                double v = p.low + uu * (p.high - p.low);
                if (p.tuple_element_is_int) v = std::floor(p.low + uu * (p.high - p.low + 1.0));
                vals.push_back(v);
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
                    double v = p.low + u * (p.high - p.low);
                    if (p.tuple_element_is_int) {
                        v = std::floor(p.low + u * (p.high - p.low + 1.0));
                    }
                    vals.push_back(v);
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
                    int idx = n > 0 ? static_cast<int>(n4m_rng_next_double(&rng_) * n) : 0;
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
                    int idx = n > 0 ? static_cast<int>(n4m_rng_next_double(&rng_) * n) : 0;
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
    for (const auto& p : space_.params) {
        if (p.cond_parent.empty()) continue;
        const TrialParam* parent = t.find(p.cond_parent);
        bool in = false;
        if (parent != nullptr) {
            for (const auto& l : p.cond_labels) {
                if (l == parent->cat_label) { in = true; break; }
            }
        }
        const bool active = p.cond_is_in ? in : !in;
        const std::string prefix = p.name + "#";
        for (auto& kv : t.params) {
            if (kv.first == p.name || kv.first.rfind(prefix, 0) == 0) {
                kv.second.active = active;
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
    if (opts_.timeout_seconds > 0.0) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
        if (elapsed > opts_.timeout_seconds) return N4M_ERR_CANCELLED;
    }

    std::vector<std::pair<std::string, double>> forced_storage;
    const std::vector<std::pair<std::string, double>>* forced = nullptr;
    if (!enqueued_.empty()) {
        forced_storage = std::move(enqueued_.front());
        enqueued_.pop_front();
        forced = &forced_storage;
    }

    auto t = std::make_unique<::n4m_trial_s>();
    t->id = next_id_;
    t->ask_time = std::chrono::steady_clock::now();
    if (!sample(*t, forced)) {
        return N4M_ERR_INVALID_ARGUMENT;  // constraints unsatisfiable (incl. an invalid warm-start)
    }
    ++next_id_;
    *out = t.get();
    trials_.push_back(std::move(t));
    return N4M_OK;
}

n4m_status_t Optimizer::enqueue(std::vector<std::pair<std::string, double>> params) {
    for (const auto& kv : params) {
        const ParamSpec* p = space_.find(kv.first);
        if (p == nullptr) return N4M_ERR_INVALID_ARGUMENT;  // unknown param
        switch (p->kind) {
            case N4M_PARAM_INT:
            case N4M_PARAM_FLOAT:
            case N4M_PARAM_LOG_INT:
            case N4M_PARAM_LOG_FLOAT:
                if (kv.second < p->low || kv.second > p->high) return N4M_ERR_INVALID_ARGUMENT;
                break;
            case N4M_PARAM_CATEGORICAL:
            case N4M_PARAM_ORDINAL: {
                const int n = static_cast<int>(std::max(p->labels.size(), p->num_values.size()));
                const int idx = static_cast<int>(std::llround(kv.second));
                if (idx < 0 || (n > 0 && idx >= n)) return N4M_ERR_INVALID_ARGUMENT;
                break;
            }
            default:  // SORTED_TUPLE not warm-startable via a single value
                return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    enqueued_.push_back(std::move(params));
    return N4M_OK;
}

n4m_status_t Optimizer::tell_result(std::int64_t id, n4m_trial_status_t status, double score) {
    ::n4m_trial_s* t = find(id);
    if (t == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    if (status == N4M_TRIAL_COMPLETED && !std::isfinite(score)) return N4M_ERR_INVALID_ARGUMENT;
    if (t->status != N4M_TRIAL_RUNNING) {
        // Terminal state is terminal: only an idempotent re-report of the SAME
        // status is accepted (e.g. tell_result(PRUNED) after an auto-prune).
        return t->status == status ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
    }
    t->status = status;
    if (status == N4M_TRIAL_COMPLETED) {
        t->score = score;
        t->has_score = true;
    }
    t->duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t->ask_time).count();
    return N4M_OK;
}

n4m_status_t Optimizer::tell_intermediate(std::int64_t id, std::int32_t step, double score,
                                          std::int32_t* out_should_prune) {
    ::n4m_trial_s* t = find(id);
    if (t == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    if (t->status != N4M_TRIAL_RUNNING) return N4M_ERR_INVALID_ARGUMENT;  // terminal ⇒ no more rungs
    if (!std::isfinite(score)) return N4M_ERR_INVALID_ARGUMENT;
    // One value per step (a re-report of the same step updates, not duplicates).
    bool updated = false;
    for (auto& im : t->intermediates) {
        if (im.first == step) { im.second = score; updated = true; break; }
    }
    if (!updated) t->intermediates.emplace_back(step, score);
    t->rung = step;
    bool prune = false;
    if (pruner_ != nullptr) prune = pruner_->should_prune(*t, step, score, trials_, dir_);
    if (prune) t->status = N4M_TRIAL_PRUNED;
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
                                          n4m_status_t* status) {
    {
        // Single authority for which pruner kinds are supported (rejects
        // hyperband/racing and any out-of-range enum value).
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
        default:  // sobol/cmaes/tpe/gp_ei reserved for F1/F4
            if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;
            return nullptr;
    }
}

}  // namespace n4m::core::opt
