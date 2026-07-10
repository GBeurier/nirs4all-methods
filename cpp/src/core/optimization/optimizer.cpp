// SPDX-License-Identifier: CECILL-2.1
//
// Ask/tell optimizer core (F0): SearchSpace + Trial helpers + the base
// Optimizer (random sampler, none pruner) + the sampler factory.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace n4m::core::opt {

// ---- SearchSpace / Trial lookups ----------------------------------------

ParamSpec* SearchSpace::find(const std::string& name) {
    for (auto& p : params) {
        if (p.name == name) return &p;
    }
    return nullptr;
}
const ParamSpec* SearchSpace::find(const std::string& name) const {
    for (const auto& p : params) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

const TrialParam* Trial::find(const std::string& name) const {
    for (const auto& kv : params) {
        if (kv.first == name) return &kv.second;
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
}

bool Optimizer::better(double candidate, double incumbent) const {
    return dir_ == N4M_OPT_MAXIMIZE ? candidate > incumbent : candidate < incumbent;
}

double Optimizer::sample_numeric(const ParamSpec& p) {
    const double u = n4m_rng_next_double(&rng_);  // [0, 1)
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

void Optimizer::sample(::n4m_trial_s& t) {
    constexpr int kMaxTries = 200;
    for (int attempt = 0; attempt < kMaxTries; ++attempt) {
        t.params.clear();
        for (const auto& p : space_.params) {
            if (p.kind == N4M_PARAM_SORTED_TUPLE) {
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
                default:  // INT / FLOAT / LOG_INT / LOG_FLOAT
                    tp.value = sample_numeric(p);
                    break;
            }
            t.params.emplace_back(p.name, tp);
        }
        apply_conditions(t);
        if (constraints_ok(t)) return;
    }
    // best-effort: constraints unsatisfiable within the retry budget; keep last.
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
    auto t = std::make_unique<::n4m_trial_s>();
    t->id = next_id_++;
    sample(*t);
    if (!enqueued_.empty()) {
        const auto forced = std::move(enqueued_.front());
        enqueued_.pop_front();
        for (const auto& fv : forced) {
            for (auto& kv : t->params) {
                if (kv.first != fv.first) continue;
                kv.second.value = fv.second;
                const ParamSpec* p = space_.find(fv.first);
                if (p != nullptr &&
                    (p->kind == N4M_PARAM_CATEGORICAL || p->kind == N4M_PARAM_ORDINAL)) {
                    int idx = static_cast<int>(std::llround(fv.second));
                    const int n = static_cast<int>(std::max(p->labels.size(), p->num_values.size()));
                    if (idx < 0) idx = 0;
                    if (n > 0 && idx >= n) idx = n - 1;
                    kv.second.cat_index = idx;
                    if (idx < static_cast<int>(p->labels.size())) {
                        kv.second.cat_label = p->labels[static_cast<std::size_t>(idx)];
                    }
                    if (idx < static_cast<int>(p->num_values.size())) {
                        kv.second.value = p->num_values[static_cast<std::size_t>(idx)];
                    }
                }
            }
        }
        apply_conditions(*t);
    }
    *out = t.get();
    trials_.push_back(std::move(t));
    return N4M_OK;
}

n4m_status_t Optimizer::enqueue(std::vector<std::pair<std::string, double>> params) {
    enqueued_.push_back(std::move(params));
    return N4M_OK;
}

n4m_status_t Optimizer::tell_result(std::int64_t id, n4m_trial_status_t status, double score) {
    ::n4m_trial_s* t = find(id);
    if (t == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    t->status = status;
    if (status == N4M_TRIAL_COMPLETED) {
        t->score = score;
        t->has_score = true;
    }
    return N4M_OK;
}

n4m_status_t Optimizer::tell_intermediate(std::int64_t id, std::int32_t step, double score,
                                          std::int32_t* out_should_prune) {
    ::n4m_trial_s* t = find(id);
    if (t == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    t->intermediates.emplace_back(step, score);
    t->rung = step;
    if (out_should_prune != nullptr) *out_should_prune = 0;  // `none` pruner never prunes
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
    if (opts.pruner != N4M_PRUNER_NONE) {
        if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;  // pruners land in F2
        return nullptr;
    }
    switch (opts.sampler) {
        case N4M_SAMPLER_RANDOM:
            if (status != nullptr) *status = N4M_OK;
            return std::make_unique<Optimizer>(space, opts);
        default:  // sobol/lhs/ternary/ga/pso/cmaes/tpe/gp_ei reserved for F1–F4
            if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;
            return nullptr;
    }
}

}  // namespace n4m::core::opt
