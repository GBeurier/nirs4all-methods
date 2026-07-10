// SPDX-License-Identifier: CECILL-2.1
//
// Internal ask/tell hyperparameter optimizer core (F0). Search-space model,
// trial table, and the base Optimizer with the `random` sampler + `none`
// pruner. Later phases add Sampler/Pruner subclasses selected by
// n4m_sampler_kind_t / n4m_pruner_kind_t without changing the C ABI.

#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "n4m/n4m.h"
#include "n4m/optimization.h"

#include "core/common/rng_engine.h"

namespace n4m::core::opt {

// One tunable parameter in the search space.
struct ParamSpec {
    std::string      name;
    n4m_param_kind_t kind{N4M_PARAM_FLOAT};

    // numeric (int / float / log_int / log_float)
    double low{0.0};
    double high{0.0};
    double step{0.0};   // <= 0 => continuous / unit
    bool   is_int{false};
    bool   is_log{false};

    // categorical / ordinal
    n4m_cat_type_t           cat_type{N4M_CAT_STR};
    std::vector<std::string> labels;      // categorical labels (STR) or stringified values
    std::vector<double>      num_values;  // ordinal values / numeric-categorical canonical values

    // sorted_tuple
    std::int32_t tuple_length{0};
    bool         tuple_element_is_int{false};

    // conditional activation (populated from a CONDITION_IN / _NOT_IN constraint)
    std::string              cond_parent;   // empty => always active
    std::vector<std::string> cond_labels;
    bool                     cond_is_in{true};
};

struct Constraint {
    n4m_constraint_kind_t    kind{N4M_CONSTRAINT_MUTEX_GROUP};
    std::vector<std::string> param_refs;
    std::vector<std::string> label_refs;   // parallel to param_refs; entry "" means "no label"
};

class SearchSpace {
  public:
    std::vector<ParamSpec>  params;
    std::vector<Constraint> constraints;

    ParamSpec*       find(std::string_view name);
    const ParamSpec* find(std::string_view name) const;
};

// A single sampled value inside a trial.
struct TrialParam {
    double       value{0.0};      // numeric value; for categorical/ordinal the chosen value
    std::int32_t cat_index{-1};   // categorical/ordinal choice index (-1 for pure numeric)
    std::string  cat_label;       // categorical label (or stringified value)
    bool         active{true};
};

// A trial: ordered (name -> value) plus lifecycle state.
struct Trial {
    std::int64_t                                       id{0};
    std::vector<std::pair<std::string, TrialParam>>    params;
    n4m_trial_status_t                                 status{N4M_TRIAL_RUNNING};
    double                                             score{0.0};
    bool                                               has_score{false};
    double                                             duration_seconds{0.0};
    std::int32_t                                       rung{0};
    std::vector<std::pair<std::int32_t, double>>       intermediates;
    std::chrono::steady_clock::time_point              ask_time{};

    const TrialParam* find(std::string_view name) const;
};

}  // namespace n4m::core::opt

// Opaque public handles: the actual objects are these `_s` types so the
// static_cast in c_api_optimization.cpp is a safe upcast (same convention as
// n4m_validation_plan_s / n4m_method_result_s).
struct n4m_trial_s        : public ::n4m::core::opt::Trial {};
struct n4m_search_space_s : public ::n4m::core::opt::SearchSpace {};

namespace n4m::core::opt {

// Early-stopping policy over the tell_intermediate() score stream. Composed with
// (and orthogonal to) the sampler. The decision is a pure function of the
// recorded intermediate histories, so it is decision-level testable against a
// canned history.
class Pruner {
  public:
    virtual ~Pruner() = default;
    virtual bool should_prune(const ::n4m_trial_s& trial, std::int32_t step, double score,
                              const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                              n4m_opt_direction_t dir) const = 0;
};

// Median stopping rule (Vizier): prune a trial when its latest intermediate
// score is worse than the median of the peer trials' scores at the same step.
// Never prunes before `min_peers` peers exist or during the warm-up steps.
class MedianPruner : public Pruner {
  public:
    MedianPruner(std::int32_t min_peers, std::int32_t warmup_steps)
        : min_peers_(min_peers), warmup_steps_(warmup_steps) {}
    bool should_prune(const ::n4m_trial_s& trial, std::int32_t step, double score,
                      const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                      n4m_opt_direction_t dir) const override;

  private:
    std::int32_t min_peers_;
    std::int32_t warmup_steps_;
};

// Asynchronous Successive Halving (ASHA): at each rung (step), a trial survives
// only if its score is in the top 1/reduction_factor of the peers that have
// reached the same rung; otherwise it is pruned. Asynchronous — decided on the
// fly against whoever has reported at that rung. F2.
class AshaPruner : public Pruner {
  public:
    explicit AshaPruner(std::int32_t reduction_factor)
        : reduction_factor_(reduction_factor < 2 ? 2 : reduction_factor) {}
    bool should_prune(const ::n4m_trial_s& trial, std::int32_t step, double score,
                      const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                      n4m_opt_direction_t dir) const override;

  private:
    std::int32_t reduction_factor_;
};

// Build the pruner for opts.pruner; nullptr + N4M_ERR_NOT_IMPLEMENTED for
// reserved-but-unimplemented kinds. NONE maps to a null pruner (never prunes).
std::unique_ptr<Pruner> make_pruner(const n4m_optimizer_options_t& opts, n4m_status_t* status);

class Optimizer {
  public:
    Optimizer(const SearchSpace& space, const n4m_optimizer_options_t& opts);
    virtual ~Optimizer() = default;

    // Draw the next trial (owned by the optimizer, valid until destroy).
    n4m_status_t ask(::n4m_trial_s** out);
    // Force the next ask to return these numeric params.
    n4m_status_t enqueue(std::vector<std::pair<std::string, double>> params);
    // Report a terminal outcome for a trial.
    n4m_status_t tell_result(std::int64_t id, n4m_trial_status_t status, double score);
    // Report an intermediate (fidelity-rung) score; `none` pruner never prunes.
    n4m_status_t tell_intermediate(std::int64_t id, std::int32_t step, double score,
                                   std::int32_t* out_should_prune);
    ::n4m_trial_s*      best(double* out_score) const;
    const std::vector<std::unique_ptr<::n4m_trial_s>>& trials() const { return trials_; }
    n4m_opt_direction_t direction() const { return dir_; }

  protected:
    // Sample one trial, honouring `forced` values (unforced dimensions are
    // sampled). Returns false when constraints cannot be satisfied within the
    // retry budget. A subclass overrides for adaptive samplers.
    virtual bool sample(::n4m_trial_s& t,
                        const std::vector<std::pair<std::string, double>>* forced);

    double sample_numeric(const ParamSpec& p);
    // Deterministic map of a unit-cube coordinate u∈[0,1) to a numeric param
    // value (shared by random / lhs / sobol; the only stochastic part is who
    // supplies u).
    double numeric_from_unit(const ParamSpec& p, double u) const;
    void   apply_conditions(::n4m_trial_s& t) const;
    bool   constraints_ok(const ::n4m_trial_s& t) const;
    bool   better(double candidate, double incumbent) const;
    void   set_trial_value(::n4m_trial_s& t, const ParamSpec& p, double forced) const;

    // Hook for adaptive samplers to override one numeric param's value; the base
    // random sampler returns false so sample_numeric() is used. Return true and
    // set *out to override the draw for `p`.
    virtual bool override_numeric(const ParamSpec& p, double* out) {
        (void)p;
        (void)out;
        return false;
    }

    SearchSpace              space_;
    n4m_optimizer_options_t  opts_;
    n4m_opt_direction_t      dir_{N4M_OPT_MINIMIZE};
    n4m_rng                  rng_{};
    std::chrono::steady_clock::time_point start_time_{};
    std::unique_ptr<Pruner>  pruner_;

  private:
    ::n4m_trial_s* find(std::int64_t id) const;
    bool ref_present(const ::n4m_trial_s& t, const std::string& param,
                     const std::string& label) const;

    std::vector<std::unique_ptr<::n4m_trial_s>>              trials_;
    std::deque<std::vector<std::pair<std::string, double>>>  enqueued_;
    std::int64_t                                             next_id_{0};
};

// Ternary search over a single unimodal integer axis (ports the nirs4all
// BinarySearchSampler); every other parameter is sampled uniformly. Converges
// in O(log n) evaluations for a unimodal objective (e.g. PLS n_components). F1.
class TernarySampler : public Optimizer {
  public:
    TernarySampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    // Recomputed from the completed-trial history each call, so it is idempotent
    // within one ask() (safe under constraint-retries) and needs no mutable state.
    bool override_numeric(const ParamSpec& p, double* out) override;

  private:
    std::int64_t next_ternary_value() const;

    std::string  target_;   // name of the tuned int axis ("" ⇒ pure random)
    std::int64_t low_{0};
    std::int64_t high_{0};
    std::int64_t step_{1};
    std::int64_t grid_{0};  // number of grid points; ternary searches index space
};

// Latin Hypercube sampling over the numeric axes for the first `n_startup_trials`
// asks (stratified + jittered per dimension), falling back to uniform random
// beyond the startup batch and for categorical axes. F1.
class LhsSampler : public Optimizer {
  public:
    LhsSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    bool override_numeric(const ParamSpec& p, double* out) override;

  private:
    std::vector<std::string>         numeric_names_;  // numeric axes, in space order
    std::vector<std::vector<double>> unit_;           // unit_[axis][i] ∈ [0,1), i<n_startup
    std::int32_t                     n_startup_{0};
};

// Resolve MINIMIZE/MAXIMIZE from a metric (used when direction == AUTO).
n4m_opt_direction_t direction_for_metric(n4m_metric_t metric);

// Create the sampler for `opts.sampler`; returns nullptr (and sets *status to
// N4M_ERR_NOT_IMPLEMENTED) for reserved-but-unimplemented kinds.
std::unique_ptr<Optimizer> make_optimizer(const SearchSpace& space,
                                          const n4m_optimizer_options_t& opts,
                                          n4m_status_t* status);

}  // namespace n4m::core::opt

struct n4m_optimizer_s {
    std::unique_ptr<::n4m::core::opt::Optimizer> impl;
};
