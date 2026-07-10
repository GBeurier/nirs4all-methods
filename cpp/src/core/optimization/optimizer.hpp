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

// Hoeffding racing — the fold-safe early-stop. Each trial's intermediate scores
// are treated as repeated observations of its performance; a trial is pruned
// when its confidence interval no longer overlaps the best trial's (i.e. we are
// statistically confident it is worse). Correct for exchangeable-fold CV, where
// successive-halving's rank-preservation assumption fails (roadmap §2c). F2.
class RacingPruner : public Pruner {
  public:
    explicit RacingPruner(double delta) : delta_(delta > 0.0 && delta < 1.0 ? delta : 0.05) {}
    bool should_prune(const ::n4m_trial_s& trial, std::int32_t step, double score,
                      const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                      n4m_opt_direction_t dir) const override;

  private:
    double delta_;
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
    // Inverse of numeric_from_unit (approximate; ignores step): map a param value
    // back to its unit coordinate ∈ [0,1]. Used by TPE to model history in a
    // uniform space.
    double unit_from_numeric(const ParamSpec& p, double value) const;
    // Decode a full unit vector u∈[0,1)^P into a trial (numeric via
    // numeric_from_unit, categorical/ordinal bucketed, sorted-tuple sampled),
    // honouring forced values and applying conditional activation. Shared by the
    // population samplers (ga / pso). u must have space_.params.size() entries.
    void   decode_candidate(const std::vector<double>& u, ::n4m_trial_s& t,
                            const std::vector<std::pair<std::string, double>>* forced);
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
    // Hook for adaptive samplers to override a categorical/ordinal choice index;
    // the base returns false so a uniform choice is drawn. Return true and set
    // *out_index to override.
    virtual bool override_categorical(const ParamSpec& p, std::int32_t* out_index) {
        (void)p;
        (void)out_index;
        return false;
    }
    // Whether enqueue()/warm-start is supported. Population samplers (ga/pso)
    // return false because a forced candidate cannot be inverse-encoded into
    // their population state.
    virtual bool allow_enqueue() const { return true; }
    // Number of trials in the id range [base, base+size) that have reached a
    // terminal state (used by the population samplers' generation guard).
    std::int32_t resolved_in_range(std::int64_t base, std::int32_t size) const;

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

// Real-coded genetic algorithm over the unit hypercube (F3). Each candidate is a
// unit vector u∈[0,1)^P decoded per parameter (numeric_from_unit for numeric,
// bucketed for categorical/ordinal). A generation of `pop_size` candidates is
// handed out via ask(); once the generation's trials complete, tournament
// selection + uniform crossover + Gaussian mutation produce the next one.
// Handles mixed spaces uniformly; sorted-tuple axes fall back to the base
// sampler. (A later F3 refinement may share the RNG-consolidated loops with the
// feature-selection GA — see FINETUNING_ROADMAP.md.)
class GaSampler : public Optimizer {
  public:
    GaSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    bool sample(::n4m_trial_s& t,
                const std::vector<std::pair<std::string, double>>* forced) override;
    bool allow_enqueue() const override { return false; }

  private:
    void        ensure_generation(std::int64_t member_base);
    std::size_t genome_length() const { return space_.params.size(); }

    std::int32_t                     pop_size_{16};
    double                           mutation_rate_{0.15};
    double                           mutation_sigma_{0.15};
    std::vector<std::vector<double>> gen_;          // current generation unit vectors
    std::int64_t                     gen_base_id_{-1};  // trial id of gen_[0]
};

// Particle Swarm Optimization over the unit hypercube (F3). A swarm of
// `swarm_size` particles (position + velocity + personal best) is asked out per
// iteration; once scored, each velocity updates toward the particle's personal
// best and the global best (Clerc-Kennedy constants), then positions advance.
// Reuses the async-population lifecycle and the shared candidate decode. F3.
class PsoSampler : public Optimizer {
  public:
    PsoSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    bool sample(::n4m_trial_s& t,
                const std::vector<std::pair<std::string, double>>* forced) override;
    bool allow_enqueue() const override { return false; }

  private:
    void ensure_iteration(std::int64_t member_base);

    std::int32_t                     swarm_size_{16};
    double                           w_{0.729};
    double                           c1_{1.494};
    double                           c2_{1.494};
    std::vector<std::vector<double>> pos_;
    std::vector<std::vector<double>> vel_;
    std::vector<std::vector<double>> pbest_pos_;
    std::vector<double>              pbest_score_;
    std::vector<double>              gbest_pos_;
    double                           gbest_score_{0.0};
    bool                             have_gbest_{false};
    std::int64_t                     iter_base_id_{-1};
};

// Separable CMA-ES (Hansen & Ostermeier; Ros & Hansen 2008 diagonal variant) over
// the unit hypercube (F4). Samples a generation of λ points from N(m, σ²·diag(C)),
// clamped to [0,1); once scored, the μ best update the mean, the (diagonal)
// covariance, the step-size and the evolution paths. The diagonal covariance
// avoids an eigendecomposition, so it scales to modest continuous dimensionality
// (Ridge alpha, learning rates, continuous preprocessing params). Non-continuous
// axes are handled by the shared decode (bucketed), giving Optuna's
// independent-fallback behaviour for mixed spaces. F4.
class CmaEsSampler : public Optimizer {
  public:
    CmaEsSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    bool sample(::n4m_trial_s& t,
                const std::vector<std::pair<std::string, double>>* forced) override;
    bool allow_enqueue() const override { return false; }

  private:
    void ensure_generation(std::int64_t member_base);
    void sample_population();

    std::vector<std::size_t> cont_axes_;  // param indices CMA-ES optimizes (numeric axes only)
    std::size_t P_{0};                    // == cont_axes_.size()
    std::int32_t lambda_{0};
    std::int32_t mu_{0};
    std::vector<double> weights_;
    double mu_eff_{1.0};
    double c_sigma_{0.0}, d_sigma_{1.0}, c_c_{0.0}, c_1_{0.0}, c_mu_{0.0}, chi_n_{1.0};

    std::vector<double> m_;       // distribution mean (unit space)
    double              sigma_{0.2};
    std::vector<double> C_;       // diagonal covariance
    std::vector<double> ps_;      // step-size evolution path
    std::vector<double> pc_;      // covariance evolution path
    std::int64_t        gen_count_{0};

    std::vector<std::vector<double>> pop_x_;  // clamped candidate positions (continuous axes)
    std::int64_t                     gen_base_id_{-1};
};

// Sobol low-discrepancy sequence (Joe-Kuo direction numbers, unscrambled) over
// the search space, one Sobol dimension per parameter. Bit-identical to
// scipy.stats.qmc.Sobol(scramble=False) for the numeric axes (verified in the
// Python parity test). Better space-filling than i.i.d. random at small budgets.
// F1. (Scrambled Sobol — the Tier-B randomised variant — is a later addition.)
class SobolSampler : public Optimizer {
  public:
    SobolSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    bool override_numeric(const ParamSpec& p, double* out) override;
    bool override_categorical(const ParamSpec& p, std::int32_t* out_index) override;

  private:
    void ensure_point();               // advance/cache the Sobol point for the current ask
    int  dim_of(const ParamSpec& p) const;  // Sobol dimension for p, or -1 if beyond the table

    int                         sobol_dims_{0};
    std::vector<std::uint32_t>  x_;            // Gray-code state per dimension
    std::vector<double>         cached_point_; // unit coords cached for the current ask
    std::int64_t                cached_ask_index_{-1};
    std::int64_t                next_index_{0};
};

// Univariate Tree-structured Parzen Estimator (F4). After `n_startup_trials`
// random trials, each parameter is modelled independently: the completed history
// is split into a "good" set (top γ by score) and the rest; l(x) and g(x) are
// Parzen (KDE) densities over good/bad numeric values (or category frequencies).
// A batch of n_ei candidates is drawn from l and the one maximising l(x)/g(x) is
// returned — the Optuna-default sampler for mixed / conditional spaces. F4.
class TpeSampler : public Optimizer {
  public:
    TpeSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts);

  protected:
    bool override_numeric(const ParamSpec& p, double* out) override;
    bool override_categorical(const ParamSpec& p, std::int32_t* out_index) override;

  private:
    std::int32_t n_startup_{10};
    double       gamma_{0.25};
    std::int32_t n_ei_{24};
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
