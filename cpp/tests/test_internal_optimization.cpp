// SPDX-License-Identifier: CECILL-2.1
//
// Internal transaction-safety tests for the HPO optimizer. These need direct
// access to hidden C++ symbols and therefore run in n4m_internal_tests, which
// links the static archive.

#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "core/optimization/optimizer.hpp"

int run_internal_optimization_tests();

namespace {

int g_optimization_failures = 0;

#define OPT_CHECK(cond)                                                      \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            ++g_optimization_failures;                                       \
        }                                                                    \
    } while (0)

class ThrowOncePruner final : public ::n4m::core::opt::Pruner {
  public:
    bool should_prune(
        const ::n4m_trial_s&, std::int32_t, double,
        const std::vector<std::unique_ptr<::n4m_trial_s>>&,
        n4m_opt_direction_t) const override {
        if (throw_once_) {
            throw_once_ = false;
            throw std::bad_alloc();
        }
        return true;
    }

  private:
    mutable bool throw_once_{true};
};

class PrunerFailureOptimizer final : public ::n4m::core::opt::Optimizer {
  public:
    PrunerFailureOptimizer(const ::n4m::core::opt::SearchSpace& space,
                           const n4m_optimizer_options_t& options)
        : Optimizer(space, options) {
        pruner_ = std::make_unique<ThrowOncePruner>();
    }
};

class ThrowOnceSamplerOptimizer final : public ::n4m::core::opt::Optimizer {
  public:
    ThrowOnceSamplerOptimizer(const ::n4m::core::opt::SearchSpace& space,
                              const n4m_optimizer_options_t& options)
        : Optimizer(space, options) {}

  protected:
    bool sample(
        ::n4m_trial_s& trial,
        const std::vector<std::pair<std::string, double>>* forced) override {
        if (throw_once_) {
            throw_once_ = false;
            throw std::bad_alloc();
        }
        if (forced == nullptr) return false;
        return Optimizer::sample(trial, forced);
    }

  private:
    bool throw_once_{true};
};

// Deterministic clock seam (no public ABI): the fake steady-clock reading
// advances by `tick_` seconds on every observation, so the ask() deadline is
// crossed after a fixed number of asks — no wall-clock flakiness.
class ClockSeamOptimizer final : public ::n4m::core::opt::Optimizer {
  public:
    ClockSeamOptimizer(const ::n4m::core::opt::SearchSpace& space,
                       const n4m_optimizer_options_t& options, double initial, double tick)
        : Optimizer(space, options), elapsed_(initial), tick_(tick) {
        start_time_ = base_;  // anchor the deadline to the fake epoch
    }

  protected:
    std::chrono::steady_clock::time_point now() const override {
        const auto t =
            base_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(elapsed_));
        elapsed_ += tick_;
        return t;
    }

  private:
    std::chrono::steady_clock::time_point base_{};
    mutable double                        elapsed_{0.0};
    double                                tick_{0.0};
};

::n4m::core::opt::SearchSpace make_space() {
    ::n4m::core::opt::SearchSpace space;
    ::n4m::core::opt::ParamSpec param;
    param.name = "k";
    param.kind = N4M_PARAM_INT;
    param.low = 1.0;
    param.high = 3.0;
    param.step = 1.0;
    param.is_int = true;
    space.params.push_back(std::move(param));
    return space;
}

n4m_optimizer_options_t default_options() {
    n4m_optimizer_options_t options{};
    options.struct_size = sizeof(n4m_optimizer_options_t);
    options.sampler = N4M_SAMPLER_RANDOM;
    options.pruner = N4M_PRUNER_NONE;
    options.direction = N4M_OPT_AUTO;
    options.eval_mode = N4M_EVAL_MEAN;
    options.metric = N4M_METRIC_RMSE;
    options.liar = N4M_LIAR_NONE;
    options.n_startup_trials = 10;
    return options;
}

void test_pruner_exception_rolls_back_intermediate() {
    const auto space = make_space();
    n4m_optimizer_options_t options = default_options();
    PrunerFailureOptimizer optimizer(space, options);
    n4m_trial_s* trial = nullptr;
    OPT_CHECK(optimizer.ask(&trial) == N4M_OK);

    int32_t should_prune = -1;
    bool saw_bad_alloc = false;
    try {
        (void)optimizer.tell_intermediate(trial->id, 0, 2.0, &should_prune);
    } catch (const std::bad_alloc&) {
        saw_bad_alloc = true;
    }
    OPT_CHECK(saw_bad_alloc);
    OPT_CHECK(trial->intermediates.empty());
    OPT_CHECK(trial->intermediate_sequences.empty());
    OPT_CHECK(trial->status == N4M_TRIAL_RUNNING);

    OPT_CHECK(optimizer.tell_intermediate(trial->id, 0, 2.0, &should_prune) ==
              N4M_OK);
    OPT_CHECK(should_prune == 1);
    OPT_CHECK(trial->intermediate_sequences.size() == 1);
    OPT_CHECK(trial->intermediate_sequences[0] == 1);
    OPT_CHECK(trial->terminal_sequence == 2);
}

void test_sampler_exception_preserves_warm_start() {
    const auto space = make_space();
    n4m_optimizer_options_t options = default_options();
    ThrowOnceSamplerOptimizer optimizer(space, options);
    OPT_CHECK(optimizer.enqueue({{"k", 3.0}}) == N4M_OK);

    n4m_trial_s* trial = nullptr;
    bool saw_bad_alloc = false;
    try {
        (void)optimizer.ask(&trial);
    } catch (const std::bad_alloc&) {
        saw_bad_alloc = true;
    }
    OPT_CHECK(saw_bad_alloc);
    OPT_CHECK(optimizer.trials().empty());
    OPT_CHECK(optimizer.ask(&trial) == N4M_OK);
    OPT_CHECK(trial != nullptr);
    const auto* k = trial != nullptr ? trial->find("k") : nullptr;
    OPT_CHECK(k != nullptr);
    if (k != nullptr) OPT_CHECK(k->value == 3.0);
    if (trial != nullptr) {
        OPT_CHECK(trial->id == 0);
        OPT_CHECK(trial->ask_sequence == 0);
    }
}

void test_ask_batch_timeout_seam() {
    const auto space = make_space();
    n4m_optimizer_options_t options = default_options();
    options.timeout_seconds = 2.5;

    // Mid-batch timeout: asks at fake elapsed 0,1,2 pass (<= 2.5); the fourth
    // (elapsed 3) trips the deadline. ask_batch returns a benign partial OK with
    // count 3, then the next ask at the crossed deadline exposes CANCELLED.
    {
        ClockSeamOptimizer optimizer(space, options, /*initial=*/0.0, /*tick=*/1.0);
        n4m_trial_s* buf[10] = {nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, nullptr, nullptr};
        std::int32_t count = 0;  // C wrapper precondition: pre-zeroed, pre-nulled
        const n4m_status_t st = optimizer.ask_batch(10, buf, &count);
        OPT_CHECK(st == N4M_OK);
        OPT_CHECK(count == 3);
        for (int i = 0; i < 3; ++i) OPT_CHECK(buf[i] != nullptr);
        for (int i = 3; i < 10; ++i) OPT_CHECK(buf[i] == nullptr);
        n4m_trial_s* next_buf[4] = {nullptr, nullptr, nullptr, nullptr};
        std::int32_t next_count = 0;
        OPT_CHECK(optimizer.ask_batch(4, next_buf, &next_count) ==
                  N4M_ERR_CANCELLED);
        OPT_CHECK(next_count == 0);
        for (auto* slot : next_buf) OPT_CHECK(slot == nullptr);
        n4m_trial_s* one = nullptr;
        OPT_CHECK(optimizer.ask(&one) == N4M_ERR_CANCELLED);
    }
    // Timeout before the first ask: fake elapsed already past the deadline, so
    // the batch commits nothing and returns CANCELLED / count 0.
    {
        ClockSeamOptimizer optimizer(space, options, /*initial=*/5.0, /*tick=*/1.0);
        n4m_trial_s* buf[4] = {nullptr, nullptr, nullptr, nullptr};
        std::int32_t count = 0;
        const n4m_status_t st = optimizer.ask_batch(4, buf, &count);
        OPT_CHECK(st == N4M_ERR_CANCELLED);
        OPT_CHECK(count == 0);
        OPT_CHECK(buf[0] == nullptr);
    }
}

}  // namespace

int run_internal_optimization_tests() {
    test_pruner_exception_rolls_back_intermediate();
    test_sampler_exception_preserves_warm_start();
    test_ask_batch_timeout_seam();
    return g_optimization_failures;
}
