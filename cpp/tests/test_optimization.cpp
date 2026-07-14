// SPDX-License-Identifier: CECILL-2.1
//
// F0 — native ask/tell hyperparameter optimizer smoke tests. Drives the
// public C ABI end-to-end (search space, optimizer, ask/tell, best, trials,
// and the pure-native n4m_finetune_estimator over a real PLS cross-validation).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "harness.hpp"

void register_optimization_tests(n4m_testing::Runner& r);

namespace {

template <typename Enum>
Enum invalid_enum(std::int32_t value) {
    static_assert(sizeof(Enum) == sizeof(value));
    Enum result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template <typename Enum>
void set_invalid_enum(Enum& target, std::int32_t value) {
    static_assert(sizeof(Enum) == sizeof(value));
    std::memcpy(&target, &value, sizeof(target));
}

n4m_optimizer_options_t default_opts() {
    n4m_optimizer_options_t o;
    n4m_optimizer_options_init(&o);
    return o;
}

n4m_status_t optimizer_create_status_with_options(n4m_context_t* ctx,
                                                  n4m_search_space_t* space,
                                                  const n4m_optimizer_options_t& options) {
    n4m_optimizer_t* opt = nullptr;
    const n4m_status_t status = n4m_optimizer_create(ctx, space, &options, &opt);
    if (opt != nullptr) n4m_optimizer_destroy(opt);
    return status;
}

n4m_status_t optimizer_create_status(n4m_context_t* ctx, n4m_search_space_t* space,
                                     n4m_sampler_kind_t sampler = N4M_SAMPLER_RANDOM) {
    n4m_optimizer_options_t o = default_opts();
    o.sampler = sampler;
    return optimizer_create_status_with_options(ctx, space, o);
}

void test_options_init() {
    n4m_optimizer_options_t o;
    n4m_optimizer_options_init(&o);
    N4M_TEST_REQUIRE(o.struct_size == sizeof(n4m_optimizer_options_t));
    N4M_TEST_REQUIRE(o.sampler == N4M_SAMPLER_RANDOM);
    N4M_TEST_REQUIRE(o.pruner == N4M_PRUNER_NONE);
    N4M_TEST_REQUIRE(o.direction == N4M_OPT_AUTO);
    N4M_TEST_REQUIRE(o.eval_mode == N4M_EVAL_MEAN);
    N4M_TEST_REQUIRE(o.metric == N4M_METRIC_RMSE);
    N4M_TEST_REQUIRE(o.liar == N4M_LIAR_NONE);
    N4M_TEST_REQUIRE(o.n_startup_trials == 10);
#if INTPTR_MAX == INT64_MAX
    static_assert(sizeof(n4m_optimizer_options_t) == 120);
    static_assert(offsetof(n4m_optimizer_options_t, struct_size) == 0);
    static_assert(offsetof(n4m_optimizer_options_t, sampler) == 8);
    static_assert(offsetof(n4m_optimizer_options_t, n_startup_trials) == 32);
    static_assert(offsetof(n4m_optimizer_options_t, seed) == 40);
    static_assert(offsetof(n4m_optimizer_options_t, timeout_seconds) == 48);
    static_assert(offsetof(n4m_optimizer_options_t, reserved) == 64);
#endif
}

void test_options_validation() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);

    auto status = [&](const n4m_optimizer_options_t& options) {
        return optimizer_create_status_with_options(ctx, sp, options);
    };
    n4m_optimizer_options_t o = default_opts();
    o.struct_size = sizeof(o) - 1;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.struct_size = sizeof(o) + 16;  // forward layout tail is ignored
    N4M_TEST_REQUIRE(status(o) == N4M_OK);
    o = default_opts();
    o.struct_size = uint64_t{1} << 32;
    set_invalid_enum(o.direction, 99);
    // LP64 copies and validates the known prefix; ILP32 must reject the size
    // instead of truncating it to zero and silently accepting defaults.
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);

    o = default_opts();
    set_invalid_enum(o.direction, 99);
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    set_invalid_enum(o.metric, 99);
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    set_invalid_enum(o.eval_mode, 99);
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.eval_mode = N4M_EVAL_BEST;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_NOT_IMPLEMENTED);
    o = default_opts();
    o.eval_mode = N4M_EVAL_ROBUST_BEST;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_NOT_IMPLEMENTED);
    o = default_opts();
    set_invalid_enum(o.liar, 99);
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.liar = N4M_LIAR_MIN;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_NOT_IMPLEMENTED);

    o = default_opts();
    o.n_startup_trials = -1;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.sampler = N4M_SAMPLER_TPE;
    o.n_startup_trials = 0;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.sampler = N4M_SAMPLER_GP_EI;
    o.n_startup_trials = 0;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.pruner = N4M_PRUNER_MEDIAN;
    o.n_startup_trials = 0;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.sampler = N4M_SAMPLER_LHS;
    o.n_startup_trials = 0;  // explicit no-design fallback remains supported
    N4M_TEST_REQUIRE(status(o) == N4M_OK);
    o = default_opts();
    o.timeout_seconds = -1.0;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.timeout_seconds = std::nan("");
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.timeout_seconds = std::numeric_limits<double>::infinity();
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);

    o = default_opts();
    o.max_resource = -1;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.max_resource = 9;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.reduction_factor = 1;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.reduction_factor = -2;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.reduction_factor = 2;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o = default_opts();
    o.pruner = N4M_PRUNER_ASHA;
    o.reduction_factor = 2;
    N4M_TEST_REQUIRE(status(o) == N4M_OK);
    o = default_opts();
    o.pruner = N4M_PRUNER_HYPERBAND;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);
    o.max_resource = 9;
    N4M_TEST_REQUIRE(status(o) == N4M_OK);

    o = default_opts();
    o.reserved[7] = 1;
    N4M_TEST_REQUIRE(status(o) == N4M_ERR_INVALID_ARGUMENT);

    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_space_build() {
    n4m_search_space_t* sp = nullptr;
    N4M_TEST_REQUIRE(n4m_search_space_create(&sp) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_int(sp, "n_components", 1, 30, 1, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_float(sp, "alpha", 1e-4, 1e0, 0.0, 1) == N4M_OK);
    const int64_t widths[3] = {2, 4, 8};
    N4M_TEST_REQUIRE(
        n4m_search_space_add_categorical(sp, "width", N4M_CAT_INT, widths, 3) == N4M_OK);
    N4M_TEST_REQUIRE(
        n4m_search_space_add_sorted_tuple(sp, "alphas", 3, 0.0, 2.0, 0) == N4M_OK);
    const char* params[2] = {"alpha", "width"};
    const char* labels[2] = {"", ""};
    N4M_TEST_REQUIRE(
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_EXCLUDE, params, labels, 2) == N4M_OK);
    int32_t n = 0;
    N4M_TEST_REQUIRE(n4m_search_space_num_params(sp, &n) == N4M_OK);
    N4M_TEST_REQUIRE(n == 4);
    n4m_search_space_destroy(sp);
}

void test_reserved_sampler_not_implemented() {
    n4m_context_t* ctx = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -1.0, 1.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    set_invalid_enum(o.sampler, 99);  // out-of-range → NOT_IMPLEMENTED
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_ERR_NOT_IMPLEMENTED);
    N4M_TEST_REQUIRE(opt == nullptr);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_quadratic_converges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 123;
    o.direction = N4M_OPT_MINIMIZE;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 400; ++i) {
        n4m_trial_t* t = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);
        double x = 0.0;
        N4M_TEST_REQUIRE(n4m_trial_get_float(t, "x", &x) == N4M_OK);
        const double score = (x - 2.0) * (x - 2.0);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, id, score) == N4M_OK);
    }
    n4m_trial_t* best = nullptr;
    double best_score = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &best_score) == N4M_OK);
    N4M_TEST_REQUIRE(best_score < 0.25);  // 400 random draws over a width-10 range
    n4m_trial_status_t st;
    n4m_trial_get_status(best, &st);
    N4M_TEST_REQUIRE(st == N4M_TRIAL_COMPLETED);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

// Tier-A parity: the unscrambled Sobol sequence must be bit-identical to
// scipy.stats.qmc.Sobol(d=3, scramble=False). Three floats over [0,1) map the
// unit coordinate straight through, so trial values ARE the Sobol coordinates.
// Reference points (exact binary fractions) generated in the Python parity test.
void test_sobol_sequence_parity() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "a", 0.0, 1.0, 0.0, 0);
    n4m_search_space_add_float(sp, "b", 0.0, 1.0, 0.0, 0);
    n4m_search_space_add_float(sp, "c", 0.0, 1.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_SOBOL;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    const double expected[5][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.5},
        {0.75, 0.25, 0.25},
        {0.25, 0.75, 0.75},
        {0.375, 0.375, 0.625},
    };
    for (int i = 0; i < 5; ++i) {
        n4m_trial_t* t = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);
        double a = 0.0, b = 0.0, c = 0.0;
        n4m_trial_get_float(t, "a", &a);
        n4m_trial_get_float(t, "b", &b);
        n4m_trial_get_float(t, "c", &c);
        N4M_TEST_REQUIRE(a == expected[i][0]);  // exact: Sobol coords are dyadic
        N4M_TEST_REQUIRE(b == expected[i][1]);
        N4M_TEST_REQUIRE(c == expected[i][2]);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, a);
    }
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

double first_ask_x(uint64_t seed) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.seed = seed;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    n4m_trial_t* t = nullptr;
    n4m_optimizer_ask(opt, &t);
    double x = 0.0;
    n4m_trial_get_float(t, "x", &x);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
    return x;
}

void test_determinism() {
    N4M_TEST_REQUIRE(first_ask_x(42) == first_ask_x(42));  // same seed → identical draw
    N4M_TEST_REQUIRE(first_ask_x(1) != first_ask_x(2));    // different seed → different draw
}

void test_ask_batch() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 100, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 7;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    n4m_trial_t* trials[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    int32_t count = 0;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 5, trials, &count) == N4M_OK);
    N4M_TEST_REQUIRE(count == 5);
    for (int i = 0; i < 5; ++i) {
        int64_t id = -1;
        n4m_trial_get_id(trials[i], &id);
        N4M_TEST_REQUIRE(id == static_cast<int64_t>(i));  // distinct, monotonic ids
    }
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_finetune_estimator() {
    const int64_t n = 24;
    const int64_t p = 6;
    std::vector<double> X(static_cast<std::size_t>(n * p));
    std::vector<double> Y(static_cast<std::size_t>(n));
    std::uint64_t s = 999;
    auto rnd = [&]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11)) * (1.0 / 9007199254740992.0);
    };
    std::vector<double> beta(static_cast<std::size_t>(p));
    for (int64_t j = 0; j < p; ++j) beta[static_cast<std::size_t>(j)] = rnd();
    for (int64_t i = 0; i < n; ++i) {
        double y = 0.0;
        for (int64_t j = 0; j < p; ++j) {
            const double x = rnd();
            X[static_cast<std::size_t>(i * p + j)] = x;
            y += beta[static_cast<std::size_t>(j)] * x;
        }
        Y[static_cast<std::size_t>(i)] = y + 0.01 * rnd();
    }
    n4m_matrix_view_t Xv{};
    Xv.data = X.data();
    Xv.rows = n;
    Xv.cols = p;
    Xv.row_stride = p;
    Xv.col_stride = 1;
    Xv.dtype = N4M_DTYPE_F64;
    n4m_matrix_view_t Yv{};
    Yv.data = Y.data();
    Yv.rows = n;
    Yv.cols = 1;
    Yv.row_stride = 1;
    Yv.col_stride = 1;
    Yv.dtype = N4M_DTYPE_F64;

    n4m_validation_plan_t* plan = nullptr;
    N4M_TEST_REQUIRE(n4m_validation_plan_create(&plan) == N4M_OK);
    n4m_validation_plan_set_n_samples(plan, n);
    std::vector<int64_t> tr1, te1, tr2, te2;
    for (int64_t i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            te1.push_back(i);
            tr2.push_back(i);
        } else {
            tr1.push_back(i);
            te2.push_back(i);
        }
    }
    n4m_validation_plan_add_fold(plan, tr1.data(), static_cast<int64_t>(tr1.size()), te1.data(),
                                 static_cast<int64_t>(te1.size()));
    n4m_validation_plan_add_fold(plan, tr2.data(), static_cast<int64_t>(tr2.size()), te2.data(),
                                 static_cast<int64_t>(te2.size()));

    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "n_components", 1, 4, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 5;
    o.metric = N4M_METRIC_RMSE;
    o.direction = N4M_OPT_MINIMIZE;

    n4m_method_result_t* res = nullptr;
    const n4m_status_t st = n4m_finetune_estimator(ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &Yv, plan, sp,
                                                   &o, 8, &res);
    N4M_TEST_REQUIRE(st == N4M_OK);
    N4M_TEST_REQUIRE(res != nullptr);
    double best_score = -1.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "best_score", &best_score) == N4M_OK);
    N4M_TEST_REQUIRE(best_score >= 0.0 && best_score < 1e6);
    double best_nc = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "best.n_components", &best_nc) == N4M_OK);
    N4M_TEST_REQUIRE(best_nc >= 1.0 && best_nc <= 4.0);
    n4m_method_result_destroy(res);
    n4m_search_space_destroy(sp);
    n4m_validation_plan_destroy(plan);
    n4m_context_destroy(ctx);
}

void test_invalid_ranges() {
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    const double nan_v = std::nan("");
    N4M_TEST_REQUIRE(n4m_search_space_add_float(sp, "a", nan_v, 1.0, 0.0, 0) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_search_space_add_float(sp, "b", -1.0, 1.0, 0.0, 1) == N4M_ERR_INVALID_ARGUMENT);  // log, low<=0
    N4M_TEST_REQUIRE(n4m_search_space_add_int(sp, "c", 0, 10, 1, 1) == N4M_ERR_INVALID_ARGUMENT);          // log int, low<=0
    N4M_TEST_REQUIRE(n4m_search_space_add_float(sp, "d", 1.0, 0.0, 0.0, 0) == N4M_ERR_INVALID_ARGUMENT);   // high<low
    n4m_search_space_destroy(sp);
}

void test_space_validation_names_and_domains() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);

    {
        const n4m_sampler_kind_t samplers[] = {
            N4M_SAMPLER_RANDOM,  N4M_SAMPLER_TERNARY, N4M_SAMPLER_TPE,
            N4M_SAMPLER_SOBOL,   N4M_SAMPLER_GA,      N4M_SAMPLER_PSO,
            N4M_SAMPLER_CMAES,   N4M_SAMPLER_LHS,     N4M_SAMPLER_GP_EI,
        };
        for (const auto sampler : samplers) {
            n4m_search_space_t* empty = nullptr;
            n4m_search_space_create(&empty);
            N4M_TEST_REQUIRE(optimizer_create_status(ctx, empty, sampler) ==
                             N4M_ERR_INVALID_ARGUMENT);
            n4m_search_space_destroy(empty);
        }
    }

    auto int_status = [ctx](const char* name, int64_t low, int64_t high, int64_t step,
                            int32_t log) {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        N4M_TEST_REQUIRE(n4m_search_space_add_int(sp, name, low, high, step, log) == N4M_OK);
        const n4m_status_t status = optimizer_create_status(ctx, sp);
        n4m_search_space_destroy(sp);
        return status;
    };
    N4M_TEST_REQUIRE(int_status("", 1, 3, 1, 0) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(int_status("bad#name", 1, 3, 1, 0) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(int_status("zero_step", 1, 3, 0, 0) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(int_status("log_step", 1, 9, 2, 1) == N4M_ERR_INVALID_ARGUMENT);

    {  // duplicate ordered identifiers are ambiguous at trial lookup time
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "x", 1, 3, 1, 0);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // negative linear steps and stepped log-floats were silently treated as continuous
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.1, 1.0, -0.1, 0);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.1, 1.0, 0.1, 1);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // string categories must be non-empty and unique
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        const char* choices[2] = {"", "valid"};
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, choices, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
        n4m_search_space_create(&sp);
        const char* duplicate[2] = {"same", "same"};
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, duplicate, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // typed choices retain malformed payloads until final validation
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        const int32_t invalid_bool[2] = {0, 2};
        n4m_search_space_add_categorical(sp, "flag", N4M_CAT_BOOL, invalid_bool, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
        n4m_search_space_create(&sp);
        const double invalid_float[2] = {0.5, std::nan("")};
        n4m_search_space_add_categorical(sp, "weight", N4M_CAT_FLOAT, invalid_float, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // ordinal values are ordered choices, but still must be finite and unique
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        const double duplicate[2] = {1.0, 1.0};
        n4m_search_space_add_ordinal(sp, "rank", duplicate, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
        n4m_search_space_create(&sp);
        const double nonfinite[2] = {1.0, std::nan("")};
        n4m_search_space_add_ordinal(sp, "rank", nonfinite, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }

    n4m_context_destroy(ctx);
}

void test_numeric_choice_labels_and_int64_exactness() {
    constexpr int64_t max_exact = 9007199254740992LL;  // 2^53
    constexpr int64_t first_lossy = max_exact + 1;
    constexpr double first_double_outside_exact_range = 9007199254740994.0;

    {  // int64 inputs that the binary64 trial storage cannot round-trip are rejected
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        N4M_TEST_REQUIRE(
            n4m_search_space_add_int(sp, "lossy_bound", first_lossy, first_lossy, 1, 0)
            == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(
            n4m_search_space_add_int(sp, "lossy_step", 0, 1, first_lossy, 0)
            == N4M_ERR_INVALID_ARGUMENT);
        const int64_t lossy_choices[2] = {0, first_lossy};
        N4M_TEST_REQUIRE(n4m_search_space_add_categorical(
                             sp, "lossy_choice", N4M_CAT_INT, lossy_choices, 2)
                         == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(n4m_search_space_add_sorted_tuple(
                             sp, "lossy_tuple", 2, first_double_outside_exact_range,
                             first_double_outside_exact_range, 1)
                         == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(n4m_search_space_add_sorted_tuple(
                             sp, "huge_integer_tuple", 2, 1e300, 1e300, 1)
                         == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(n4m_search_space_add_sorted_tuple(
                             sp, "fractional_integer_tuple", 2, 1.5, 2.0, 1)
                         == N4M_ERR_INVALID_ARGUMENT);
        int32_t n_params = -1;
        N4M_TEST_REQUIRE(n4m_search_space_num_params(sp, &n_params) == N4M_OK);
        N4M_TEST_REQUIRE(n_params == 0);
        n4m_search_space_destroy(sp);
    }

    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    N4M_TEST_REQUIRE(
        n4m_search_space_add_int(sp, "exact_int", max_exact, max_exact, 1, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_sorted_tuple(
                         sp, "exact_tuple", 2, static_cast<double>(max_exact),
                         static_cast<double>(max_exact), 1)
                     == N4M_OK);

    // These pairs collapsed to the same six-decimal std::to_string label.
    const double close_values[2] = {1.0000001, 1.0000002};
    N4M_TEST_REQUIRE(n4m_search_space_add_categorical(
                         sp, "close_cat", N4M_CAT_FLOAT, close_values, 2)
                     == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_ordinal(sp, "close_ord", close_values, 2) == N4M_OK);

    n4m_optimizer_options_t o = default_opts();
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);

    const char* forced_names[2] = {"close_cat", "close_ord"};
    const double forced_zero[2] = {0.0, 0.0};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, forced_names, forced_zero, 2) == N4M_OK);
    n4m_trial_t* first = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &first) == N4M_OK);
    int64_t exact_value = 0;
    N4M_TEST_REQUIRE(n4m_trial_get_int(first, "exact_int", &exact_value) == N4M_OK);
    N4M_TEST_REQUIRE(exact_value == max_exact);
    N4M_TEST_REQUIRE(n4m_trial_get_int(first, "exact_tuple#0", &exact_value) == N4M_OK);
    N4M_TEST_REQUIRE(exact_value == max_exact);
    N4M_TEST_REQUIRE(n4m_trial_get_int(first, "exact_tuple#1", &exact_value) == N4M_OK);
    N4M_TEST_REQUIRE(exact_value == max_exact);
    const char* cat_label_ptr = nullptr;
    const char* ord_label_ptr = nullptr;
    N4M_TEST_REQUIRE(
        n4m_trial_get_category(first, "close_cat", nullptr, &cat_label_ptr) == N4M_OK);
    N4M_TEST_REQUIRE(
        n4m_trial_get_category(first, "close_ord", nullptr, &ord_label_ptr) == N4M_OK);
    const std::string cat_label_zero = cat_label_ptr != nullptr ? cat_label_ptr : "";
    const std::string ord_label_zero = ord_label_ptr != nullptr ? ord_label_ptr : "";
    double cat_value = 0.0;
    double ord_value = 0.0;
    n4m_trial_get_float(first, "close_cat", &cat_value);
    n4m_trial_get_float(first, "close_ord", &ord_value);
    N4M_TEST_REQUIRE(cat_value == close_values[0]);
    N4M_TEST_REQUIRE(ord_value == close_values[0]);

    const double forced_one[2] = {1.0, 1.0};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, forced_names, forced_one, 2) == N4M_OK);
    n4m_trial_t* second = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &second) == N4M_OK);
    N4M_TEST_REQUIRE(
        n4m_trial_get_category(second, "close_cat", nullptr, &cat_label_ptr) == N4M_OK);
    N4M_TEST_REQUIRE(
        n4m_trial_get_category(second, "close_ord", nullptr, &ord_label_ptr) == N4M_OK);
    const std::string cat_label_one = cat_label_ptr != nullptr ? cat_label_ptr : "";
    const std::string ord_label_one = ord_label_ptr != nullptr ? ord_label_ptr : "";
    n4m_trial_get_float(second, "close_cat", &cat_value);
    n4m_trial_get_float(second, "close_ord", &ord_value);
    N4M_TEST_REQUIRE(cat_value == close_values[1]);
    N4M_TEST_REQUIRE(ord_value == close_values[1]);
    N4M_TEST_REQUIRE(!cat_label_zero.empty() && !cat_label_one.empty());
    N4M_TEST_REQUIRE(!ord_label_zero.empty() && !ord_label_one.empty());
    N4M_TEST_REQUIRE(cat_label_zero == "1.0000001000000001");
    N4M_TEST_REQUIRE(cat_label_one == "1.0000001999999999");
    N4M_TEST_REQUIRE(ord_label_zero == cat_label_zero);
    N4M_TEST_REQUIRE(ord_label_one == cat_label_one);
    N4M_TEST_REQUIRE(cat_label_zero != cat_label_one);
    N4M_TEST_REQUIRE(ord_label_zero != ord_label_one);

    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_binary64_choice_label_vectors() {
    struct LabelVector {
        double value;
        const char* expected;
    };
    const LabelVector vectors[] = {
        {0.0, "0"},
        {-0.0, "-0"},
        {std::numeric_limits<double>::denorm_min(), "4.9406564584124654e-324"},
        {std::nextafter(std::numeric_limits<double>::min(), 0.0),
         "2.2250738585072009e-308"},
        {std::numeric_limits<double>::min(), "2.2250738585072014e-308"},
        {1e-5, "1.0000000000000001e-05"},
        {9007199254740992.0, "9007199254740992"},
        {-9007199254740992.0, "-9007199254740992"},
        {std::numeric_limits<double>::max(), "1.7976931348623157e+308"},
    };

    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    for (const auto& vector : vectors) {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        N4M_TEST_REQUIRE(n4m_search_space_add_categorical(
                             sp, "value", N4M_CAT_FLOAT, &vector.value, 1) == N4M_OK);
        n4m_optimizer_options_t o = default_opts();
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        n4m_trial_t* trial = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &trial) == N4M_OK);
        const char* label = nullptr;
        N4M_TEST_REQUIRE(
            n4m_trial_get_category(trial, "value", nullptr, &label) == N4M_OK);
        N4M_TEST_REQUIRE(label != nullptr && std::string(label) == vector.expected);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }

    n4m_search_space_t* duplicate_zero = nullptr;
    n4m_search_space_create(&duplicate_zero);
    const double signed_zeros[2] = {-0.0, 0.0};
    n4m_search_space_add_categorical(
        duplicate_zero, "zero", N4M_CAT_FLOAT, signed_zeros, 2);
    N4M_TEST_REQUIRE(optimizer_create_status(ctx, duplicate_zero) ==
                     N4M_ERR_INVALID_ARGUMENT);
    n4m_search_space_destroy(duplicate_zero);
    n4m_context_destroy(ctx);
}

void test_constraint_validation() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);

    {  // hard-constraint arity
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        const char* refs[1] = {"x"};
        N4M_TEST_REQUIRE(
            n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_REQUIRES, refs, nullptr, 1)
            == N4M_OK);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // unknown/duplicate refs and unknown constraint kinds
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        const char* unknown[2] = {"x", "missing"};
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_EXCLUDE, unknown, nullptr, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        const char* repeated[2] = {"x", "x"};
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_EXCLUDE, repeated, nullptr, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_float(sp, "y", 0.0, 1.0, 0.0, 0);
#if !defined(N4M_UBSAN_BUILD)
        const char* refs[2] = {"x", "y"};
        // The C ABI must reject unknown enum values, but constructing an
        // out-of-range enum-by-value argument is itself undefined in a C++ UBSAN
        // build. Struct-field enum validation is still covered under UBSAN.
        n4m_search_space_add_constraint(
            sp, invalid_enum<n4m_constraint_kind_t>(99), refs, nullptr, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
#endif
        n4m_search_space_destroy(sp);
    }
    {  // labels must resolve on categorical/ordinal references
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        const char* modes[2] = {"a", "b"};
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, modes, 2);
        const char* refs[2] = {"x", "mode"};
        const char* labels[2] = {"", "missing"};
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_EXCLUDE, refs, labels, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_float(sp, "y", 0.0, 1.0, 0.0, 0);
        const char* numeric_refs[2] = {"x", "y"};
        const char* numeric_labels[2] = {"not-a-choice", ""};
        n4m_search_space_add_constraint(
            sp, N4M_CONSTRAINT_EXCLUDE, numeric_refs, numeric_labels, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // A malformed label is INVALID_ARGUMENT before the otherwise valid
       // sorted-tuple hard-constraint pair is classified UNSUPPORTED.
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_sorted_tuple(sp, "tuple", 2, 0.0, 1.0, 0);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        const char* refs[2] = {"tuple", "x"};
        const char* malformed_labels[2] = {"not-a-choice", nullptr};
        n4m_search_space_add_constraint(
            sp, N4M_CONSTRAINT_EXCLUDE, refs, malformed_labels, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        n4m_search_space_add_sorted_tuple(sp, "tuple", 2, 0.0, 1.0, 0);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_EXCLUDE, refs, nullptr, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_UNSUPPORTED);
        n4m_search_space_destroy(sp);
    }
    {  // malformed conditions, unknown refs, and bad labels finalize at create time
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "child", 0.0, 1.0, 0.0, 0);
        const char* short_refs[1] = {"child"};
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_CONDITION_IN, short_refs, nullptr, 1)
                         == N4M_OK);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "child", 0.0, 1.0, 0.0, 0);
        const char* refs[2] = {"child", "missing_parent"};
        const char* labels[2] = {"", "on"};
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_CONDITION_IN, refs, labels, 2)
                         == N4M_OK);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        const char* parent_choices[2] = {"off", "on"};
        n4m_search_space_add_categorical(
            sp, "parent", N4M_CAT_STR, parent_choices, 2);
        const char* missing_child_refs[2] = {"missing_child", "parent"};
        const char* missing_child_labels[2] = {"", "on"};
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_CONDITION_IN, missing_child_refs,
                             missing_child_labels, 2)
                         == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(
            n4m_search_space_add_float(sp, "missing_child", 0.0, 1.0, 0.0, 0) == N4M_OK);
        // The rejected condition was never retained as an inert constraint.
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_OK);
        n4m_search_space_destroy(sp);

        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "child", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_categorical(
            sp, "parent", N4M_CAT_STR, parent_choices, 2);
        const char* bad_label_refs[2] = {"child", "parent"};
        const char* bad_labels[2] = {"", "missing"};
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_CONDITION_IN, bad_label_refs,
                             bad_labels, 2)
                         == N4M_OK);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }

    n4m_context_destroy(ctx);
}

void test_constraint_reference_identity() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    const char* choices[2] = {"a", "b"};

    {  // Same axis, distinct labels are distinct canonical references.
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, choices, 2);
        const char* refs[2] = {"mode", "mode"};
        const char* labels[2] = {"a", "b"};
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_EXCLUDE, refs, labels, 2) == N4M_OK);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_OK);
        n4m_search_space_destroy(sp);
    }
    {  // An exact duplicate (axis, label) remains invalid.
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, choices, 2);
        const char* refs[2] = {"mode", "mode"};
        const char* labels[2] = {"a", "a"};
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_EXCLUDE, refs, labels, 2);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }
    {  // Bare presence and a labelled atom on the same axis are distinct and
       // the requires predicate filters every proposal to the labelled value.
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, choices, 2);
        const char* refs[2] = {"mode", "mode"};
        const char* labels[2] = {nullptr, "b"};
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_REQUIRES, refs, labels, 2);
        n4m_optimizer_options_t o = default_opts();
        o.seed = 17;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        for (int i = 0; i < 32; ++i) {
            n4m_trial_t* trial = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &trial) == N4M_OK);
            const char* label = nullptr;
            N4M_TEST_REQUIRE(
                n4m_trial_get_category(trial, "mode", nullptr, &label) == N4M_OK);
            N4M_TEST_REQUIRE(label != nullptr && std::string(label) == "b");
            int64_t id = 0;
            n4m_trial_get_id(trial, &id);
            N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, id, 0.0) == N4M_OK);
        }
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }
    n4m_context_destroy(ctx);
}

void test_condition_cycles_rejected() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    const char* choices[2] = {"on", "off"};
    n4m_search_space_add_categorical(sp, "a", N4M_CAT_STR, choices, 2);
    n4m_search_space_add_categorical(sp, "b", N4M_CAT_STR, choices, 2);
    const char* a_refs[2] = {"a", "b"};
    const char* a_labels[2] = {"", "on"};
    const char* b_refs[2] = {"b", "a"};
    const char* b_labels[2] = {"", "on"};
    N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                         sp, N4M_CONSTRAINT_CONDITION_IN, a_refs, a_labels, 2)
                     == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                         sp, N4M_CONSTRAINT_CONDITION_IN, b_refs, b_labels, 2)
                     == N4M_OK);
    N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_sampler_hard_constraint_matrix() {
    // MT11 — compatibility is uniform enforce-or-refuse. For each hard kind
    // (mutex_group / requires / exclude) across all nine samplers: random, LHS,
    // ternary and TPE create N4M_OK and enforce it by bounded rejection; Sobol,
    // GA, PSO, CMA-ES and GP-EI refuse at create with N4M_ERR_UNSUPPORTED. A
    // hard constraint over a sorted_tuple root is refused for EVERY sampler.
    // condition_in / condition_not_in are activation rules and create N4M_OK for
    // every sampler; a tuple may be a conditional child (prefix-deactivated) but
    // a labelled tuple parent is a validation error.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);

    const n4m_sampler_kind_t all_samplers[9] = {
        N4M_SAMPLER_RANDOM, N4M_SAMPLER_SOBOL, N4M_SAMPLER_LHS,
        N4M_SAMPLER_TERNARY, N4M_SAMPLER_GA, N4M_SAMPLER_PSO,
        N4M_SAMPLER_CMAES, N4M_SAMPLER_TPE, N4M_SAMPLER_GP_EI};
    auto is_rejection_sampler = [](n4m_sampler_kind_t s) {
        return s == N4M_SAMPLER_RANDOM || s == N4M_SAMPLER_LHS ||
               s == N4M_SAMPLER_TERNARY || s == N4M_SAMPLER_TPE;
    };
    const n4m_constraint_kind_t hard_kinds[3] = {
        N4M_CONSTRAINT_MUTEX_GROUP, N4M_CONSTRAINT_REQUIRES, N4M_CONSTRAINT_EXCLUDE};
    const char* choices[2] = {"off", "on"};

    // ---- flat hard constraints x all three kinds x all nine samplers ----
    for (const auto kind : hard_kinds) {
        for (const auto sampler : all_samplers) {
            n4m_search_space_t* sp = nullptr;
            n4m_search_space_create(&sp);
            n4m_search_space_add_categorical(sp, "a", N4M_CAT_STR, choices, 2);
            n4m_search_space_add_categorical(sp, "b", N4M_CAT_STR, choices, 2);
            const char* refs[2] = {"a", "b"};
            const char* labels[2] = {"on", "on"};
            N4M_TEST_REQUIRE(
                n4m_search_space_add_constraint(sp, kind, refs, labels, 2) == N4M_OK);
            const n4m_status_t create = optimizer_create_status(ctx, sp, sampler);
            N4M_TEST_REQUIRE(create ==
                             (is_rejection_sampler(sampler) ? N4M_OK : N4M_ERR_UNSUPPORTED));
            n4m_search_space_destroy(sp);
        }
    }

    // Emitted asks from the rejection samplers actually satisfy each hard kind.
    const n4m_sampler_kind_t rejection_samplers[4] = {
        N4M_SAMPLER_RANDOM, N4M_SAMPLER_LHS, N4M_SAMPLER_TERNARY, N4M_SAMPLER_TPE};
    for (const auto kind : hard_kinds) {
        for (const auto sampler : rejection_samplers) {
            n4m_search_space_t* sp = nullptr;
            n4m_search_space_create(&sp);
            n4m_search_space_add_categorical(sp, "a", N4M_CAT_STR, choices, 2);
            n4m_search_space_add_categorical(sp, "b", N4M_CAT_STR, choices, 2);
            const char* refs[2] = {"a", "b"};
            const char* labels[2] = {"on", "on"};
            n4m_search_space_add_constraint(sp, kind, refs, labels, 2);
            n4m_optimizer_options_t o = default_opts();
            o.sampler = sampler;
            o.seed = 5;
            n4m_optimizer_t* opt = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
            for (int i = 0; i < 24; ++i) {
                n4m_trial_t* t = nullptr;
                N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);
                const char* la = nullptr;
                const char* lb = nullptr;
                n4m_trial_get_category(t, "a", nullptr, &la);
                n4m_trial_get_category(t, "b", nullptr, &lb);
                const bool a_on = la != nullptr && std::string(la) == "on";
                const bool b_on = lb != nullptr && std::string(lb) == "on";
                if (kind == N4M_CONSTRAINT_REQUIRES) {
                    N4M_TEST_REQUIRE(!(a_on && !b_on));  // a=on requires b=on
                } else {
                    N4M_TEST_REQUIRE(!(a_on && b_on));   // mutex/exclude: not both on
                }
                int64_t id = 0;
                n4m_trial_get_id(t, &id);
                n4m_optimizer_tell(opt, id, 0.0);
            }
            n4m_optimizer_destroy(opt);
            n4m_search_space_destroy(sp);
        }
    }

    // An intrinsically unsatisfiable hard space must stop after the bounded
    // 200-attempt rejection budget, rather than hang or emit an invalid trial.
    // Bare references are always present, so EXCLUDE(a, b) can never hold.
    for (const auto sampler : rejection_samplers) {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "a", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_float(sp, "b", 0.0, 1.0, 0.0, 0);
        const char* refs[2] = {"a", "b"};
        const char* labels[2] = {"", ""};
        N4M_TEST_REQUIRE(
            n4m_search_space_add_constraint(
                sp, N4M_CONSTRAINT_EXCLUDE, refs, labels, 2) == N4M_OK);
        n4m_optimizer_options_t o = default_opts();
        o.sampler = sampler;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        n4m_trial_t* trial = reinterpret_cast<n4m_trial_t*>(std::uintptr_t{1});
        N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &trial) == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(trial == nullptr);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }

    // ---- tuple-hard x all three kinds x all nine: always UNSUPPORTED ----
    // A tuple root has no trial-level presence record (only knots#i), so no
    // sampler may reference it in a hard constraint.
    for (const auto kind : hard_kinds) {
        for (const auto sampler : all_samplers) {
            n4m_search_space_t* sp = nullptr;
            n4m_search_space_create(&sp);
            n4m_search_space_add_sorted_tuple(sp, "knots", 2, 0.0, 1.0, 0);
            n4m_search_space_add_categorical(sp, "flag", N4M_CAT_STR, choices, 2);
            const char* refs[2] = {"knots", "flag"};
            const char* labels[2] = {"", "on"};
            N4M_TEST_REQUIRE(
                n4m_search_space_add_constraint(sp, kind, refs, labels, 2) == N4M_OK);
            N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp, sampler) ==
                             N4M_ERR_UNSUPPORTED);
            n4m_search_space_destroy(sp);
        }
    }

    // ---- condition_in / condition_not_in x all nine: OK + activation ----
    const n4m_constraint_kind_t cond_kinds[2] = {
        N4M_CONSTRAINT_CONDITION_IN, N4M_CONSTRAINT_CONDITION_NOT_IN};
    for (const auto kind : cond_kinds) {
        for (const auto sampler : all_samplers) {
            n4m_search_space_t* sp = nullptr;
            n4m_search_space_create(&sp);
            n4m_search_space_add_float(sp, "child", 0.0, 1.0, 0.0, 0);
            n4m_search_space_add_categorical(sp, "parent", N4M_CAT_STR, choices, 2);
            const char* refs[2] = {"child", "parent"};
            const char* labels[2] = {"", "on"};
            N4M_TEST_REQUIRE(
                n4m_search_space_add_constraint(sp, kind, refs, labels, 2) == N4M_OK);
            n4m_optimizer_options_t o = default_opts();
            o.sampler = sampler;
            o.seed = 11;
            n4m_optimizer_t* opt = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
            bool saw_active = false;
            bool saw_inactive = false;
            for (int i = 0; i < 64; ++i) {
                n4m_trial_t* t = nullptr;
                N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);
                const char* pl = nullptr;
                int32_t child_active = -1;
                n4m_trial_get_category(t, "parent", nullptr, &pl);
                N4M_TEST_REQUIRE(n4m_trial_is_active(t, "child", &child_active) == N4M_OK);
                const bool parent_on = pl != nullptr && std::string(pl) == "on";
                const bool expect_active =
                    (kind == N4M_CONSTRAINT_CONDITION_IN) ? parent_on : !parent_on;
                N4M_TEST_REQUIRE(child_active == (expect_active ? 1 : 0));
                saw_active = saw_active || expect_active;
                saw_inactive = saw_inactive || !expect_active;
                int64_t id = 0;
                n4m_trial_get_id(t, &id);
                n4m_optimizer_tell(opt, id, parent_on ? 0.0 : 1.0);
            }
            N4M_TEST_REQUIRE(saw_active && saw_inactive);
            n4m_optimizer_destroy(opt);
            n4m_search_space_destroy(sp);
        }
    }

    // ---- tuple as a conditional CHILD: prefix-deactivation semantics ----
    {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_sorted_tuple(sp, "knots", 2, 0.0, 1.0, 0);
        n4m_search_space_add_categorical(sp, "gate", N4M_CAT_STR, choices, 2);
        const char* refs[2] = {"knots", "gate"};
        const char* labels[2] = {"", "on"};
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_CONDITION_IN, refs, labels, 2) == N4M_OK);
        n4m_optimizer_options_t o = default_opts();
        o.seed = 13;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        bool saw_active = false;
        bool saw_inactive = false;
        for (int i = 0; i < 48; ++i) {
            n4m_trial_t* t = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);
            const char* gl = nullptr;
            int32_t k0 = -1;
            int32_t k1 = -1;
            n4m_trial_get_category(t, "gate", nullptr, &gl);
            N4M_TEST_REQUIRE(n4m_trial_is_active(t, "knots#0", &k0) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_trial_is_active(t, "knots#1", &k1) == N4M_OK);
            const bool gate_on = gl != nullptr && std::string(gl) == "on";
            N4M_TEST_REQUIRE(k0 == (gate_on ? 1 : 0));  // whole tuple prefix follows the gate
            N4M_TEST_REQUIRE(k1 == (gate_on ? 1 : 0));
            saw_active = saw_active || gate_on;
            saw_inactive = saw_inactive || !gate_on;
            int64_t id = 0;
            n4m_trial_get_id(t, &id);
            n4m_optimizer_tell(opt, id, 0.0);
        }
        N4M_TEST_REQUIRE(saw_active && saw_inactive);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }

    // ---- tuple as a labelled condition PARENT: validation error ---------
    {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "child", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_sorted_tuple(sp, "knots", 2, 0.0, 1.0, 0);
        const char* refs[2] = {"child", "knots"};
        const char* labels[2] = {"", "0.5"};  // a labelled non-categorical parent
        N4M_TEST_REQUIRE(n4m_search_space_add_constraint(
                             sp, N4M_CONSTRAINT_CONDITION_IN, refs, labels, 2) == N4M_OK);
        N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_INVALID_ARGUMENT);
        n4m_search_space_destroy(sp);
    }

    n4m_context_destroy(ctx);
}

void test_struct_size_guard() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.struct_size = 0;  // caller forgot to init
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_ERR_INVALID_ARGUMENT);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_enqueue_warm_start() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 9;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    // unknown param rejected
    const char* bad[1] = {"nope"};
    const double bv[1] = {1.0};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, bad, bv, 1) == N4M_ERR_INVALID_ARGUMENT);
    // valid warm-start forces the next ask
    const char* names[1] = {"x"};
    const double vals[1] = {3.14};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, vals, 1) == N4M_OK);
    n4m_trial_t* t = nullptr;
    n4m_optimizer_ask(opt, &t);
    double x = 0.0;
    n4m_trial_get_float(t, "x", &x);
    N4M_TEST_REQUIRE(x == 3.14);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_conditional_activation() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    const char* kernels[2] = {"linear", "rbf"};
    n4m_search_space_add_categorical(sp, "kernel", N4M_CAT_STR, kernels, 2);
    n4m_search_space_add_float(sp, "gamma", 1e-3, 1e0, 0.0, 1);
    // gamma active only when kernel == "rbf"
    const char* refs[2] = {"gamma", "kernel"};
    const char* labs[2] = {"", "rbf"};
    N4M_TEST_REQUIRE(
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, refs, labs, 2) == N4M_OK);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 3;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    for (int i = 0; i < 20; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        int32_t kidx = -1;
        const char* klabel = nullptr;
        n4m_trial_get_category(t, "kernel", &kidx, &klabel);
        int32_t gamma_active = -1;
        n4m_trial_is_active(t, "gamma", &gamma_active);
        const bool is_rbf = (klabel != nullptr && std::string(klabel) == "rbf");
        N4M_TEST_REQUIRE(gamma_active == (is_rbf ? 1 : 0));
    }
    // A second condition with a different parent is centrally rejected at create.
    const char* other_choices[2] = {"x", "y"};
    n4m_search_space_add_categorical(sp, "other", N4M_CAT_STR, other_choices, 2);
    const char* refs2[2] = {"gamma", "other"};
    const char* labs2[2] = {"", "x"};
    N4M_TEST_REQUIRE(
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, refs2, labs2, 2)
        == N4M_OK);
    N4M_TEST_REQUIRE(optimizer_create_status(ctx, sp) == N4M_ERR_UNSUPPORTED);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_conditional_deep_nesting() {
    // Three-level chain (E2): a∈{p,q}; b active iff a=="p"; c active iff b=="x".
    // The grandchild c must be INACTIVE whenever a!="p", even when b's stale
    // sampled label is "x" — the case a single label-only pass gets wrong. This
    // is the correctness the nested sub-pipeline search space depends on.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    const char* aa[2] = {"p", "q"};
    const char* bb[2] = {"x", "y"};
    n4m_search_space_add_categorical(sp, "a", N4M_CAT_STR, aa, 2);
    n4m_search_space_add_categorical(sp, "b", N4M_CAT_STR, bb, 2);
    n4m_search_space_add_float(sp, "c", 0.0, 1.0, 0.0, 0);
    const char* rb[2] = {"b", "a"};
    const char* lb[2] = {"", "p"};  // b active iff a == "p"
    N4M_TEST_REQUIRE(n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, rb, lb, 2) == N4M_OK);
    const char* rc[2] = {"c", "b"};
    const char* lc[2] = {"", "x"};  // c active iff b == "x"
    N4M_TEST_REQUIRE(n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, rc, lc, 2) == N4M_OK);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 4;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    bool saw_q_with_b_x = false, saw_active_c = false;
    for (int i = 0; i < 200; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        int32_t ai = 0, bi = 0, bact = 0, cact = 0;
        const char* al = nullptr;
        const char* bl = nullptr;
        n4m_trial_get_category(t, "a", &ai, &al);
        n4m_trial_get_category(t, "b", &bi, &bl);
        n4m_trial_is_active(t, "b", &bact);
        n4m_trial_is_active(t, "c", &cact);
        const bool a_p = (al != nullptr && std::string(al) == "p");
        const bool b_x = (bl != nullptr && std::string(bl) == "x");
        if (!a_p) {
            N4M_TEST_REQUIRE(bact == 0);
            N4M_TEST_REQUIRE(cact == 0);  // grandchild dead when branch is dead
            if (b_x) saw_q_with_b_x = true;  // the exact path the old code broke
        } else {
            N4M_TEST_REQUIRE(bact == 1);
            N4M_TEST_REQUIRE(cact == (b_x ? 1 : 0));
            if (b_x) saw_active_c = true;
        }
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, 0.0);
    }
    N4M_TEST_REQUIRE(saw_q_with_b_x);  // the bug path was actually exercised
    N4M_TEST_REQUIRE(saw_active_c);    // c does activate on the live branch
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_conditional_child_before_parent() {
    // The ordered-space contract allows a conditional child to be declared before
    // its parent. Validation resolves the complete graph at optimizer creation.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "model_pls_nc", 1, 20, 1, 0);  // child declared FIRST
    const char* refs[2] = {"model_pls_nc", "model"};
    const char* labs[2] = {"", "pls"};
    // The child exists, so its condition can be bound before the parent axis.
    N4M_TEST_REQUIRE(
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, refs, labs, 2) == N4M_OK);
    const char* models[2] = {"pls", "ridge"};
    n4m_search_space_add_categorical(sp, "model", N4M_CAT_STR, models, 2);  // parent AFTER
    n4m_optimizer_options_t o = default_opts();
    o.seed = 2;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    bool saw_pls = false, saw_ridge = false;
    for (int i = 0; i < 100; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        int32_t mi = 0, act = -1;
        const char* ml = nullptr;
        n4m_trial_get_category(t, "model", &mi, &ml);
        n4m_trial_is_active(t, "model_pls_nc", &act);
        if (ml != nullptr && std::string(ml) == "ridge") {
            N4M_TEST_REQUIRE(act == 0);
            saw_ridge = true;
        } else {
            N4M_TEST_REQUIRE(act == 1);
            saw_pls = true;
        }
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, 0.0);
    }
    N4M_TEST_REQUIRE(saw_pls);
    N4M_TEST_REQUIRE(saw_ridge);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_finetune_rejects_unsupported_param() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "n_components", 1, 4, 1, 0);
    n4m_search_space_add_float(sp, "unsupported", 0.0, 1.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    // tiny valid X/Y/plan (loop rejects before CV, so contents are irrelevant)
    double xd[4] = {1.0, 2.0, 3.0, 4.0};
    double yd[2] = {1.0, 2.0};
    n4m_matrix_view_t Xv{};
    Xv.data = xd; Xv.rows = 2; Xv.cols = 2; Xv.row_stride = 2; Xv.col_stride = 1; Xv.dtype = N4M_DTYPE_F64;
    n4m_matrix_view_t Yv{};
    Yv.data = yd; Yv.rows = 2; Yv.cols = 1; Yv.row_stride = 1; Yv.col_stride = 1; Yv.dtype = N4M_DTYPE_F64;
    n4m_validation_plan_t* plan = nullptr;
    n4m_validation_plan_create(&plan);
    n4m_validation_plan_set_n_samples(plan, 2);
    int64_t tr[1] = {0}, te[1] = {1};
    n4m_validation_plan_add_fold(plan, tr, 1, te, 1);
    n4m_method_result_t* res = nullptr;
    N4M_TEST_REQUIRE(
        n4m_finetune_estimator(ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &Yv, plan, sp, &o, 4, &res)
        == N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(res == nullptr);
    n4m_validation_plan_destroy(plan);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_finetune_preflight_contract() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    double xd[4] = {1.0, 2.0, 3.0, 4.0};
    double yd[2] = {1.0, 2.0};
    n4m_matrix_view_t Xv{};
    Xv.data = xd;
    Xv.rows = 2;
    Xv.cols = 2;
    Xv.row_stride = 2;
    Xv.col_stride = 1;
    Xv.dtype = N4M_DTYPE_F64;
    n4m_matrix_view_t Yv{};
    Yv.data = yd;
    Yv.rows = 2;
    Yv.cols = 1;
    Yv.row_stride = 1;
    Yv.col_stride = 1;
    Yv.dtype = N4M_DTYPE_F64;
    n4m_validation_plan_t* plan = nullptr;
    n4m_validation_plan_create(&plan);
    n4m_validation_plan_set_n_samples(plan, 2);
    int64_t tr[1] = {0};
    int64_t te[1] = {1};
    n4m_validation_plan_add_fold(plan, tr, 1, te, 1);

    auto call = [&](n4m_algorithm_t estimator, n4m_search_space_t* space,
                    const n4m_optimizer_options_t& options) {
        n4m_method_result_t* result = nullptr;
        const n4m_status_t status = n4m_finetune_estimator(
            ctx, estimator, &Xv, &Yv, plan, space, &options, 2, &result);
        if (result != nullptr) n4m_method_result_destroy(result);
        return status;
    };

    n4m_search_space_t* valid = nullptr;
    n4m_search_space_create(&valid);
    n4m_search_space_add_int(valid, "n_components", 1, 2, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    // PCR is an eligible generic regression route in MT8, so the estimator gate
    // no longer rejects it; a genuinely ineligible estimator (classification
    // chassis) is still UNSUPPORTED before any study is created.
    N4M_TEST_REQUIRE(call(N4M_ALGO_PLS_DA, valid, o) == N4M_ERR_UNSUPPORTED);
    o.pruner = N4M_PRUNER_MEDIAN;
    N4M_TEST_REQUIRE(call(N4M_ALGO_PLS_REGRESSION, valid, o) == N4M_ERR_UNSUPPORTED);
    o = default_opts();
    o.metric = N4M_METRIC_ACCURACY;
    N4M_TEST_REQUIRE(call(N4M_ALGO_PLS_REGRESSION, valid, o) == N4M_ERR_NOT_IMPLEMENTED);
    n4m_search_space_destroy(valid);

    n4m_search_space_t* wrong_kind = nullptr;
    n4m_search_space_create(&wrong_kind);
    n4m_search_space_add_float(wrong_kind, "n_components", 1.0, 2.0, 0.0, 0);
    o = default_opts();
    N4M_TEST_REQUIRE(
        call(N4M_ALGO_PLS_REGRESSION, wrong_kind, o) == N4M_ERR_UNSUPPORTED);
    n4m_search_space_destroy(wrong_kind);

    n4m_search_space_t* zero_low = nullptr;
    n4m_search_space_create(&zero_low);
    n4m_search_space_add_int(zero_low, "n_components", 0, 2, 1, 0);
    N4M_TEST_REQUIRE(call(N4M_ALGO_PLS_REGRESSION, zero_low, o) == N4M_ERR_UNSUPPORTED);
    n4m_search_space_destroy(zero_low);

    n4m_search_space_t* log_int = nullptr;
    n4m_search_space_create(&log_int);
    n4m_search_space_add_int(log_int, "n_components", 1, 8, 1, 1);
    N4M_TEST_REQUIRE(call(N4M_ALGO_PLS_REGRESSION, log_int, o) == N4M_ERR_UNSUPPORTED);
    n4m_search_space_destroy(log_int);

    n4m_validation_plan_destroy(plan);
    n4m_context_destroy(ctx);
}

void test_finetune_rejects_global_input_errors_before_study() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "n_components", 1, 1, 1, 0);
    n4m_optimizer_options_t o = default_opts();

    double xd[8] = {1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0};
    double yd[4] = {1.0, 2.0, 3.0, 4.0};
    n4m_matrix_view_t Xv{};
    Xv.data = xd;
    Xv.rows = 4;
    Xv.cols = 2;
    Xv.row_stride = 2;
    Xv.col_stride = 1;
    Xv.dtype = N4M_DTYPE_F64;
    n4m_matrix_view_t Yv{};
    Yv.data = yd;
    Yv.rows = 4;
    Yv.cols = 1;
    Yv.row_stride = 1;
    Yv.col_stride = 1;
    Yv.dtype = N4M_DTYPE_F64;

    n4m_validation_plan_t* plan = nullptr;
    n4m_validation_plan_create(&plan);
    n4m_validation_plan_set_n_samples(plan, 4);
    int64_t train[2] = {0, 1};
    int64_t test[2] = {2, 3};
    n4m_validation_plan_add_fold(plan, train, 2, test, 2);
    int64_t train_second[2] = {2, 3};
    int64_t test_second[2] = {0, 1};
    n4m_validation_plan_add_fold(plan, train_second, 2, test_second, 2);

    n4m_method_result_t* result = nullptr;
    n4m_matrix_view_t short_y = Yv;
    short_y.rows = 3;
    // A near-zero timeout proves global matrix/plan validation runs before the
    // optimizer is created or asked: the structural error wins over CANCELLED.
    o.timeout_seconds = 1e-12;
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &short_y, plan, sp, &o, 2,
                         &result) == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(result == nullptr);

    float f32_yd[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    n4m_matrix_view_t f32_y = Yv;
    f32_y.data = f32_yd;
    f32_y.dtype = N4M_DTYPE_F32;
    o.timeout_seconds = 0.0;
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &f32_y, plan, sp, &o, 1,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    n4m_method_result_destroy(result);
    result = nullptr;
    o.timeout_seconds = 1e-12;

    xd[0] = std::numeric_limits<double>::quiet_NaN();
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &Yv, plan, sp, &o, 2,
                         &result) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    const char* error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(error != nullptr && std::string(error).find("NaN or Inf") != std::string::npos);
    xd[0] = 1.0;

    yd[0] = std::numeric_limits<double>::infinity();
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &Yv, plan, sp, &o, 2,
                         &result) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(error != nullptr && std::string(error).find("NaN or Inf") != std::string::npos);
    yd[0] = 1.0;

    n4m_matrix_view_t too_wide = Xv;
    too_wide.cols = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    too_wide.row_stride = too_wide.cols;
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &too_wide, &Yv, plan, sp, &o, 2,
                         &result) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);

    n4m_validation_plan_t* invalid_plan = nullptr;
    n4m_validation_plan_create(&invalid_plan);
    n4m_validation_plan_set_n_samples(invalid_plan, 4);
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &Xv, &Yv, invalid_plan, sp, &o, 2,
                         &result) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(error != nullptr && std::string(error).find("fold") != std::string::npos);

    n4m_validation_plan_destroy(invalid_plan);
    n4m_validation_plan_destroy(plan);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

// Deterministic regression dataset + a 2-fold (even/odd) plan shared by the MT8
// finetune-registry tests. 12 training rows x 6 features per fold is enough rank
// for every dense route; the response is a noisy linear combination of all six
// features so sparsity actually changes the fit.
struct FinetuneFixture {
    std::vector<double> X;
    std::vector<double> Y;
    int64_t n{24};
    int64_t p{6};
    n4m_matrix_view_t Xv{};
    n4m_matrix_view_t Yv{};
    n4m_validation_plan_t* plan{nullptr};
};

void build_finetune_fixture(FinetuneFixture& f) {
    f.X.assign(static_cast<std::size_t>(f.n * f.p), 0.0);
    f.Y.assign(static_cast<std::size_t>(f.n), 0.0);
    std::uint64_t s = 20240712ULL;
    auto rnd = [&]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11)) * (1.0 / 9007199254740992.0);
    };
    std::vector<double> beta(static_cast<std::size_t>(f.p));
    for (int64_t j = 0; j < f.p; ++j) beta[static_cast<std::size_t>(j)] = rnd() - 0.5;
    for (int64_t i = 0; i < f.n; ++i) {
        double y = 0.0;
        for (int64_t j = 0; j < f.p; ++j) {
            const double x = rnd();
            f.X[static_cast<std::size_t>(i * f.p + j)] = x;
            y += beta[static_cast<std::size_t>(j)] * x;
        }
        f.Y[static_cast<std::size_t>(i)] = y + 0.01 * rnd();
    }
    f.Xv.data = f.X.data();
    f.Xv.rows = f.n;
    f.Xv.cols = f.p;
    f.Xv.row_stride = f.p;
    f.Xv.col_stride = 1;
    f.Xv.dtype = N4M_DTYPE_F64;
    f.Yv.data = f.Y.data();
    f.Yv.rows = f.n;
    f.Yv.cols = 1;
    f.Yv.row_stride = 1;
    f.Yv.col_stride = 1;
    f.Yv.dtype = N4M_DTYPE_F64;
    N4M_TEST_REQUIRE(n4m_validation_plan_create(&f.plan) == N4M_OK);
    n4m_validation_plan_set_n_samples(f.plan, f.n);
    std::vector<int64_t> tr1, te1, tr2, te2;
    for (int64_t i = 0; i < f.n; ++i) {
        if (i % 2 == 0) {
            te1.push_back(i);
            tr2.push_back(i);
        } else {
            tr1.push_back(i);
            te2.push_back(i);
        }
    }
    n4m_validation_plan_add_fold(f.plan, tr1.data(), static_cast<int64_t>(tr1.size()),
                                 te1.data(), static_cast<int64_t>(te1.size()));
    n4m_validation_plan_add_fold(f.plan, tr2.data(), static_cast<int64_t>(tr2.size()),
                                 te2.data(), static_cast<int64_t>(te2.size()));
}

// Every eligible generic regression route yields a finite CV score and the
// documented result keys (best.n_components, best_score, metric, estimator, and
// best.sparsity_lambda only when a sparsity axis is declared).
void test_finetune_all_routes() {
    FinetuneFixture f;
    build_finetune_fixture(f);
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);

    struct Route {
        n4m_algorithm_t algo;
        bool            sparse;
    };
    const Route routes[] = {
        {N4M_ALGO_PLS_REGRESSION, false}, {N4M_ALGO_PLS_CANONICAL, false},
        {N4M_ALGO_PLS_SVD, false},        {N4M_ALGO_OPLS, false},
        {N4M_ALGO_PCR, false},            {N4M_ALGO_SPARSE_PLS, true},
    };
    for (const Route& route : routes) {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "n_components", 1, 3, 1, 0);
        if (route.sparse) {
            n4m_search_space_add_float(sp, "sparsity_lambda", 0.0, 0.9, 0.0, 0);
        }
        n4m_optimizer_options_t o = default_opts();
        o.seed = 7;
        o.metric = N4M_METRIC_RMSE;
        o.direction = N4M_OPT_MINIMIZE;
        n4m_method_result_t* res = nullptr;
        const n4m_status_t st =
            n4m_finetune_estimator(ctx, route.algo, &f.Xv, &f.Yv, f.plan, sp, &o, 8, &res);
        N4M_TEST_REQUIRE(st == N4M_OK);
        N4M_TEST_REQUIRE(res != nullptr);
        double best_score = -1.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "best_score", &best_score) == N4M_OK);
        N4M_TEST_REQUIRE(std::isfinite(best_score) && best_score >= 0.0 && best_score < 1e6);
        double best_nc = 0.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "best.n_components", &best_nc) == N4M_OK);
        N4M_TEST_REQUIRE(best_nc >= 1.0 && best_nc <= 3.0);
        double est = -1.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "estimator", &est) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<int>(est) == static_cast<int>(route.algo));
        double metric = -1.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "metric", &metric) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<int>(metric) == static_cast<int>(N4M_METRIC_RMSE));
        double lam = -1.0;
        const n4m_status_t lam_st =
            n4m_method_result_get_scalar(res, "best.sparsity_lambda", &lam);
        if (route.sparse) {
            N4M_TEST_REQUIRE(lam_st == N4M_OK);
            N4M_TEST_REQUIRE(lam >= 0.0 && lam < 1.0);
        } else {
            N4M_TEST_REQUIRE(lam_st != N4M_OK);
        }
        n4m_method_result_destroy(res);
        n4m_search_space_destroy(sp);
    }
    n4m_validation_plan_destroy(f.plan);
    n4m_context_destroy(ctx);
}

// The full ineligible + invalid-enum matrix is rejected with exactly
// N4M_ERR_UNSUPPORTED before any study is created (no result handle leaks).
void test_finetune_rejects_all_ineligible() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    double xd[8] = {1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0};
    double yd[4] = {1.0, 2.0, 3.0, 4.0};
    n4m_matrix_view_t Xv{};
    Xv.data = xd;
    Xv.rows = 4;
    Xv.cols = 2;
    Xv.row_stride = 2;
    Xv.col_stride = 1;
    Xv.dtype = N4M_DTYPE_F64;
    n4m_matrix_view_t Yv{};
    Yv.data = yd;
    Yv.rows = 4;
    Yv.cols = 1;
    Yv.row_stride = 1;
    Yv.col_stride = 1;
    Yv.dtype = N4M_DTYPE_F64;
    n4m_validation_plan_t* plan = nullptr;
    n4m_validation_plan_create(&plan);
    n4m_validation_plan_set_n_samples(plan, 4);
    int64_t tr[2] = {0, 1};
    int64_t te[2] = {2, 3};
    n4m_validation_plan_add_fold(plan, tr, 2, te, 2);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "n_components", 1, 2, 1, 0);
    n4m_optimizer_options_t o = default_opts();

    const n4m_algorithm_t ineligible[] = {
        N4M_ALGO_PLS_DA,
        N4M_ALGO_OPLS_DA,
        N4M_ALGO_MB_PLS,
        N4M_ALGO_LW_PLS,
        N4M_ALGO_AOM_PLS,
    };
    for (n4m_algorithm_t algo : ineligible) {
        n4m_method_result_t* res = nullptr;
        N4M_TEST_REQUIRE(
            n4m_finetune_estimator(ctx, algo, &Xv, &Yv, plan, sp, &o, 4, &res) ==
            N4M_ERR_UNSUPPORTED);
        N4M_TEST_REQUIRE(res == nullptr);
    }
#if !defined(N4M_UBSAN_BUILD)
    const n4m_algorithm_t invalid_algorithms[] = {
        invalid_enum<n4m_algorithm_t>(11),
        invalid_enum<n4m_algorithm_t>(42),
        invalid_enum<n4m_algorithm_t>(-1),
    };
    for (n4m_algorithm_t algo : invalid_algorithms) {
        n4m_method_result_t* res = nullptr;
        N4M_TEST_REQUIRE(
            n4m_finetune_estimator(ctx, algo, &Xv, &Yv, plan, sp, &o, 4, &res) ==
            N4M_ERR_UNSUPPORTED);
        N4M_TEST_REQUIRE(res == nullptr);
    }
#endif
    n4m_search_space_destroy(sp);
    n4m_validation_plan_destroy(plan);
    n4m_context_destroy(ctx);
}

// Structural axis-schema coverage: every kind/log/domain/step/duplicate/unknown/
// forbidden violation is a stable UNSUPPORTED preflight error, while the
// documented SPARSE_PLS omitted-axis defaults and safe LOG_FLOAT lambda pass.
void test_finetune_schema_matrix() {
    FinetuneFixture f;
    build_finetune_fixture(f);
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 3;

    auto call = [&](n4m_algorithm_t algo, n4m_search_space_t* space) -> n4m_status_t {
        n4m_method_result_t* res = nullptr;
        const n4m_status_t st =
            n4m_finetune_estimator(ctx, algo, &f.Xv, &f.Yv, f.plan, space, &o, 4, &res);
        if (res != nullptr) n4m_method_result_destroy(res);
        return st;
    };
    auto reject = [&](n4m_algorithm_t algo, auto build) {
        n4m_search_space_t* space = nullptr;
        n4m_search_space_create(&space);
        build(space);
        N4M_TEST_REQUIRE(call(algo, space) == N4M_ERR_UNSUPPORTED);
        n4m_search_space_destroy(space);
    };
    auto accept = [&](n4m_algorithm_t algo, auto build) {
        n4m_search_space_t* space = nullptr;
        n4m_search_space_create(&space);
        build(space);
        N4M_TEST_REQUIRE(call(algo, space) == N4M_OK);
        n4m_search_space_destroy(space);
    };

    // missing required n_components (dense route)
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t*) {});
    // wrong kind: float n_components
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "n_components", 1.0, 3.0, 0.0, 0);
    });
    // log int n_components
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 8, 1, 1);
    });
    // out-of-domain low
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 0, 3, 1, 0);
    });
    // out-of-domain high (> INT32_MAX)
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 2147483648LL, 1, 0);
    });
    // bad step semantics (step == 0)
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 4, 0, 0);
    });
    // duplicate axis
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 3, 1, 0);
        n4m_search_space_add_int(sp, "n_components", 1, 4, 1, 0);
    });
    // unknown axis
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 3, 1, 0);
        n4m_search_space_add_float(sp, "unknown", 0.0, 1.0, 0.0, 0);
    });
    // forbidden sparsity axis on a dense route
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 3, 1, 0);
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.0, 0.5, 0.0, 0);
    });
    // SPARSE_PLS: empty subset rejected
    reject(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t*) {});
    // SPARSE_PLS: sparsity lambda out of domain (high == 1)
    reject(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.0, 1.0, 0.0, 0);
    });
    // SPARSE_PLS: sparsity lambda negative low
    reject(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", -0.1, 0.5, 0.0, 0);
    });
    // Negative linear step is accepted by the low-level builder but must be a
    // stable MT8 preflight refusal, not a later optimizer-create error.
    reject(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.0, 0.5, -0.1, 0);
    });
    const char* step_error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(step_error != nullptr &&
                     std::string(step_error).find("step") != std::string::npos);
    // A quantized step that cannot advance at the domain scale is refused by
    // the same preflight.
    reject(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.5, 0.75,
                                   std::numeric_limits<double>::denorm_min(), 0);
    });
    // LOG_FLOAT must be continuous (step == 0).
    reject(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.01, 0.5, 0.1, 1);
    });
    // Conditional finetune axes are deliberately unsupported: all registered
    // axes are numeric, while the optimizer requires a categorical/ordinal
    // condition parent. Refuse instead of publishing a dormant best value.
    reject(N4M_ALGO_PLS_REGRESSION, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 3, 1, 0);
        const char* modes[2] = {"on", "off"};
        n4m_search_space_add_categorical(sp, "mode", N4M_CAT_STR, modes, 2);
        const char* refs[2] = {"n_components", "mode"};
        const char* labels[2] = {nullptr, "on"};
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, refs, labels, 2);
    });
    const char* condition_error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(condition_error != nullptr &&
                     std::string(condition_error).find("conditional") !=
                         std::string::npos);

    // SPARSE_PLS: only n_components (sparsity_lambda uses documented default 0.0)
    accept(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_int(sp, "n_components", 1, 3, 1, 0);
    });
    // SPARSE_PLS: only sparsity_lambda (n_components uses documented default 2)
    accept(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.0, 0.5, 0.0, 0);
    });
    // SPARSE_PLS: safe LOG_FLOAT sparsity lambda (low > 0)
    accept(N4M_ALGO_SPARSE_PLS, [](n4m_search_space_t* sp) {
        n4m_search_space_add_float(sp, "sparsity_lambda", 0.01, 0.5, 0.0, 1);
    });

    n4m_validation_plan_destroy(f.plan);
    n4m_context_destroy(ctx);
}

// The sampled sparsity_lambda is actually threaded into the sparse SIMPLS fit:
// two degenerate single-point lambda axes (0.0 vs 0.9) give different CV scores,
// and best.sparsity_lambda records the value that was applied.
void test_finetune_sparse_lambda_applied() {
    FinetuneFixture f;
    build_finetune_fixture(f);
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);

    auto run_fixed_lambda = [&](double lambda, double* out_score, double* out_lambda) {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "n_components", 2, 2, 1, 0);
        n4m_search_space_add_float(sp, "sparsity_lambda", lambda, lambda, 0.0, 0);
        n4m_optimizer_options_t o = default_opts();
        o.seed = 13;
        o.metric = N4M_METRIC_RMSE;
        n4m_method_result_t* res = nullptr;
        N4M_TEST_REQUIRE(
            n4m_finetune_estimator(ctx, N4M_ALGO_SPARSE_PLS, &f.Xv, &f.Yv, f.plan, sp, &o, 2,
                                   &res) == N4M_OK);
        N4M_TEST_REQUIRE(res != nullptr);
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "best_score", out_score) == N4M_OK);
        N4M_TEST_REQUIRE(
            n4m_method_result_get_scalar(res, "best.sparsity_lambda", out_lambda) == N4M_OK);
        double best_nc = 0.0;
        N4M_TEST_REQUIRE(
            n4m_method_result_get_scalar(res, "best.n_components", &best_nc) == N4M_OK);
        N4M_TEST_REQUIRE(best_nc == 2.0);
        double est = -1.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(res, "estimator", &est) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<int>(est) == static_cast<int>(N4M_ALGO_SPARSE_PLS));
        n4m_method_result_destroy(res);
        n4m_search_space_destroy(sp);
    };

    double score_zero = 0.0, lambda_zero = -1.0;
    double score_high = 0.0, lambda_high = -1.0;
    run_fixed_lambda(0.0, &score_zero, &lambda_zero);
    run_fixed_lambda(0.9, &score_high, &lambda_high);
    N4M_TEST_REQUIRE(lambda_zero == 0.0);
    N4M_TEST_REQUIRE(lambda_high == 0.9);
    N4M_TEST_REQUIRE(std::isfinite(score_zero) && std::isfinite(score_high));
    // A non-zero soft-threshold changes the fitted coefficients, hence the score.
    N4M_TEST_REQUIRE(std::fabs(score_high - score_zero) > 1e-9);

    n4m_validation_plan_destroy(f.plan);
    n4m_context_destroy(ctx);
}

// A late candidate failure is preserved in the trace but must not poison a
// successful return's context error. Previewing the deterministic random stream
// lets the test select a seed whose first component count is valid (<= p) and
// whose second is invalid (> p), without hard-coding RNG output.
void test_finetune_success_clears_late_candidate_error() {
    FinetuneFixture f;
    build_finetune_fixture(f);
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "n_components", 1, 12, 1, 0);

    n4m_optimizer_options_t o = default_opts();
    bool found = false;
    for (std::uint64_t seed = 1; seed < 2048 && !found; ++seed) {
        o.seed = seed;
        n4m_optimizer_t* preview = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &preview) == N4M_OK);
        n4m_trial_t* first = nullptr;
        n4m_trial_t* second = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_ask(preview, &first) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_optimizer_ask(preview, &second) == N4M_OK);
        int64_t first_components = 0;
        int64_t second_components = 0;
        n4m_trial_get_int(first, "n_components", &first_components);
        n4m_trial_get_int(second, "n_components", &second_components);
        found = first_components <= f.p && second_components > f.p;
        n4m_optimizer_destroy(preview);
    }
    N4M_TEST_REQUIRE(found);

    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PLS_REGRESSION, &f.Xv, &f.Yv, f.plan, sp,
                         &o, 2, &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    const double* statuses = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "trial_status", &statuses, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == 1 && cols == 2);
    N4M_TEST_REQUIRE(statuses[0] == static_cast<double>(N4M_TRIAL_COMPLETED));
    N4M_TEST_REQUIRE(statuses[1] == static_cast<double>(N4M_TRIAL_FAILED));
    const char* error = n4m_context_last_error(ctx);
    N4M_TEST_REQUIRE(error == nullptr || *error == '\0');

    n4m_method_result_destroy(result);
    n4m_search_space_destroy(sp);
    n4m_validation_plan_destroy(f.plan);
    n4m_context_destroy(ctx);
}

void test_finetune_timeout_contract() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_method_result_t* result = nullptr;

    // PCR makes the first candidate long enough for a short deadline to expire
    // before the next ask. Completed progress is returned as a partial success.
    FinetuneFixture heavy;
    heavy.n = 400;
    heavy.p = 120;
    build_finetune_fixture(heavy);
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "n_components", 3, 3, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.timeout_seconds = 0.02;
    N4M_TEST_REQUIRE(n4m_finetune_estimator(
                         ctx, N4M_ALGO_PCR, &heavy.Xv, &heavy.Yv, heavy.plan, sp,
                         &o, 8, &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    double timed_out = 0.0;
    double requested = 0.0;
    N4M_TEST_REQUIRE(
        n4m_method_result_get_scalar(result, "timed_out", &timed_out) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(
                         result, "requested_trials", &requested) == N4M_OK);
    N4M_TEST_REQUIRE(timed_out == 1.0 && requested == 8.0);
    const double* trial_ids = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "trial_ids", &trial_ids, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == 1 && cols >= 1 && cols < 8);

    n4m_method_result_destroy(result);
    n4m_search_space_destroy(sp);
    n4m_validation_plan_destroy(heavy.plan);
    n4m_context_destroy(ctx);
}

void test_auto_direction_maximizes_r2() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 100, 1, 0);
    n4m_optimizer_options_t o = default_opts();  // direction == AUTO
    o.metric = N4M_METRIC_R2;                     // higher is better
    o.seed = 11;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    // tell three scores; best under AUTO+R2 must be the largest
    for (int i = 0; i < 3; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        const double score = (i == 1) ? 0.9 : 0.2;  // trial 1 is best
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = -1.0;
    n4m_optimizer_best(opt, &best, &bs);
    N4M_TEST_REQUIRE(bs == 0.9);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_ternary_converges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 30, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_TERNARY;
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 25; ++i) {  // unimodal objective, optimum at k = 7
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        int64_t k = 0;
        n4m_trial_get_int(t, "k", &k);
        const double score = static_cast<double>((k - 7) * (k - 7));
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs <= 1.0);  // converged to k in {6, 7, 8}
    int64_t bk = 0;
    n4m_trial_get_int(best, "k", &bk);
    N4M_TEST_REQUIRE(bk >= 6 && bk <= 8);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_ternary_respects_step_and_batch() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 2, 20, 2, 0);  // even values only
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_TERNARY;
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 2;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    // batch ask before any tell: reservations must yield distinct on-grid values
    n4m_trial_t* trials[3] = {nullptr, nullptr, nullptr};
    int32_t count = 0;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 3, trials, &count) == N4M_OK);
    N4M_TEST_REQUIRE(count == 3);
    int64_t vals[3];
    for (int i = 0; i < 3; ++i) {
        n4m_trial_get_int(trials[i], "k", &vals[i]);
        N4M_TEST_REQUIRE(vals[i] % 2 == 0);              // honours step=2
        N4M_TEST_REQUIRE(vals[i] >= 2 && vals[i] <= 20);
    }
    N4M_TEST_REQUIRE(vals[0] != vals[1] && vals[1] != vals[2] && vals[0] != vals[2]);  // reserved distinct
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_ternary_keeps_large_first_integer_axis() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "large", 0, 100000000, 1, 0);
    n4m_search_space_add_int(sp, "later", 1, 3, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_TERNARY;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* first = nullptr;
    n4m_trial_t* second = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &first) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &second) == N4M_OK);
    int64_t first_value = -1;
    int64_t second_value = -1;
    n4m_trial_get_int(first, "large", &first_value);
    n4m_trial_get_int(second, "large", &second_value);
    N4M_TEST_REQUIRE(first_value == 0);
    N4M_TEST_REQUIRE(second_value == 100000000);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_stepped_grid_sampling_and_enqueue_consistency() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);

    {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "k", 2, 9, 2, 0);
        n4m_optimizer_options_t o = default_opts();
        o.seed = 991;
        n4m_optimizer_t* opt = nullptr;
        n4m_optimizer_t* verifier = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &verifier) == N4M_OK);
        int counts[4] = {0, 0, 0, 0};
        const char* names[1] = {"k"};
        for (int i = 0; i < 1200; ++i) {
            n4m_trial_t* trial = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &trial) == N4M_OK);
            int64_t value = 0;
            n4m_trial_get_int(trial, "k", &value);
            N4M_TEST_REQUIRE(value >= 2 && value <= 8 && (value - 2) % 2 == 0);
            counts[(value - 2) / 2] += 1;
            const double encoded[1] = {static_cast<double>(value)};
            N4M_TEST_REQUIRE(n4m_optimizer_enqueue(verifier, names, encoded, 1) == N4M_OK);
            int64_t id = 0;
            n4m_trial_get_id(trial, &id);
            n4m_optimizer_tell(opt, id, 0.0);
        }
        int min_count = counts[0];
        int max_count = counts[0];
        for (const int count : counts) {
            min_count = std::min(min_count, count);
            max_count = std::max(max_count, count);
        }
        N4M_TEST_REQUIRE(max_count - min_count < 100);
        n4m_optimizer_destroy(verifier);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }

    {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.35, 0);
        n4m_optimizer_options_t o = default_opts();
        o.seed = 992;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        int counts[3] = {0, 0, 0};
        for (int i = 0; i < 900; ++i) {
            n4m_trial_t* trial = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &trial) == N4M_OK);
            double value = -1.0;
            n4m_trial_get_float(trial, "x", &value);
            int bucket = -1;
            if (value == 0.0) bucket = 0;
            if (value == 0.35) bucket = 1;
            if (value == 0.70) bucket = 2;
            N4M_TEST_REQUIRE(bucket >= 0);
            counts[bucket] += 1;
            int64_t id = 0;
            n4m_trial_get_id(trial, &id);
            n4m_optimizer_tell(opt, id, 0.0);
        }
        int min_count = std::min({counts[0], counts[1], counts[2]});
        int max_count = std::max({counts[0], counts[1], counts[2]});
        N4M_TEST_REQUIRE(max_count - min_count < 100);
        n4m_optimizer_destroy(opt);

        n4m_optimizer_t* verifier = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &verifier) == N4M_OK);
        const char* names[1] = {"x"};
        const double off_grid[1] = {1.0};
        const double on_grid[1] = {0.70};
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(verifier, names, off_grid, 1) ==
                         N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(verifier, names, on_grid, 1) == N4M_OK);
        n4m_optimizer_destroy(verifier);
        n4m_search_space_destroy(sp);
    }

    {
        n4m_search_space_t* tiny = nullptr;
        n4m_search_space_create(&tiny);
        n4m_search_space_add_float(tiny, "x", 0.0, 1e-18, 1e-20, 0);
        n4m_optimizer_options_t o = default_opts();
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, tiny, &o, &opt) == N4M_OK);
        const char* names[1] = {"x"};
        const double halfway[1] = {5e-21};
        const double aligned[1] = {4e-20};
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, halfway, 1) ==
                         N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, aligned, 1) == N4M_OK);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(tiny);

        n4m_search_space_t* large = nullptr;
        n4m_search_space_create(&large);
        n4m_search_space_add_float(large, "x", 1e12, 1e12 + 1.0, 0.25, 0);
        n4m_optimizer_create(ctx, large, &o, &opt);
        const double large_aligned[1] = {1e12 + 0.5};
        const double large_off_grid[1] = {1e12 + 0.375};
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, large_aligned, 1) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, large_off_grid, 1) ==
                         N4M_ERR_INVALID_ARGUMENT);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(large);
    }

    n4m_context_destroy(ctx);
}

void test_enqueue_out_of_range_rejected() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    const char* names[1] = {"k"};
    const double bad[1] = {99.0};  // out of [1,10]
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, bad, 1) == N4M_ERR_INVALID_ARGUMENT);
    const double nan_value[1] = {std::nan("")};
    N4M_TEST_REQUIRE(
        n4m_optimizer_enqueue(opt, names, nan_value, 1) == N4M_ERR_INVALID_ARGUMENT);
    const double inf_value[1] = {std::numeric_limits<double>::infinity()};
    N4M_TEST_REQUIRE(
        n4m_optimizer_enqueue(opt, names, inf_value, 1) == N4M_ERR_INVALID_ARGUMENT);
    const double fractional[1] = {5.5};
    N4M_TEST_REQUIRE(
        n4m_optimizer_enqueue(opt, names, fractional, 1) == N4M_ERR_INVALID_ARGUMENT);
    const char* duplicate_names[2] = {"k", "k"};
    const double duplicate_values[2] = {4.0, 5.0};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, duplicate_names, duplicate_values, 2) ==
                     N4M_ERR_INVALID_ARGUMENT);
    const double ok[1] = {5.0};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, ok, 1) == N4M_OK);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);

    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "even", 2, 10, 2, 0);
    n4m_optimizer_create(ctx, sp, &o, &opt);
    const char* even_name[1] = {"even"};
    const double odd[1] = {3.0};
    const double even[1] = {4.0};
    N4M_TEST_REQUIRE(
        n4m_optimizer_enqueue(opt, even_name, odd, 1) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, even_name, even, 1) == N4M_OK);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);

    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "grid", 0.0, 1.0, 0.2, 0);
    n4m_optimizer_create(ctx, sp, &o, &opt);
    const char* grid_name[1] = {"grid"};
    const double off_grid[1] = {0.3};
    const double on_grid[1] = {0.4};
    N4M_TEST_REQUIRE(
        n4m_optimizer_enqueue(opt, grid_name, off_grid, 1) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, grid_name, on_grid, 1) == N4M_OK);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);

    n4m_search_space_create(&sp);
    const char* choices[2] = {"a", "b"};
    n4m_search_space_add_categorical(sp, "choice", N4M_CAT_STR, choices, 2);
    n4m_optimizer_create(ctx, sp, &o, &opt);
    const char* choice_name[1] = {"choice"};
    const double fractional_index[1] = {0.5};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, choice_name, fractional_index, 1) ==
                     N4M_ERR_INVALID_ARGUMENT);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_median_pruner() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.pruner = N4M_PRUNER_MEDIAN;
    o.direction = N4M_OPT_MINIMIZE;
    o.n_startup_trials = 2;  // min peers before pruning
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* t[3] = {nullptr, nullptr, nullptr};
    int64_t id[3];
    for (int i = 0; i < 3; ++i) {
        n4m_optimizer_ask(opt, &t[i]);
        n4m_trial_get_id(t[i], &id[i]);
    }
    int32_t prune = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[0], 0, 1.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);  // no peers yet
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[1], 0, 2.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);  // 1 peer < min_peers=2
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 0, 5.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 1);  // worse than the peer median of {1,2} (= 1.5)
    n4m_trial_status_t st;
    n4m_trial_get_status(t[2], &st);
    N4M_TEST_REQUIRE(st == N4M_TRIAL_PRUNED);
    // a strong trial is not pruned
    n4m_trial_t* t3 = nullptr;
    int64_t id3 = 0;
    n4m_optimizer_ask(opt, &t3);
    n4m_trial_get_id(t3, &id3);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id3, 0, 0.5, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);  // better than the median
    n4m_optimizer_destroy(opt);

    o.direction = N4M_OPT_MAXIMIZE;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    const double max_finite = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++i) {
        n4m_trial_t* trial = nullptr;
        int64_t trial_id = 0;
        n4m_optimizer_ask(opt, &trial);
        n4m_trial_get_id(trial, &trial_id);
        N4M_TEST_REQUIRE(
            n4m_optimizer_tell_intermediate(opt, trial_id, 0, max_finite, &prune) == N4M_OK);
        N4M_TEST_REQUIRE(prune == 0);
    }
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_asha_pruner() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.pruner = N4M_PRUNER_ASHA;  // reduction_factor = 3
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* t[3] = {nullptr, nullptr, nullptr};
    int64_t id[3];
    for (int i = 0; i < 3; ++i) {
        n4m_optimizer_ask(opt, &t[i]);
        n4m_trial_get_id(t[i], &id[i]);
    }
    int32_t prune = -1;
    // fewer than reduction_factor peers at the rung → survive
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[0], 0, 1.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[1], 0, 2.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);
    // 3 at the rung, top 1/3 = 1 survives; worst is pruned
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 0, 9.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 1);
    // the best-scoring newcomer survives
    n4m_trial_t* t3 = nullptr;
    int64_t id3 = 0;
    n4m_optimizer_ask(opt, &t3);
    n4m_trial_get_id(t3, &id3);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id3, 0, 0.5, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_hyperband_brackets() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.pruner = N4M_PRUNER_HYPERBAND;
    o.max_resource = 9;       // rungs at resource 1,3,9 → steps 0,2,8
    o.reduction_factor = 3;   // eta → 3 brackets (0,1,2), assigned by ask order
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* t[7] = {nullptr};
    int64_t id[7];
    for (int i = 0; i < 7; ++i) {  // brackets 0,1,2,0,1,2,0
        n4m_optimizer_ask(opt, &t[i]);
        n4m_trial_get_id(t[i], &id[i]);
    }
    int32_t prune = -1;
    // Successive halving within bracket 0 (trials idx 0,3,6) at rung 0.
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[0], 0, 1.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);  // <eta peers yet
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[3], 0, 2.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[6], 0, 9.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 1);  // worst of the 3 in bracket 0 → pruned
    // Grace period: a bracket-1 trial is exempt at rung 0 even with a terrible score.
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[1], 0, 100.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);  // rung 0 < bracket 1's grace rung → survives
    // A non-rung step (resource 2 is not a power of eta) never prunes.
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[4], 1, 100.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_hyperband_edges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    // (a) max_resource == 0 is rejected: R must be known upfront for stable brackets.
    {
        n4m_optimizer_options_t o = default_opts();
        o.pruner = N4M_PRUNER_HYPERBAND;
        o.max_resource = 0;
        o.reduction_factor = 3;
        n4m_optimizer_t* bad = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &bad) == N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(bad == nullptr);
    }
    // (b) rungs above max_resource never prune (k > k_max guard).
    {
        n4m_optimizer_options_t o = default_opts();
        o.pruner = N4M_PRUNER_HYPERBAND;
        o.max_resource = 3;      // k_max = 1 → rungs 0,1; resource 9 (step 8) is rung 2, above R
        o.reduction_factor = 3;  // n_brackets = 2; bracket-0 = idx 0,2,4
        o.direction = N4M_OPT_MINIMIZE;
        o.seed = 1;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        n4m_trial_t* t[5] = {nullptr};
        int64_t id[5];
        for (int i = 0; i < 5; ++i) {
            n4m_optimizer_ask(opt, &t[i]);
            n4m_trial_get_id(t[i], &id[i]);
        }
        int32_t prune = -1;
        // Three same-bracket trials at rung 2 (above R): without the guard the worst
        // would be halved out; with it, all survive.
        N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[0], 8, 1.0, &prune) == N4M_OK);
        N4M_TEST_REQUIRE(prune == 0);
        N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 8, 2.0, &prune) == N4M_OK);
        N4M_TEST_REQUIRE(prune == 0);
        N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[4], 8, 9.0, &prune) == N4M_OK);
        N4M_TEST_REQUIRE(prune == 0);  // rung 2 > k_max=1 → never prune
        n4m_optimizer_destroy(opt);
    }
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_ga_converges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_search_space_add_float(sp, "y", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_GA;
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 7;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 240; ++i) {  // ~15 generations of pop 16; optimum (2,-3)
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        double y = 0.0;
        n4m_trial_get_float(t, "x", &x);
        n4m_trial_get_float(t, "y", &y);
        const double score = (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs < 1.0);  // GA converges toward the interior optimum
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_tpe_converges_mixed() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", 0.0, 10.0, 0.0, 0);
    const char* cats[3] = {"a", "b", "c"};
    n4m_search_space_add_categorical(sp, "c", N4M_CAT_STR, cats, 3);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_TPE;
    o.direction = N4M_OPT_MINIMIZE;
    o.n_startup_trials = 15;
    o.seed = 2;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 140; ++i) {  // objective minimised at x=3, c="a"
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        n4m_trial_get_float(t, "x", &x);
        int32_t ci = -1;
        const char* cl = nullptr;
        n4m_trial_get_category(t, "c", &ci, &cl);
        const double pen = (ci == 0) ? 0.0 : (ci == 1) ? 5.0 : 10.0;
        const double score = (x - 3.0) * (x - 3.0) + pen;
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs < 1.0);  // x near 3 with the best category
    int32_t bci = -1;
    const char* bcl = nullptr;
    n4m_trial_get_category(best, "c", &bci, &bcl);
    N4M_TEST_REQUIRE(bci == 0);  // category "a"
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_cmaes_converges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_search_space_add_float(sp, "y", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_CMAES;
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 5;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 300; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        double y = 0.0;
        n4m_trial_get_float(t, "x", &x);
        n4m_trial_get_float(t, "y", &y);
        const double score = (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs < 0.1);  // CMA-ES converges tightly on a smooth objective
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_gp_ei_converges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_search_space_add_float(sp, "y", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_GP_EI;
    o.direction = N4M_OPT_MINIMIZE;
    o.n_startup_trials = 8;
    o.seed = 7;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 60; ++i) {  // sample-efficient: far fewer trials than random/CMA
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        double y = 0.0;
        n4m_trial_get_float(t, "x", &x);
        n4m_trial_get_float(t, "y", &y);
        const double score = (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs < 0.5);  // GP-EI locates the basin in ~60 evals
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_gp_ei_maximize() {
    // Exercises the MAXIMIZE branch of the EI sign: the optimum is a peak.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_search_space_add_float(sp, "y", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_GP_EI;
    o.direction = N4M_OPT_MAXIMIZE;
    o.n_startup_trials = 8;
    o.seed = 3;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 60; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        double y = 0.0;
        n4m_trial_get_float(t, "x", &x);
        n4m_trial_get_float(t, "y", &y);
        const double score = -((x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0));  // peak 0
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = -1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs > -0.5);  // climbs the peak → near 0 (wrong EI sign would flee it)
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_gp_ei_edge_cases() {
    // (a) No continuous axis (pure categorical) → GP degrades to random; must not
    //     crash and must return a best. (b) A constant objective → all-equal y and
    //     duplicate decoded coords; the jittered Cholesky must stay stable.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    {  // (a) pure-categorical
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        const char* labels[3] = {"a", "b", "c"};
        n4m_search_space_add_categorical(sp, "c", N4M_CAT_STR, labels, 3);
        n4m_optimizer_options_t o = default_opts();
        o.sampler = N4M_SAMPLER_GP_EI;
        o.direction = N4M_OPT_MINIMIZE;
        o.seed = 1;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        for (int i = 0; i < 30; ++i) {
            n4m_trial_t* t = nullptr;
            n4m_optimizer_ask(opt, &t);
            int32_t idx = 0;
            const char* lab = nullptr;
            n4m_trial_get_category(t, "c", &idx, &lab);
            int64_t id = 0;
            n4m_trial_get_id(t, &id);
            n4m_optimizer_tell(opt, id, static_cast<double>(idx));  // prefers "a"
        }
        n4m_trial_t* best = nullptr;
        double bs = 1e9;
        N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
        N4M_TEST_REQUIRE(bs == 0.0);  // "a" (index 0) is reachable by random fallback
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }
    {  // (b) constant objective → all-equal y, single continuous axis, duplicate coords
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "k", 1, 3, 1, 0);  // small int → repeated decoded coords
        n4m_optimizer_options_t o = default_opts();
        o.sampler = N4M_SAMPLER_GP_EI;
        o.n_startup_trials = 4;
        o.seed = 2;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        for (int i = 0; i < 30; ++i) {
            n4m_trial_t* t = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);  // must not throw across the ABI
            int64_t id = 0;
            n4m_trial_get_id(t, &id);
            n4m_optimizer_tell(opt, id, 7.0);  // constant
        }
        n4m_trial_t* best = nullptr;
        double bs = 0.0;
        N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
        N4M_TEST_REQUIRE(bs == 7.0);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }
    n4m_context_destroy(ctx);
}

void test_population_batch_and_enqueue() {
    // MT12 — population generation boundary is a BENIGN partial for GA, PSO and
    // CMA-ES: ask_batch commits the current generation and returns N4M_OK with
    // 0 < count < n. The same boundary at zero capacity surfaces its stable
    // status (N4M_ERR_INVALID_ARGUMENT / count 0). Scoring the generation lets
    // the next batch advance. Population sizes differ (GA/PSO 16, CMA-ES λ), so
    // the boundary is discovered dynamically.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    const n4m_sampler_kind_t pop_samplers[3] = {
        N4M_SAMPLER_GA, N4M_SAMPLER_PSO, N4M_SAMPLER_CMAES};
    for (const auto sampler : pop_samplers) {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
        n4m_search_space_add_float(sp, "y", 0.0, 1.0, 0.0, 0);
        n4m_optimizer_options_t o = default_opts();
        o.sampler = sampler;
        o.direction = N4M_OPT_MINIMIZE;
        o.seed = 1;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);

        // A large batch stops at the first generation boundary: OK, 0<count<n.
        auto* const sentinel =
            reinterpret_cast<n4m_trial_t*>(static_cast<std::uintptr_t>(1));
        n4m_trial_t* buf[512];
        for (auto& slot : buf) slot = sentinel;
        int32_t count = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 512, buf, &count) == N4M_OK);
        N4M_TEST_REQUIRE(count > 0 && count < 512);
        const int32_t pop = count;
        for (int32_t i = pop; i < 512; ++i) N4M_TEST_REQUIRE(buf[i] == nullptr);

        // Zero capacity: the boundary surfaces INVALID_ARGUMENT with count 0.
        n4m_trial_t* again[4] = {sentinel, sentinel, sentinel, sentinel};
        int32_t again_count = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 4, again, &again_count) ==
                         N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(again_count == 0);
        for (auto* slot : again) N4M_TEST_REQUIRE(slot == nullptr);
        n4m_trial_t* one = nullptr;  // a single ask agrees at the boundary
        N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &one) == N4M_ERR_INVALID_ARGUMENT);

        // Score the committed generation; the next batch then advances.
        for (int32_t i = 0; i < pop; ++i) {
            int64_t id = 0;
            n4m_trial_get_id(buf[i], &id);
            N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, id, 0.5) == N4M_OK);
        }
        for (auto& slot : buf) slot = sentinel;
        int32_t next_count = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 512, buf, &next_count) == N4M_OK);
        N4M_TEST_REQUIRE(next_count == pop);  // a fresh full generation
        for (int32_t i = pop; i < 512; ++i) N4M_TEST_REQUIRE(buf[i] == nullptr);

        // enqueue/warm-start stays unsupported for population samplers.
        const char* names[1] = {"x"};
        const double v[1] = {0.3};
        N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, v, 1) == N4M_ERR_UNSUPPORTED);
        n4m_optimizer_destroy(opt);
        n4m_search_space_destroy(sp);
    }
    n4m_context_destroy(ctx);
}

void test_ask_batch_contract() {
    // MT12 C contract: null precedence, out_count, out_trials-NULL-only-for-n==0,
    // n<0, n==0, full batch, every slot initialised, borrowed handles valid, a
    // non-boundary partial-fatal error (invalid queued warm-start), and a
    // timeout before the first ask.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 100, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.seed = 7;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);

    auto* const sentinel =
        reinterpret_cast<n4m_trial_t*>(static_cast<std::uintptr_t>(1));
    n4m_trial_t* buf[8];
    for (auto& slot : buf) slot = sentinel;
    int32_t count = -1;

    // Null precedence: out_count is required (cannot report anything without it).
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 3, buf, nullptr) == N4M_ERR_NULL_POINTER);
    // opt required; out_count still zeroed when provided.
    count = 999;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(nullptr, 3, buf, &count) == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(count == 0);
    // n < 0 => INVALID_ARGUMENT / count 0 (even with a NULL out_trials).
    count = 999;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, -1, nullptr, &count) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(count == 0);
    // n == 0 => OK / count 0; out_trials may be NULL.
    count = 999;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 0, nullptr, &count) == N4M_OK);
    N4M_TEST_REQUIRE(count == 0);
    // n > 0 with a NULL out_trials is a null pointer error.
    count = 999;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 3, nullptr, &count) == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(count == 0);

    // Full batch: OK, count == n, every requested slot filled with distinct ids,
    // trailing slots untouched, and borrowed handles remain usable.
    for (auto& slot : buf) slot = sentinel;
    count = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 5, buf, &count) == N4M_OK);
    N4M_TEST_REQUIRE(count == 5);
    for (int i = 0; i < 5; ++i) {
        N4M_TEST_REQUIRE(buf[i] != nullptr);
        int64_t id = -1;
        N4M_TEST_REQUIRE(n4m_trial_get_id(buf[i], &id) == N4M_OK);
        N4M_TEST_REQUIRE(id == static_cast<int64_t>(i));
    }
    for (int i = 5; i < 8; ++i) N4M_TEST_REQUIRE(buf[i] == sentinel);
    // Borrowed handles from the batch are still valid after further asks.
    n4m_trial_t* extra = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &extra) == N4M_OK);
    int64_t first_id = -1;
    N4M_TEST_REQUIRE(n4m_trial_get_id(buf[0], &first_id) == N4M_OK);
    N4M_TEST_REQUIRE(first_id == 0);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);

    // Non-boundary partial-fatal: a valid warm-start commits trial 0, then an
    // invalid (cross-constraint) warm-start makes the next ask fail with the
    // exact non-OK status; [0,count) stay valid, remaining slots NULL. Uses a
    // hard EXCLUDE on a rejection sampler (random) so the warm-start is checked.
    n4m_search_space_t* cs = nullptr;
    n4m_search_space_create(&cs);
    const char* choices[2] = {"off", "on"};
    n4m_search_space_add_categorical(cs, "a", N4M_CAT_STR, choices, 2);
    n4m_search_space_add_categorical(cs, "b", N4M_CAT_STR, choices, 2);
    const char* refs[2] = {"a", "b"};
    const char* labels[2] = {"on", "on"};
    n4m_search_space_add_constraint(cs, N4M_CONSTRAINT_EXCLUDE, refs, labels, 2);
    n4m_optimizer_options_t co = default_opts();
    co.sampler = N4M_SAMPLER_RANDOM;
    co.seed = 3;
    n4m_optimizer_t* copt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, cs, &co, &copt) == N4M_OK);
    const char* names[2] = {"a", "b"};
    const double ok_vals[2] = {0.0, 0.0};   // both off: satisfies EXCLUDE
    const double bad_vals[2] = {1.0, 1.0};  // both on: violates EXCLUDE
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(copt, names, ok_vals, 2) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(copt, names, bad_vals, 2) == N4M_OK);
    n4m_trial_t* cbuf[4] = {sentinel, sentinel, sentinel, sentinel};
    int32_t ccount = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(copt, 4, cbuf, &ccount) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(ccount == 1);       // the valid warm-start committed
    N4M_TEST_REQUIRE(cbuf[0] != nullptr);
    N4M_TEST_REQUIRE(cbuf[1] == nullptr && cbuf[2] == nullptr && cbuf[3] == nullptr);
    n4m_trial_status_t committed_status = N4M_TRIAL_FAILED;
    N4M_TEST_REQUIRE(n4m_trial_get_status(cbuf[0], &committed_status) == N4M_OK);
    N4M_TEST_REQUIRE(committed_status == N4M_TRIAL_RUNNING);
    int64_t committed_id = -1;
    N4M_TEST_REQUIRE(n4m_trial_get_id(cbuf[0], &committed_id) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(copt, committed_id, 0.0) == N4M_OK);
    n4m_optimizer_destroy(copt);
    n4m_search_space_destroy(cs);

    // Timeout before the first ask: CANCELLED / count 0 (deterministic — any
    // real elapsed time exceeds a 1 ns deadline).
    n4m_search_space_t* ts = nullptr;
    n4m_search_space_create(&ts);
    n4m_search_space_add_int(ts, "k", 1, 100, 1, 0);
    n4m_optimizer_options_t to = default_opts();
    to.timeout_seconds = 1e-9;
    n4m_optimizer_t* topt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, ts, &to, &topt) == N4M_OK);
    n4m_trial_t* tbuf[2] = {sentinel, sentinel};
    int32_t tcount = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(topt, 2, tbuf, &tcount) == N4M_ERR_CANCELLED);
    N4M_TEST_REQUIRE(tcount == 0);
    N4M_TEST_REQUIRE(tbuf[0] == nullptr && tbuf[1] == nullptr);
    n4m_optimizer_destroy(topt);
    n4m_search_space_destroy(ts);

    n4m_context_destroy(ctx);
}

void test_pso_converges() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
    n4m_search_space_add_float(sp, "y", -5.0, 5.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_PSO;
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 3;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    for (int i = 0; i < 240; ++i) {  // ~15 swarm iterations of 16 particles
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        double y = 0.0;
        n4m_trial_get_float(t, "x", &x);
        n4m_trial_get_float(t, "y", &y);
        const double score = (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, score);
    }
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    N4M_TEST_REQUIRE(n4m_optimizer_best(opt, &best, &bs) == N4M_OK);
    N4M_TEST_REQUIRE(bs < 1.0);  // swarm converges to the interior optimum
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_racing_pruner() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.pruner = N4M_PRUNER_RACING;
    o.direction = N4M_OPT_MINIMIZE;
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* t[3] = {nullptr, nullptr, nullptr};
    int64_t id[3];
    for (int i = 0; i < 3; ++i) {
        n4m_optimizer_ask(opt, &t[i]);
        n4m_trial_get_id(t[i], &id[i]);
    }
    int32_t pr = 0;
    for (int s = 0; s < 10; ++s) {  // two good trials: 10 observations each at 1.0
        n4m_optimizer_tell_intermediate(opt, id[0], s, 1.0, &pr);
        n4m_optimizer_tell_intermediate(opt, id[1], s, 1.0, &pr);
    }
    // a clearly-worse trial: not pruned on the first observation, pruned once the
    // confidence interval tightens with more observations
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 0, 10.0, &pr) == N4M_OK);
    N4M_TEST_REQUIRE(pr == 0);  // n=1 < 2
    bool pruned = false;
    for (int s = 1; s < 10 && !pruned; ++s) {
        n4m_optimizer_tell_intermediate(opt, id[2], s, 10.0, &pr);
        if (pr == 1) pruned = true;
    }
    N4M_TEST_REQUIRE(pruned);
    n4m_trial_status_t st;
    n4m_trial_get_status(t[2], &st);
    N4M_TEST_REQUIRE(st == N4M_TRIAL_PRUNED);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_pruner_lifecycle() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.pruner = N4M_PRUNER_MEDIAN;
    o.direction = N4M_OPT_MINIMIZE;
    o.n_startup_trials = 2;
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    n4m_trial_t* t[3] = {nullptr, nullptr, nullptr};
    int64_t id[3];
    for (int i = 0; i < 3; ++i) {
        n4m_optimizer_ask(opt, &t[i]);
        n4m_trial_get_id(t[i], &id[i]);
    }
    int32_t pr = 0;
    n4m_optimizer_tell_intermediate(opt, id[0], 0, 1.0, &pr);
    n4m_optimizer_tell_intermediate(opt, id[1], 0, 2.0, &pr);
    n4m_optimizer_tell_intermediate(opt, id[2], 0, 9.0, &pr);  // t2 pruned
    N4M_TEST_REQUIRE(pr == 1);
    pr = 0;
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 0, 9.0, &pr) == N4M_OK);
    N4M_TEST_REQUIRE(pr == 1);  // exact at-least-once replay preserves the prune decision
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 0, 8.0, &pr) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(pr == 0);
    // a pruned trial is terminal: it cannot be completed, and further rungs are rejected
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, id[2], 0.0) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id[2], 1, 0.0, &pr) == N4M_ERR_INVALID_ARGUMENT);
    // best() ignores the pruned trial even though its (rejected) score 0.0 would have won
    n4m_optimizer_tell(opt, id[0], 1.0);
    n4m_optimizer_tell(opt, id[1], 2.0);
    n4m_trial_t* best = nullptr;
    double bs = 1e9;
    n4m_optimizer_best(opt, &best, &bs);
    N4M_TEST_REQUIRE(bs == 1.0);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_terminal_status_and_intermediate_step_validation() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 3, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* trial = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &trial) == N4M_OK);
    int64_t id = -1;
    n4m_trial_get_id(trial, &id);

    int32_t prune = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id, -1, 1.0, &prune) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, id, N4M_TRIAL_RUNNING, 0.0, nullptr) ==
                     N4M_ERR_INVALID_ARGUMENT);
#if !defined(N4M_UBSAN_BUILD)
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, id, invalid_enum<n4m_trial_status_t>(99), 0.0, nullptr) ==
                     N4M_ERR_INVALID_ARGUMENT);
#endif
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, id, N4M_TRIAL_COMPLETED, 1.0, nullptr) == N4M_OK);

    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_cancelled_structured_error_and_idempotence() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 3, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);

    n4m_trial_t* failed = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &failed) == N4M_OK);
    int64_t failed_id = -1;
    n4m_trial_get_id(failed, &failed_id);
    const char invalid_utf8[] = {static_cast<char>(0xFF), '\0'};
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, failed_id, N4M_TRIAL_FAILED, 0.0,
                         invalid_utf8) == N4M_ERR_INVALID_ARGUMENT);
    n4m_trial_status_t current = N4M_TRIAL_FAILED;
    N4M_TEST_REQUIRE(n4m_trial_get_status(failed, &current) == N4M_OK);
    N4M_TEST_REQUIRE(current == N4M_TRIAL_RUNNING);
    int32_t prune = -1;
    N4M_TEST_REQUIRE(
        n4m_optimizer_tell_intermediate(opt, failed_id, 0, 4.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(prune == 0);
    // Exact replay is idempotent. A rewrite or out-of-order insert is rejected.
    N4M_TEST_REQUIRE(
        n4m_optimizer_tell_intermediate(opt, failed_id, 0, 4.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, failed_id, 0, 5.0, &prune) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(
        n4m_optimizer_tell_intermediate(opt, failed_id, 2, 3.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, failed_id, 1, 2.0, &prune) ==
                     N4M_ERR_INVALID_ARGUMENT);

    const char* failed_error =
        "n4m.error.v1|OBJECTIVE_EXCEPTION|0|deterministic failure|with separator";
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, failed_id, N4M_TRIAL_FAILED, 0.0, failed_error) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, failed_id, N4M_TRIAL_FAILED, 0.0, failed_error) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, failed_id, N4M_TRIAL_FAILED, 0.0,
                         "n4m.error.v1|OBJECTIVE_EXCEPTION|0|rewritten") ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, 999, N4M_TRIAL_FAILED, 0.0, failed_error) ==
                     N4M_ERR_INVALID_ARGUMENT);

    n4m_trial_t* cancelled = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &cancelled) == N4M_OK);
    int64_t cancelled_id = -1;
    n4m_trial_get_id(cancelled, &cancelled_id);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, cancelled_id, N4M_TRIAL_CANCELLED, 0.0,
                         "n4m.error.v2|BUDGET_CANCELLED|1|future") ==
                     N4M_ERR_INVALID_ARGUMENT);
    current = N4M_TRIAL_FAILED;
    N4M_TEST_REQUIRE(n4m_trial_get_status(cancelled, &current) == N4M_OK);
    N4M_TEST_REQUIRE(current == N4M_TRIAL_RUNNING);
    const char* cancelled_error =
        "n4m.error.v1|BUDGET_CANCELLED|1|study timeout";
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, cancelled_id, N4M_TRIAL_CANCELLED, 0.0,
                         cancelled_error) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, cancelled_id, N4M_TRIAL_CANCELLED, 0.0,
                         cancelled_error) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, cancelled_id, N4M_TRIAL_CANCELLED, 0.0,
                         "n4m.error.v1|BUDGET_CANCELLED|0|study timeout") ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_trial_get_status(cancelled, &current) == N4M_OK);
    N4M_TEST_REQUIRE(current == N4M_TRIAL_CANCELLED);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, cancelled_id, 1.0) ==
                     N4M_ERR_INVALID_ARGUMENT);

    n4m_trial_t* completed = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &completed) == N4M_OK);
    int64_t completed_id = -1;
    n4m_trial_get_id(completed, &completed_id);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, completed_id, N4M_TRIAL_COMPLETED, 1.0, "not allowed") ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, completed_id, 1.0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, completed_id, 1.0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, completed_id, 2.0) ==
                     N4M_ERR_INVALID_ARGUMENT);

    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_trace_utf8_boundaries() {
    const char invalid_utf8[] = {static_cast<char>(0xC0),
                                 static_cast<char>(0xAF), '\0'};
    n4m_search_space_t* space = nullptr;
    N4M_TEST_REQUIRE(n4m_search_space_create(&space) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_int(
                         space, invalid_utf8, 1, 3, 1, 0) ==
                     N4M_ERR_INVALID_ARGUMENT);
    const char* invalid_choices[] = {"valid", invalid_utf8};
    N4M_TEST_REQUIRE(n4m_search_space_add_categorical(
                         space, "kind", N4M_CAT_STR, invalid_choices, 2) ==
                     N4M_ERR_INVALID_ARGUMENT);
    int32_t n_params = -1;
    N4M_TEST_REQUIRE(n4m_search_space_num_params(space, &n_params) == N4M_OK);
    N4M_TEST_REQUIRE(n_params == 0);

    const char* valid_choices[] = {"caf\xC3\xA9", "th\xC3\xA9"};
    N4M_TEST_REQUIRE(n4m_search_space_add_categorical(
                         space, "kind", N4M_CAT_STR, valid_choices, 2) == N4M_OK);
    n4m_search_space_destroy(space);
}

void test_rich_trial_trace_since_id_and_ownership() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 3, 1, 0);
    const char* choices[] = {"alpha", "beta"};
    n4m_search_space_add_categorical(sp, "kind", N4M_CAT_STR, choices, 2);
    n4m_optimizer_options_t o = default_opts();
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);

    n4m_method_result_t* before_ask = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_get_trials(opt, 0, &before_ask) == N4M_OK);
    double before_n_trials = -1.0;
    double before_n_params = -1.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(
                         before_ask, "n_trials", &before_n_trials) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(
                         before_ask, "n_params", &before_n_params) == N4M_OK);
    N4M_TEST_REQUIRE(before_n_trials == 0.0);
    N4M_TEST_REQUIRE(before_n_params == 2.0);
    const int64_t* before_name_offsets = nullptr;
    int64_t n_before_name_offsets = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         before_ask, "trial_param_name_offsets",
                         &before_name_offsets, &n_before_name_offsets) == N4M_OK);
    N4M_TEST_REQUIRE(n_before_name_offsets == 3);
    n4m_method_result_destroy(before_ask);

    n4m_trial_t* ignored = nullptr;
    n4m_trial_t* cancelled = nullptr;
    n4m_trial_t* completed = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &ignored) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &cancelled) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &completed) == N4M_OK);
    int64_t ignored_id = -1;
    int64_t cancelled_id = -1;
    int64_t completed_id = -1;
    n4m_trial_get_id(ignored, &ignored_id);
    n4m_trial_get_id(cancelled, &cancelled_id);
    n4m_trial_get_id(completed, &completed_id);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, ignored_id, 9.0) == N4M_OK);
    char cancelled_wire[] =
        "n4m.error.v1|BUDGET_CANCELLED|1|budget exhausted";
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         opt, cancelled_id, N4M_TRIAL_CANCELLED, 0.0,
                         cancelled_wire) == N4M_OK);
    std::memset(cancelled_wire, 'x', sizeof(cancelled_wire) - 1);
    int32_t prune = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(
                         opt, completed_id, 0, 2.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(
                         opt, completed_id, 1, 1.0, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, completed_id, 1.0) == N4M_OK);

    n4m_method_result_t* invalid =
        reinterpret_cast<n4m_method_result_t*>(static_cast<std::uintptr_t>(1));
    N4M_TEST_REQUIRE(n4m_optimizer_get_trials(opt, -1, &invalid) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(invalid == nullptr);
    n4m_method_result_t* empty = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_get_trials(opt, 99, &empty) == N4M_OK);
    const int64_t* empty_ids = nullptr;
    int64_t empty_count = -1;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         empty, "trial_ids_i64", &empty_ids, &empty_count) == N4M_OK);
    N4M_TEST_REQUIRE(empty_count == 0);
    n4m_method_result_destroy(empty);

    n4m_method_result_t* trace = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_get_trials(opt, cancelled_id, &trace) == N4M_OK);
    // The result owns every buffer. Destroy every producer before reading it.
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);

    double format_version = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(
                         trace, "trace_format_version", &format_version) == N4M_OK);
    N4M_TEST_REQUIRE(format_version == 1.0);
    const int64_t* ids = nullptr;
    int64_t n_ids = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_ids_i64", &ids, &n_ids) == N4M_OK);
    N4M_TEST_REQUIRE(n_ids == 2);
    N4M_TEST_REQUIRE(ids[0] == cancelled_id);
    N4M_TEST_REQUIRE(ids[1] == completed_id);
    const int64_t* ask_sequences = nullptr;
    int64_t n_ask_sequences = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_ask_sequence", &ask_sequences,
                         &n_ask_sequences) == N4M_OK);
    N4M_TEST_REQUIRE(n_ask_sequences == 2);
    N4M_TEST_REQUIRE(ask_sequences[0] == 1 && ask_sequences[1] == 2);
    const int64_t* terminal_sequences = nullptr;
    int64_t n_terminal_sequences = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_terminal_sequence", &terminal_sequences,
                         &n_terminal_sequences) == N4M_OK);
    N4M_TEST_REQUIRE(n_terminal_sequences == 2);
    N4M_TEST_REQUIRE(terminal_sequences[0] == 4 && terminal_sequences[1] == 7);

    const double* statuses = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         trace, "trial_status", &statuses, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == 1 && cols == 2);
    N4M_TEST_REQUIRE(statuses[0] == static_cast<double>(N4M_TRIAL_CANCELLED));
    N4M_TEST_REQUIRE(statuses[1] == static_cast<double>(N4M_TRIAL_COMPLETED));

    const double* param_values = nullptr;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         trace, "trial_param_values", &param_values, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == 2 && cols == 2);
    N4M_TEST_REQUIRE(std::isfinite(param_values[0]));
    N4M_TEST_REQUIRE(std::isfinite(param_values[3]));
    const int32_t* param_kinds = nullptr;
    int32_t n_param_kinds = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         trace, "trial_param_kind", &param_kinds,
                         &n_param_kinds) == N4M_OK);
    N4M_TEST_REQUIRE(n_param_kinds == 2);
    N4M_TEST_REQUIRE(param_kinds[0] == static_cast<int32_t>(N4M_PARAM_INT));
    N4M_TEST_REQUIRE(param_kinds[1] == static_cast<int32_t>(N4M_PARAM_CATEGORICAL));
    const int32_t* param_category_types = nullptr;
    int32_t n_param_category_types = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         trace, "trial_param_category_type", &param_category_types,
                         &n_param_category_types) == N4M_OK);
    N4M_TEST_REQUIRE(n_param_category_types == 2);
    N4M_TEST_REQUIRE(param_category_types[0] == -1);
    N4M_TEST_REQUIRE(param_category_types[1] == static_cast<int32_t>(N4M_CAT_STR));

    const int32_t* name_bytes = nullptr;
    int32_t n_name_bytes = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         trace, "trial_param_name_utf8", &name_bytes,
                         &n_name_bytes) == N4M_OK);
    const int64_t* name_offsets = nullptr;
    int64_t n_name_offsets = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_param_name_offsets", &name_offsets,
                         &n_name_offsets) == N4M_OK);
    N4M_TEST_REQUIRE(n_name_offsets == 3);
    std::string first_name;
    for (int64_t index = name_offsets[0]; index < name_offsets[1]; ++index) {
        first_name.push_back(static_cast<char>(name_bytes[index]));
    }
    N4M_TEST_REQUIRE(first_name == "k");

    const int64_t* intermediate_offsets = nullptr;
    int64_t n_intermediate_offsets = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_intermediate_offsets", &intermediate_offsets,
                         &n_intermediate_offsets) == N4M_OK);
    N4M_TEST_REQUIRE(n_intermediate_offsets == 3);
    N4M_TEST_REQUIRE(intermediate_offsets[0] == 0);
    N4M_TEST_REQUIRE(intermediate_offsets[1] == 0);
    N4M_TEST_REQUIRE(intermediate_offsets[2] == 2);
    const int64_t* intermediate_sequences = nullptr;
    int64_t n_intermediate_sequences = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_intermediate_sequence", &intermediate_sequences,
                         &n_intermediate_sequences) == N4M_OK);
    N4M_TEST_REQUIRE(n_intermediate_sequences == 2);
    N4M_TEST_REQUIRE(intermediate_sequences[0] == 5);
    N4M_TEST_REQUIRE(intermediate_sequences[1] == 6);

    const int32_t* code_bytes = nullptr;
    int32_t n_code_bytes = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         trace, "trial_error_code_utf8", &code_bytes,
                         &n_code_bytes) == N4M_OK);
    const int64_t* code_offsets = nullptr;
    int64_t n_code_offsets = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_error_code_offsets", &code_offsets,
                         &n_code_offsets) == N4M_OK);
    N4M_TEST_REQUIRE(n_code_offsets == 3);
    std::string first_code;
    for (int64_t index = code_offsets[0]; index < code_offsets[1]; ++index) {
        first_code.push_back(static_cast<char>(code_bytes[index]));
    }
    N4M_TEST_REQUIRE(first_code == "BUDGET_CANCELLED");
    N4M_TEST_REQUIRE(code_offsets[2] == code_offsets[1]);
    const int32_t* message_bytes = nullptr;
    int32_t n_message_bytes = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         trace, "trial_error_message_utf8", &message_bytes,
                         &n_message_bytes) == N4M_OK);
    const int64_t* message_offsets = nullptr;
    int64_t n_message_offsets = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
                         trace, "trial_error_message_offsets", &message_offsets,
                         &n_message_offsets) == N4M_OK);
    N4M_TEST_REQUIRE(n_message_offsets == 3);
    std::string first_message;
    for (int64_t index = message_offsets[0]; index < message_offsets[1]; ++index) {
        first_message.push_back(static_cast<char>(message_bytes[index]));
    }
    N4M_TEST_REQUIRE(first_message == "budget exhausted");
    N4M_TEST_REQUIRE(message_offsets[2] == message_offsets[1]);

    const int32_t* retryable = nullptr;
    int32_t n_retryable = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         trace, "trial_error_retryable", &retryable,
                         &n_retryable) == N4M_OK);
    N4M_TEST_REQUIRE(n_retryable == 2);
    N4M_TEST_REQUIRE(retryable[0] == 1 && retryable[1] == 0);
    n4m_method_result_destroy(trace);
}

void test_invalid_pruner_and_nan() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    set_invalid_enum(o.pruner, 99);  // out-of-range → NOT_IMPLEMENTED
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_ERR_NOT_IMPLEMENTED);
    N4M_TEST_REQUIRE(opt == nullptr);
    o.pruner = N4M_PRUNER_MEDIAN;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    n4m_trial_t* t = nullptr;
    int64_t id = 0;
    n4m_optimizer_ask(opt, &t);
    n4m_trial_get_id(t, &id);
    int32_t pr = 0;
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(opt, id, 0, std::nan(""), &pr) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(opt, id, std::nan("")) == N4M_ERR_INVALID_ARGUMENT);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

void test_lhs_stratifies() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_LHS;
    o.n_startup_trials = 10;
    o.seed = 4;
    n4m_optimizer_t* opt = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
    bool bins[10] = {false, false, false, false, false, false, false, false, false, false};
    for (int i = 0; i < 10; ++i) {
        n4m_trial_t* t = nullptr;
        n4m_optimizer_ask(opt, &t);
        double x = 0.0;
        n4m_trial_get_float(t, "x", &x);
        int b = static_cast<int>(x * 10.0);
        if (b < 0) b = 0;
        if (b > 9) b = 9;
        N4M_TEST_REQUIRE(!bins[b]);  // Latin-hypercube: each decile hit exactly once
        bins[b] = true;
        int64_t id = 0;
        n4m_trial_get_id(t, &id);
        n4m_optimizer_tell(opt, id, x);
    }
    for (int b = 0; b < 10; ++b) N4M_TEST_REQUIRE(bins[b]);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
    n4m_context_destroy(ctx);
}

std::uint32_t checkpoint_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    N4M_TEST_REQUIRE(offset + 4 <= bytes.size());
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(bytes[offset + i]) << (8U * i);
    }
    return value;
}

std::uint64_t checkpoint_u64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    N4M_TEST_REQUIRE(offset + 8 <= bytes.size());
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8U * i);
    }
    return value;
}

void checkpoint_write_u32(std::vector<std::uint8_t>& bytes, std::size_t offset,
                          std::uint32_t value) {
    N4M_TEST_REQUIRE(offset + 4 <= bytes.size());
    for (std::size_t i = 0; i < 4; ++i) {
        bytes[offset + i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
    }
}

void checkpoint_write_u64(std::vector<std::uint8_t>& bytes, std::size_t offset,
                          std::uint64_t value) {
    N4M_TEST_REQUIRE(offset + 8 <= bytes.size());
    for (std::size_t i = 0; i < 8; ++i) {
        bytes[offset + i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
    }
}

std::uint64_t checkpoint_fnv1a(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void checkpoint_refresh_checksum(std::vector<std::uint8_t>& bytes) {
    N4M_TEST_REQUIRE(bytes.size() >= 8);
    const std::size_t offset = bytes.size() - 8;
    const std::uint64_t hash = checkpoint_fnv1a(bytes, offset);
    for (std::size_t i = 0; i < 8; ++i) {
        bytes[offset + i] = static_cast<std::uint8_t>((hash >> (8U * i)) & 0xffU);
    }
}

std::vector<std::uint8_t> save_checkpoint(n4m_optimizer_t* optimizer) {
    n4m_array_t* array = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_save(optimizer, &array) == N4M_OK);
    N4M_TEST_REQUIRE(array != nullptr);
    n4m_matrix_view_t view{};
    N4M_TEST_REQUIRE(n4m_array_view(array, &view) == N4M_OK);
    N4M_TEST_REQUIRE(view.dtype == N4M_DTYPE_I64);
    N4M_TEST_REQUIRE(view.rows == 1 && view.cols > 0);
    const std::size_t size = static_cast<std::size_t>(view.cols) * sizeof(std::uint64_t);
    const auto* first = static_cast<const std::uint8_t*>(view.data);
    std::vector<std::uint8_t> result(first, first + size);
    n4m_array_free(array);
    return result;
}

void test_ask_batch_checkpoint_and_determinism() {
    // MT12 — ask_batch(n) is exactly n sequential ask() calls with no
    // intervening tell (constant-liar stays NONE), and a checkpoint saved AT a
    // generation boundary restores that boundary (same stable status) and
    // continues. Covers GA, PSO and CMA-ES. This is a same-process ordered
    // reproducibility claim, not a wall-clock parallel-completion claim.
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    const n4m_sampler_kind_t pop_samplers[3] = {
        N4M_SAMPLER_GA, N4M_SAMPLER_PSO, N4M_SAMPLER_CMAES};
    auto make = [&](n4m_sampler_kind_t sampler) -> n4m_optimizer_t* {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_float(sp, "x", -5.0, 5.0, 0.0, 0);
        n4m_search_space_add_float(sp, "y", -5.0, 5.0, 0.0, 0);
        n4m_optimizer_options_t o = default_opts();
        o.sampler = sampler;
        o.direction = N4M_OPT_MINIMIZE;
        o.seed = 42;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        n4m_search_space_destroy(sp);
        return opt;
    };
    auto xy = [](n4m_trial_t* t, double* x, double* y) {
        N4M_TEST_REQUIRE(n4m_trial_get_float(t, "x", x) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_float(t, "y", y) == N4M_OK);
    };
    for (const auto sampler : pop_samplers) {
        // (1) ask_batch is equivalent to sequential asks with NO intervening
        // tell. After all asks, replay the same non-trivial TERMINAL-event order
        // and score payloads on both optimizers; the next generation is bit-exact.
        n4m_optimizer_t* batched = make(sampler);
        n4m_optimizer_t* seq = make(sampler);
        n4m_trial_t* bbuf[512] = {nullptr};
        int32_t bcount = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(batched, 512, bbuf, &bcount) == N4M_OK);
        N4M_TEST_REQUIRE(bcount > 0 && bcount < 512);
        const int32_t pop = bcount;
        for (int gen = 0; gen < 2; ++gen) {
            if (gen == 1) {  // second generation: batched advances, seq mirrors
                int32_t bcount2 = -1;
                N4M_TEST_REQUIRE(
                    n4m_optimizer_ask_batch(batched, 512, bbuf, &bcount2) == N4M_OK);
                N4M_TEST_REQUIRE(bcount2 == pop);
            }
            std::vector<n4m_trial_t*> sequential(static_cast<std::size_t>(pop), nullptr);
            for (int32_t i = 0; i < pop; ++i) {
                N4M_TEST_REQUIRE(
                    n4m_optimizer_ask(seq, &sequential[static_cast<std::size_t>(i)]) ==
                    N4M_OK);
                double bx = 0, by = 0, sx = 0, sy = 0;
                xy(bbuf[i], &bx, &by);
                xy(sequential[static_cast<std::size_t>(i)], &sx, &sy);
                N4M_TEST_REQUIRE(bx == sx && by == sy);  // bit-exact continuation
                int64_t bid = 0, sid = 0;
                n4m_trial_get_id(bbuf[i], &bid);
                n4m_trial_get_id(sequential[static_cast<std::size_t>(i)], &sid);
                N4M_TEST_REQUIRE(bid == sid);
            }
            // Tell in reverse ask order to make the normative event-order
            // requirement observable rather than relying on ascending ids.
            for (int32_t offset = 0; offset < pop; ++offset) {
                const int32_t i = pop - 1 - offset;
                int64_t bid = -1;
                int64_t sid = -1;
                n4m_trial_get_id(bbuf[i], &bid);
                n4m_trial_get_id(sequential[static_cast<std::size_t>(i)], &sid);
                const double score = 0.25 * static_cast<double>(i + 1 + gen * pop);
                N4M_TEST_REQUIRE(n4m_optimizer_tell(batched, bid, score) == N4M_OK);
                N4M_TEST_REQUIRE(n4m_optimizer_tell(seq, sid, score) == N4M_OK);
            }
        }
        n4m_optimizer_destroy(seq);
        n4m_optimizer_destroy(batched);

        // (2) checkpoint roundtrip AT the boundary.
        n4m_optimizer_t* opt = make(sampler);
        n4m_trial_t* buf[512] = {nullptr};
        int32_t count = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(opt, 512, buf, &count) == N4M_OK);
        N4M_TEST_REQUIRE(count == pop);
        std::vector<std::uint8_t> blob = save_checkpoint(opt);  // saved AT the boundary
        n4m_optimizer_destroy(opt);

        n4m_optimizer_t* restored = nullptr;
        N4M_TEST_REQUIRE(
            n4m_optimizer_load(ctx, blob.data(), blob.size(), &restored) == N4M_OK);
        // Boundary survived the roundtrip: zero-capacity retry is INVALID_ARGUMENT/0.
        n4m_trial_t* rbuf[512] = {nullptr};
        int32_t rcount = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(restored, 4, rbuf, &rcount) ==
                         N4M_ERR_INVALID_ARGUMENT);
        N4M_TEST_REQUIRE(rcount == 0);
        for (int32_t i = 0; i < pop; ++i) {
            N4M_TEST_REQUIRE(
                n4m_optimizer_tell(restored, i, 0.25 * static_cast<double>(i + 1)) == N4M_OK);
        }
        int32_t rnext = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(restored, 512, rbuf, &rnext) == N4M_OK);
        N4M_TEST_REQUIRE(rnext == pop);  // restored copy advances one full generation
        n4m_optimizer_destroy(restored);
    }

    // (3) Completed-history samplers: the same ordered space, seed, ask stream,
    // score payloads and exact TERMINAL-event order give a bit-exact continuation.
    // This deliberately says nothing about arbitrary wall-clock completion
    // order; hosts must record and replay the event order they actually used.
    const n4m_sampler_kind_t adaptive_samplers[3] = {
        N4M_SAMPLER_TERNARY, N4M_SAMPLER_TPE, N4M_SAMPLER_GP_EI};
    auto make_adaptive = [&](n4m_sampler_kind_t sampler) -> n4m_optimizer_t* {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "k", 0, 100, 1, 0);
        n4m_search_space_add_float(sp, "x", -1.0, 1.0, 0.0, 0);
        n4m_optimizer_options_t o = default_opts();
        o.sampler = sampler;
        o.direction = N4M_OPT_MINIMIZE;
        o.n_startup_trials = 4;
        o.seed = 0x1234abcdULL;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        n4m_search_space_destroy(sp);
        return opt;
    };
    for (const auto sampler : adaptive_samplers) {
        n4m_optimizer_t* batched = make_adaptive(sampler);
        n4m_optimizer_t* sequential_opt = make_adaptive(sampler);
        constexpr int32_t first_size = 12;
        n4m_trial_t* first_batch[first_size] = {nullptr};
        int32_t first_count = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(
                             batched, first_size, first_batch, &first_count) == N4M_OK);
        N4M_TEST_REQUIRE(first_count == first_size);
        std::vector<n4m_trial_t*> first_sequential(first_size, nullptr);
        std::vector<double> scores(first_size, 0.0);
        for (int32_t i = 0; i < first_size; ++i) {
            N4M_TEST_REQUIRE(
                n4m_optimizer_ask(
                    sequential_opt, &first_sequential[static_cast<std::size_t>(i)]) ==
                N4M_OK);
            int64_t batch_id = -1;
            int64_t sequential_id = -1;
            int64_t batch_k = -1;
            int64_t sequential_k = -1;
            double batch_x = 0.0;
            double sequential_x = 0.0;
            n4m_trial_get_id(first_batch[i], &batch_id);
            n4m_trial_get_id(first_sequential[static_cast<std::size_t>(i)],
                             &sequential_id);
            n4m_trial_get_int(first_batch[i], "k", &batch_k);
            n4m_trial_get_int(first_sequential[static_cast<std::size_t>(i)], "k",
                              &sequential_k);
            n4m_trial_get_float(first_batch[i], "x", &batch_x);
            n4m_trial_get_float(first_sequential[static_cast<std::size_t>(i)], "x",
                                &sequential_x);
            N4M_TEST_REQUIRE(batch_id == sequential_id);
            N4M_TEST_REQUIRE(batch_k == sequential_k && batch_x == sequential_x);
            const double dk = static_cast<double>(batch_k - 17);
            scores[static_cast<std::size_t>(i)] = dk * dk + batch_x * batch_x;
        }
        for (int32_t offset = 0; offset < first_size; ++offset) {
            const int32_t i = first_size - 1 - offset;
            int64_t batch_id = -1;
            int64_t sequential_id = -1;
            n4m_trial_get_id(first_batch[i], &batch_id);
            n4m_trial_get_id(first_sequential[static_cast<std::size_t>(i)],
                             &sequential_id);
            const double score = scores[static_cast<std::size_t>(i)];
            N4M_TEST_REQUIRE(n4m_optimizer_tell(batched, batch_id, score) == N4M_OK);
            N4M_TEST_REQUIRE(
                n4m_optimizer_tell(sequential_opt, sequential_id, score) == N4M_OK);
        }

        constexpr int32_t continuation_size = 6;
        n4m_trial_t* continuation_batch[continuation_size] = {nullptr};
        int32_t continuation_count = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(
                             batched, continuation_size, continuation_batch,
                             &continuation_count) == N4M_OK);
        N4M_TEST_REQUIRE(continuation_count == continuation_size);
        std::vector<n4m_trial_t*> continuation_sequential(continuation_size, nullptr);
        for (int32_t i = 0; i < continuation_size; ++i) {
            N4M_TEST_REQUIRE(
                n4m_optimizer_ask(
                    sequential_opt,
                    &continuation_sequential[static_cast<std::size_t>(i)]) == N4M_OK);
            int64_t batch_k = -1;
            int64_t sequential_k = -1;
            double batch_x = 0.0;
            double sequential_x = 0.0;
            n4m_trial_get_int(continuation_batch[i], "k", &batch_k);
            n4m_trial_get_int(
                continuation_sequential[static_cast<std::size_t>(i)], "k",
                &sequential_k);
            n4m_trial_get_float(continuation_batch[i], "x", &batch_x);
            n4m_trial_get_float(
                continuation_sequential[static_cast<std::size_t>(i)], "x",
                &sequential_x);
            N4M_TEST_REQUIRE(batch_k == sequential_k && batch_x == sequential_x);
        }
        for (int32_t i = 0; i < continuation_size; ++i) {
            int64_t batch_id = -1;
            int64_t sequential_id = -1;
            n4m_trial_get_id(continuation_batch[i], &batch_id);
            n4m_trial_get_id(
                continuation_sequential[static_cast<std::size_t>(i)],
                &sequential_id);
            const double score = static_cast<double>(i);
            N4M_TEST_REQUIRE(n4m_optimizer_tell(batched, batch_id, score) == N4M_OK);
            N4M_TEST_REQUIRE(
                n4m_optimizer_tell(sequential_opt, sequential_id, score) == N4M_OK);
        }
        n4m_optimizer_destroy(sequential_opt);
        n4m_optimizer_destroy(batched);
    }

    // (4) The normative stream includes INTERMEDIATE events and their payloads,
    // not merely terminal tells: pruners consume the ordered intermediate
    // history. Replay the same non-trivial ASK / INTERMEDIATE / TERMINAL stream
    // against TPE + median pruning and require the decisions, states and adaptive
    // continuation to remain bit-exact.
    auto make_pruned_adaptive = [&]() -> n4m_optimizer_t* {
        n4m_search_space_t* sp = nullptr;
        n4m_search_space_create(&sp);
        n4m_search_space_add_int(sp, "k", 0, 100, 1, 0);
        n4m_search_space_add_float(sp, "x", -1.0, 1.0, 0.0, 0);
        n4m_optimizer_options_t o = default_opts();
        o.sampler = N4M_SAMPLER_TPE;
        o.pruner = N4M_PRUNER_MEDIAN;
        o.direction = N4M_OPT_MINIMIZE;
        o.n_startup_trials = 2;
        o.seed = 0x987654321ULL;
        n4m_optimizer_t* opt = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(ctx, sp, &o, &opt) == N4M_OK);
        n4m_search_space_destroy(sp);
        return opt;
    };
    n4m_optimizer_t* pruned_batch = make_pruned_adaptive();
    n4m_optimizer_t* pruned_sequential = make_pruned_adaptive();
    constexpr int32_t pruned_stream_size = 4;
    n4m_trial_t* pruned_batch_trials[pruned_stream_size] = {nullptr};
    int32_t pruned_batch_count = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(
                         pruned_batch, pruned_stream_size, pruned_batch_trials,
                         &pruned_batch_count) == N4M_OK);
    N4M_TEST_REQUIRE(pruned_batch_count == pruned_stream_size);
    n4m_trial_t* pruned_sequential_trials[pruned_stream_size] = {nullptr};
    int64_t pruned_batch_ids[pruned_stream_size] = {-1, -1, -1, -1};
    int64_t pruned_sequential_ids[pruned_stream_size] = {-1, -1, -1, -1};
    for (int32_t i = 0; i < pruned_stream_size; ++i) {
        N4M_TEST_REQUIRE(
            n4m_optimizer_ask(pruned_sequential, &pruned_sequential_trials[i]) ==
            N4M_OK);
        int64_t batch_k = -1;
        int64_t sequential_k = -1;
        double batch_x = 0.0;
        double sequential_x = 0.0;
        N4M_TEST_REQUIRE(
            n4m_trial_get_id(pruned_batch_trials[i], &pruned_batch_ids[i]) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_id(
                             pruned_sequential_trials[i], &pruned_sequential_ids[i]) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(pruned_batch_ids[i] == pruned_sequential_ids[i]);
        N4M_TEST_REQUIRE(
            n4m_trial_get_int(pruned_batch_trials[i], "k", &batch_k) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_int(
                             pruned_sequential_trials[i], "k", &sequential_k) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(
            n4m_trial_get_float(pruned_batch_trials[i], "x", &batch_x) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_float(
                             pruned_sequential_trials[i], "x", &sequential_x) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(batch_k == sequential_k && batch_x == sequential_x);
    }

    constexpr int32_t intermediate_order[pruned_stream_size] = {1, 0, 2, 3};
    constexpr double intermediate_scores[pruned_stream_size] = {1.0, 2.0, 5.0,
                                                                 0.5};
    constexpr int32_t expected_prune[pruned_stream_size] = {0, 0, 1, 0};
    for (const int32_t i : intermediate_order) {
        int32_t batch_decision = -1;
        int32_t sequential_decision = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(
                             pruned_batch, pruned_batch_ids[i], 0,
                             intermediate_scores[i], &batch_decision) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(
                             pruned_sequential, pruned_sequential_ids[i], 0,
                             intermediate_scores[i], &sequential_decision) == N4M_OK);
        N4M_TEST_REQUIRE(batch_decision == sequential_decision);
        N4M_TEST_REQUIRE(batch_decision == expected_prune[i]);
        n4m_trial_status_t batch_status = N4M_TRIAL_FAILED;
        n4m_trial_status_t sequential_status = N4M_TRIAL_FAILED;
        N4M_TEST_REQUIRE(
            n4m_trial_get_status(pruned_batch_trials[i], &batch_status) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_status(
                             pruned_sequential_trials[i], &sequential_status) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(batch_status == sequential_status);
        N4M_TEST_REQUIRE(batch_status ==
                         (expected_prune[i] ? N4M_TRIAL_PRUNED : N4M_TRIAL_RUNNING));
    }

    constexpr int32_t terminal_order[pruned_stream_size - 1] = {3, 0, 1};
    constexpr double terminal_scores[pruned_stream_size] = {0.75, 1.75, 0.0, 0.25};
    for (const int32_t i : terminal_order) {
        N4M_TEST_REQUIRE(n4m_optimizer_tell(
                             pruned_batch, pruned_batch_ids[i], terminal_scores[i]) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(n4m_optimizer_tell(
                             pruned_sequential, pruned_sequential_ids[i],
                             terminal_scores[i]) == N4M_OK);
    }

    constexpr int32_t pruned_continuation_size = 4;
    n4m_trial_t* pruned_continuation_batch[pruned_continuation_size] = {nullptr};
    int32_t pruned_continuation_count = -1;
    N4M_TEST_REQUIRE(n4m_optimizer_ask_batch(
                         pruned_batch, pruned_continuation_size,
                         pruned_continuation_batch, &pruned_continuation_count) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(pruned_continuation_count == pruned_continuation_size);
    n4m_trial_t* pruned_continuation_sequential[pruned_continuation_size] = {nullptr};
    for (int32_t i = 0; i < pruned_continuation_size; ++i) {
        N4M_TEST_REQUIRE(n4m_optimizer_ask(
                             pruned_sequential,
                             &pruned_continuation_sequential[i]) == N4M_OK);
        int64_t batch_id = -1;
        int64_t sequential_id = -1;
        int64_t batch_k = -1;
        int64_t sequential_k = -1;
        double batch_x = 0.0;
        double sequential_x = 0.0;
        N4M_TEST_REQUIRE(
            n4m_trial_get_id(pruned_continuation_batch[i], &batch_id) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_id(
                             pruned_continuation_sequential[i], &sequential_id) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(batch_id == sequential_id);
        N4M_TEST_REQUIRE(n4m_trial_get_int(
                             pruned_continuation_batch[i], "k", &batch_k) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_int(
                             pruned_continuation_sequential[i], "k", &sequential_k) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_float(
                             pruned_continuation_batch[i], "x", &batch_x) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_float(
                             pruned_continuation_sequential[i], "x", &sequential_x) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(batch_k == sequential_k && batch_x == sequential_x);
    }
    for (int32_t i = 0; i < pruned_continuation_size; ++i) {
        int64_t batch_id = -1;
        int64_t sequential_id = -1;
        n4m_trial_get_id(pruned_continuation_batch[i], &batch_id);
        n4m_trial_get_id(pruned_continuation_sequential[i], &sequential_id);
        const double score = static_cast<double>(i + 10);
        N4M_TEST_REQUIRE(n4m_optimizer_tell(pruned_batch, batch_id, score) == N4M_OK);
        N4M_TEST_REQUIRE(
            n4m_optimizer_tell(pruned_sequential, sequential_id, score) == N4M_OK);
    }
    n4m_optimizer_destroy(pruned_sequential);
    n4m_optimizer_destroy(pruned_batch);
    n4m_context_destroy(ctx);
}

void test_checkpoint_all_sampler_trajectories() {
    const n4m_sampler_kind_t samplers[] = {
        N4M_SAMPLER_RANDOM, N4M_SAMPLER_SOBOL, N4M_SAMPLER_LHS,
        N4M_SAMPLER_TERNARY, N4M_SAMPLER_GA, N4M_SAMPLER_PSO,
        N4M_SAMPLER_CMAES, N4M_SAMPLER_TPE, N4M_SAMPLER_GP_EI,
    };
    n4m_context_t* context = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    for (const n4m_sampler_kind_t sampler : samplers) {
        n4m_search_space_t* space = nullptr;
        N4M_TEST_REQUIRE(n4m_search_space_create(&space) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_search_space_add_int(space, "k", 1, 7, 1, 0) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_search_space_add_float(space, "x", -2.0, 2.0, 0.0, 0) == N4M_OK);
        n4m_optimizer_options_t options = default_opts();
        options.sampler = sampler;
        options.n_startup_trials = 2;
        options.seed = 0x123456789ULL;
        n4m_optimizer_t* original = nullptr;
        N4M_TEST_REQUIRE(n4m_optimizer_create(context, space, &options, &original) == N4M_OK);
        // End exactly on a completed generation/swarm boundary for population
        // samplers. The continuation below must therefore re-evolve from the
        // restored adaptive state rather than merely consume cached members.
        const int prefix_trials =
            (sampler == N4M_SAMPLER_GA || sampler == N4M_SAMPLER_PSO)
                ? 32
                : (sampler == N4M_SAMPLER_CMAES ? 24 : 20);
        for (int i = 0; i < prefix_trials; ++i) {
            n4m_trial_t* trial = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(original, &trial) == N4M_OK);
            std::int64_t id = -1;
            std::int64_t k = 0;
            double x = 0.0;
            N4M_TEST_REQUIRE(n4m_trial_get_id(trial, &id) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_trial_get_int(trial, "k", &k) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_trial_get_float(trial, "x", &x) == N4M_OK);
            const double k_delta = static_cast<double>(k - 2);
            const double score = (x - 0.25) * (x - 0.25) + k_delta * k_delta;
            N4M_TEST_REQUIRE(n4m_optimizer_tell(original, id, score) == N4M_OK);
        }
        const std::vector<std::uint8_t> blob = save_checkpoint(original);
        N4M_TEST_REQUIRE(blob.size() % 8 == 0);
        N4M_TEST_REQUIRE(std::memcmp(blob.data(), "N4MOPT\r\n", 8) == 0);
        n4m_optimizer_t* restored = nullptr;
        N4M_TEST_REQUIRE(
            n4m_optimizer_load(context, blob.data(), blob.size(), &restored) == N4M_OK);
        N4M_TEST_REQUIRE(restored != nullptr);
        const int continuation_trials =
            (sampler == N4M_SAMPLER_GA || sampler == N4M_SAMPLER_PSO ||
             sampler == N4M_SAMPLER_CMAES)
                ? 20
                : 3;
        for (int i = 0; i < continuation_trials; ++i) {
            n4m_trial_t* left = nullptr;
            n4m_trial_t* right = nullptr;
            N4M_TEST_REQUIRE(n4m_optimizer_ask(original, &left) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_optimizer_ask(restored, &right) == N4M_OK);
            std::int64_t left_id = -1;
            std::int64_t right_id = -1;
            std::int64_t left_k = 0;
            std::int64_t right_k = 0;
            double left_x = 0.0;
            double right_x = 0.0;
            n4m_trial_get_id(left, &left_id);
            n4m_trial_get_id(right, &right_id);
            n4m_trial_get_int(left, "k", &left_k);
            n4m_trial_get_int(right, "k", &right_k);
            n4m_trial_get_float(left, "x", &left_x);
            n4m_trial_get_float(right, "x", &right_x);
            N4M_TEST_REQUIRE(left_id == right_id);
            N4M_TEST_REQUIRE(left_k == right_k);
            N4M_TEST_REQUIRE(left_x == right_x);
            const double score = left_x * left_x + static_cast<double>(left_k);
            N4M_TEST_REQUIRE(n4m_optimizer_tell(original, left_id, score) == N4M_OK);
            N4M_TEST_REQUIRE(n4m_optimizer_tell(restored, right_id, score) == N4M_OK);
        }
        n4m_optimizer_destroy(restored);
        n4m_optimizer_destroy(original);
        n4m_search_space_destroy(space);
    }
    n4m_context_destroy(context);
}

void test_checkpoint_lifecycle_queue_and_fail_closed_decode() {
    n4m_context_t* context = nullptr;
    n4m_search_space_t* space = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_create(&space) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_int(space, "k", 1, 7, 1, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_add_float(space, "x", -2.0, 2.0, 0.0, 0) == N4M_OK);
    n4m_optimizer_options_t options = default_opts();
    options.seed = 77;
    n4m_optimizer_t* optimizer = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_create(context, space, &options, &optimizer) == N4M_OK);

    n4m_trial_t* completed = nullptr;
    n4m_trial_t* failed = nullptr;
    n4m_trial_t* running = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(optimizer, &completed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_ask(optimizer, &failed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_ask(optimizer, &running) == N4M_OK);
    std::int64_t completed_id = -1;
    std::int64_t failed_id = -1;
    std::int64_t running_id = -1;
    n4m_trial_get_id(completed, &completed_id);
    n4m_trial_get_id(failed, &failed_id);
    n4m_trial_get_id(running, &running_id);
    int32_t prune = 0;
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(
                         optimizer, completed_id, 0, 2.5, &prune) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(optimizer, completed_id, 1.5) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                         optimizer, failed_id, N4M_TRIAL_FAILED, 0.0,
                         "n4m.error.v1|EVAL_ERROR|1|worker failed") == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell_intermediate(
                         optimizer, running_id, 3, 4.5, &prune) == N4M_OK);
    const char* names[] = {"k", "x"};
    const double values[] = {3.0, 0.25};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(optimizer, names, values, 2) == N4M_OK);

    const std::vector<std::uint8_t> blob = save_checkpoint(optimizer);
    n4m_optimizer_t* restored = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_load(context, blob.data(), blob.size(), &restored) == N4M_OK);
    n4m_trial_t* queued = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(restored, &queued) == N4M_OK);
    std::int64_t k = 0;
    double x = 0.0;
    n4m_trial_get_int(queued, "k", &k);
    n4m_trial_get_float(queued, "x", &x);
    N4M_TEST_REQUIRE(k == 3 && x == 0.25);
    n4m_optimizer_destroy(restored);

    N4M_TEST_REQUIRE(n4m_optimizer_save(nullptr, nullptr) == N4M_ERR_NULL_POINTER);
    n4m_array_t* output_array = reinterpret_cast<n4m_array_t*>(1);
    N4M_TEST_REQUIRE(n4m_optimizer_save(nullptr, &output_array) == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(output_array == nullptr);
    N4M_TEST_REQUIRE(n4m_optimizer_load(context, blob.data(), blob.size(), nullptr) ==
                     N4M_ERR_NULL_POINTER);

    auto reject = [&](std::vector<std::uint8_t> corrupt, n4m_status_t expected) {
        n4m_optimizer_t* output = reinterpret_cast<n4m_optimizer_t*>(1);
        const n4m_status_t status =
            n4m_optimizer_load(context, corrupt.data(), corrupt.size(), &output);
        N4M_TEST_REQUIRE(status == expected);
        N4M_TEST_REQUIRE(output == nullptr);  // transactional: no partial handle escapes
    };
    std::vector<std::uint8_t> corrupt = blob;
    N4M_TEST_REQUIRE(!corrupt.empty());
    corrupt.at(0) = static_cast<std::uint8_t>(corrupt.at(0) ^ 0xffU);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = blob;
    corrupt[corrupt.size() / 2] ^= 0x01U;
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = blob;
    checkpoint_write_u32(corrupt, 8, 2U);
    checkpoint_refresh_checksum(corrupt);
    reject(corrupt, N4M_ERR_VERSION_INCOMPATIBLE);
    corrupt.assign(blob.begin(), blob.end() - 8);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = blob;
    corrupt.push_back(0U);  // exact total-size gate rejects trailing bytes
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = blob;
    corrupt.insert(corrupt.end() - 8, 8, 0U);  // forged extra all-zero padding
    checkpoint_write_u64(corrupt, 16, static_cast<std::uint64_t>(corrupt.size()));
    checkpoint_refresh_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);  // padding must be the canonical 0..7 bytes

    // Recompute the outer checksum after tampering: the independent canonical
    // section fingerprints still reject search-space/options mismatches.
    const std::uint64_t space_size = checkpoint_u64(blob, 40);
    corrupt = blob;
    corrupt[56] = static_cast<std::uint8_t>('j');  // rename first axis k -> j
    {
        std::uint64_t section_hash = 14695981039346656037ULL;
        for (std::size_t i = 48; i < 48 + static_cast<std::size_t>(space_size); ++i) {
            section_hash ^= corrupt[i];
            section_hash *= 1099511628211ULL;
        }
        checkpoint_write_u64(corrupt, 32, section_hash);
    }
    checkpoint_refresh_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);  // space no longer matches trial state

    const std::size_t options_hash_offset = 48U + static_cast<std::size_t>(space_size);
    const std::uint64_t options_size = checkpoint_u64(blob, options_hash_offset + 8U);
    const std::size_t options_offset = 64U + static_cast<std::size_t>(space_size);
    N4M_TEST_REQUIRE(options_offset < blob.size() - 8);
    corrupt = blob;
    checkpoint_write_u32(corrupt, options_offset, N4M_SAMPLER_TPE);
    {
        std::uint64_t section_hash = 14695981039346656037ULL;
        for (std::size_t i = options_offset;
             i < options_offset + static_cast<std::size_t>(options_size); ++i) {
            section_hash ^= corrupt[i];
            section_hash *= 1099511628211ULL;
        }
        checkpoint_write_u64(corrupt, options_hash_offset, section_hash);
    }
    checkpoint_refresh_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);  // options sampler != sampler-state tag

    n4m_optimizer_destroy(optimizer);
    n4m_search_space_destroy(space);
    n4m_context_destroy(context);
}

void test_checkpoint_pso_all_failed_generation() {
    n4m_context_t* context = nullptr;
    n4m_search_space_t* space = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_create(&space) == N4M_OK);
    N4M_TEST_REQUIRE(
        n4m_search_space_add_float(space, "x", -1.0, 1.0, 0.0, 0) == N4M_OK);
    n4m_optimizer_options_t options = default_opts();
    options.sampler = N4M_SAMPLER_PSO;
    options.seed = 0xdecafbadULL;
    n4m_optimizer_t* original = nullptr;
    N4M_TEST_REQUIRE(
        n4m_optimizer_create(context, space, &options, &original) == N4M_OK);

    // Before the first swarm is folded, pbest_pos is an exact copy of pos.
    // Locate the two identical serialized 16x1 matrices, alter only pbest_pos,
    // and refresh the outer checksum: structural range checks alone must not
    // accept state that changes the next PSO update.
    n4m_optimizer_t* pristine = nullptr;
    N4M_TEST_REQUIRE(
        n4m_optimizer_create(context, space, &options, &pristine) == N4M_OK);
    n4m_trial_t* pristine_trial = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(pristine, &pristine_trial) == N4M_OK);
    const std::vector<std::uint8_t> pristine_blob = save_checkpoint(pristine);
    constexpr std::size_t kSwarmSize = 16;
    constexpr std::size_t kMatrixBytes = 4 + kSwarmSize * (4 + sizeof(double));
    auto matrix_shape_at = [&](std::size_t offset) {
        if (offset + kMatrixBytes > pristine_blob.size() - 8 ||
            checkpoint_u32(pristine_blob, offset) != kSwarmSize) {
            return false;
        }
        for (std::size_t row = 0; row < kSwarmSize; ++row) {
            if (checkpoint_u32(pristine_blob, offset + 4 + row * 12) != 1U) return false;
        }
        return true;
    };
    std::size_t pos_offset = std::numeric_limits<std::size_t>::max();
    std::size_t pbest_offset = std::numeric_limits<std::size_t>::max();
    for (std::size_t first = 32; first + kMatrixBytes <= pristine_blob.size() - 8;
         ++first) {
        if (!matrix_shape_at(first)) continue;
        for (std::size_t second = first + kMatrixBytes;
             second + kMatrixBytes <= pristine_blob.size() - 8; ++second) {
            if (matrix_shape_at(second) &&
                std::equal(pristine_blob.begin() + static_cast<std::ptrdiff_t>(first),
                           pristine_blob.begin() +
                               static_cast<std::ptrdiff_t>(first + kMatrixBytes),
                           pristine_blob.begin() + static_cast<std::ptrdiff_t>(second))) {
                pos_offset = first;
                pbest_offset = second;
                break;
            }
        }
        if (pbest_offset != std::numeric_limits<std::size_t>::max()) break;
    }
    N4M_TEST_REQUIRE(pos_offset != std::numeric_limits<std::size_t>::max());
    N4M_TEST_REQUIRE(pbest_offset != std::numeric_limits<std::size_t>::max());
    N4M_TEST_REQUIRE(pbest_offset == pos_offset + 2 * kMatrixBytes);  // pos, vel, pbest
    std::vector<std::uint8_t> corrupt_pristine = pristine_blob;
    corrupt_pristine[pbest_offset + 8] ^= 0x01U;  // first pbest coordinate, still finite/in-range
    checkpoint_refresh_checksum(corrupt_pristine);
    n4m_optimizer_t* rejected_pristine = reinterpret_cast<n4m_optimizer_t*>(1);
    N4M_TEST_REQUIRE(n4m_optimizer_load(context, corrupt_pristine.data(),
                                        corrupt_pristine.size(), &rejected_pristine) ==
                     N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(rejected_pristine == nullptr);
    n4m_optimizer_destroy(pristine);

    // PSO intentionally retains +/-infinity as the worst-score sentinel. If
    // every particle fails, crossing the generation boundary makes that
    // sentinel the provisional global best; it is a legitimate resumable
    // state, while NaN is never legitimate.
    for (int i = 0; i < 16; ++i) {
        n4m_trial_t* trial = nullptr;
        std::int64_t id = -1;
        N4M_TEST_REQUIRE(n4m_optimizer_ask(original, &trial) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_trial_get_id(trial, &id) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_optimizer_tell_result(
                             original, id, N4M_TRIAL_FAILED, 0.0,
                             "n4m.error.v1|EVAL_ERROR|0|particle failed") == N4M_OK);
    }
    n4m_trial_t* first_next_generation = nullptr;
    N4M_TEST_REQUIRE(
        n4m_optimizer_ask(original, &first_next_generation) == N4M_OK);

    const std::vector<std::uint8_t> blob = save_checkpoint(original);
    n4m_optimizer_t* restored = nullptr;
    N4M_TEST_REQUIRE(
        n4m_optimizer_load(context, blob.data(), blob.size(), &restored) == N4M_OK);
    n4m_trial_t* left = nullptr;
    n4m_trial_t* right = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(original, &left) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_ask(restored, &right) == N4M_OK);
    std::int64_t left_id = -1;
    std::int64_t right_id = -1;
    double left_x = 0.0;
    double right_x = 0.0;
    N4M_TEST_REQUIRE(n4m_trial_get_id(left, &left_id) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_trial_get_id(right, &right_id) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_trial_get_float(left, "x", &left_x) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_trial_get_float(right, "x", &right_x) == N4M_OK);
    N4M_TEST_REQUIRE(left_id == right_id);
    N4M_TEST_REQUIRE(left_x == right_x);

    // Rehash a structurally well-formed payload after changing every PSO
    // minimize sentinel from +inf to the impossible -inf sign. There are 16
    // personal-best sentinels plus the provisional global-best sentinel.
    std::vector<std::uint8_t> corrupt = blob;
    constexpr std::uint8_t kPositiveInfinityLe[8] = {0x00U, 0x00U, 0x00U, 0x00U,
                                                     0x00U, 0x00U, 0xf0U, 0x7fU};
    std::size_t rewritten = 0;
    for (std::size_t i = 32; i + 8 <= corrupt.size() - 8;) {
        if (std::equal(kPositiveInfinityLe, kPositiveInfinityLe + 8,
                       corrupt.begin() + static_cast<std::ptrdiff_t>(i))) {
            corrupt[i + 7] = 0xffU;
            ++rewritten;
            i += 8;
        } else {
            ++i;
        }
    }
    N4M_TEST_REQUIRE(rewritten == 17);
    checkpoint_refresh_checksum(corrupt);
    n4m_optimizer_t* rejected = reinterpret_cast<n4m_optimizer_t*>(1);
    N4M_TEST_REQUIRE(
        n4m_optimizer_load(context, corrupt.data(), corrupt.size(), &rejected) ==
        N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(rejected == nullptr);

    n4m_optimizer_destroy(restored);
    n4m_optimizer_destroy(original);
    n4m_search_space_destroy(space);
    n4m_context_destroy(context);
}

void test_checkpoint_wide_space_and_empty_queue() {
    n4m_context_t* context = nullptr;
    n4m_search_space_t* space = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_search_space_create(&space) == N4M_OK);
    std::vector<std::string> names;
    names.reserve(200);
    for (int i = 0; i < 200; ++i) {
        names.push_back("p" + std::to_string(i));
        N4M_TEST_REQUIRE(
            n4m_search_space_add_float(space, names.back().c_str(), 0.0, 1.0, 0.0, 0) ==
            N4M_OK);
    }
    n4m_optimizer_options_t options = default_opts();
    options.seed = 0x44556677ULL;
    n4m_optimizer_t* optimizer = nullptr;
    N4M_TEST_REQUIRE(
        n4m_optimizer_create(context, space, &options, &optimizer) == N4M_OK);
    n4m_trial_t* trial = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(optimizer, &trial) == N4M_OK);
    std::int64_t id = -1;
    N4M_TEST_REQUIRE(n4m_trial_get_id(trial, &id) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_optimizer_tell(optimizer, id, 0.0) == N4M_OK);

    // An empty warm-start is valid and means "sample every field". Its wire
    // record is only the four-byte item count, so the decoder's count guard must
    // use that true minimum rather than assuming a non-empty entry.
    const char* dummy_name = "unused";
    const double dummy_value = 0.0;
    N4M_TEST_REQUIRE(
        n4m_optimizer_enqueue(optimizer, &dummy_name, &dummy_value, 0) == N4M_OK);
    const std::vector<std::uint8_t> blob = save_checkpoint(optimizer);
    n4m_optimizer_t* restored = nullptr;
    N4M_TEST_REQUIRE(
        n4m_optimizer_load(context, blob.data(), blob.size(), &restored) == N4M_OK);
    n4m_trial_t* resumed = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(restored, &resumed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_trial_get_id(resumed, &id) == N4M_OK);
    N4M_TEST_REQUIRE(id == 1);

    n4m_optimizer_destroy(restored);
    n4m_optimizer_destroy(optimizer);
    n4m_search_space_destroy(space);
    n4m_context_destroy(context);
}

}  // namespace

void register_optimization_tests(n4m_testing::Runner& r) {
    r.run("optimization: options init", test_options_init);
    r.run("optimization: checkpoint all sampler trajectories",
          test_checkpoint_all_sampler_trajectories);
    r.run("optimization: checkpoint lifecycle + fail-closed decode",
          test_checkpoint_lifecycle_queue_and_fail_closed_decode);
    r.run("optimization: checkpoint PSO all-failed generation",
          test_checkpoint_pso_all_failed_generation);
    r.run("optimization: checkpoint wide space + empty queue",
          test_checkpoint_wide_space_and_empty_queue);
    r.run("optimization: options fail-closed validation", test_options_validation);
    r.run("optimization: search space build", test_space_build);
    r.run("optimization: reserved sampler NOT_IMPLEMENTED", test_reserved_sampler_not_implemented);
    r.run("optimization: random quadratic converges", test_quadratic_converges);
    r.run("optimization: sobol sequence parity (vs scipy)", test_sobol_sequence_parity);
    r.run("optimization: determinism given seed", test_determinism);
    r.run("optimization: ask_batch distinct trials", test_ask_batch);
    r.run("optimization: ask_batch best-effort contract", test_ask_batch_contract);
    r.run("optimization: ask_batch checkpoint + determinism",
          test_ask_batch_checkpoint_and_determinism);
    r.run("optimization: finetune_estimator PLS CV", test_finetune_estimator);
    r.run("optimization: invalid ranges rejected", test_invalid_ranges);
    r.run("optimization: ordered-space names + domains validated",
          test_space_validation_names_and_domains);
    r.run("optimization: numeric labels + int64 exactness",
          test_numeric_choice_labels_and_int64_exactness);
    r.run("optimization: binary64 choice label vectors", test_binary64_choice_label_vectors);
    r.run("optimization: constraint arity + refs + labels validated", test_constraint_validation);
    r.run("optimization: constraint reference identity", test_constraint_reference_identity);
    r.run("optimization: condition cycles rejected", test_condition_cycles_rejected);
    r.run("optimization: sampler hard-constraint matrix", test_sampler_hard_constraint_matrix);
    r.run("optimization: struct_size guard", test_struct_size_guard);
    r.run("optimization: enqueue warm-start", test_enqueue_warm_start);
    r.run("optimization: conditional activation", test_conditional_activation);
    r.run("optimization: conditional deep nesting (E2)", test_conditional_deep_nesting);
    r.run("optimization: conditional child before parent (E2)",
          test_conditional_child_before_parent);
    r.run("optimization: finetune rejects unsupported param", test_finetune_rejects_unsupported_param);
    r.run("optimization: finetune preflight contract", test_finetune_preflight_contract);
    r.run("optimization: finetune rejects global errors before study",
          test_finetune_rejects_global_input_errors_before_study);
    r.run("optimization: finetune all eligible routes", test_finetune_all_routes);
    r.run("optimization: finetune rejects all ineligible", test_finetune_rejects_all_ineligible);
    r.run("optimization: finetune schema matrix", test_finetune_schema_matrix);
    r.run("optimization: finetune sparse lambda applied", test_finetune_sparse_lambda_applied);
    r.run("optimization: finetune clears late candidate error",
          test_finetune_success_clears_late_candidate_error);
    r.run("optimization: finetune timeout contract", test_finetune_timeout_contract);
    r.run("optimization: auto direction maximizes R2", test_auto_direction_maximizes_r2);
    r.run("optimization: ternary converges (unimodal int)", test_ternary_converges);
    r.run("optimization: ternary respects step + batch reservations", test_ternary_respects_step_and_batch);
    r.run("optimization: ternary keeps large first integer axis",
          test_ternary_keeps_large_first_integer_axis);
    r.run("optimization: stepped grid sampling + enqueue consistency",
          test_stepped_grid_sampling_and_enqueue_consistency);
    r.run("optimization: ga converges (2D continuous)", test_ga_converges);
    r.run("optimization: population batch boundary + enqueue reject", test_population_batch_and_enqueue);
    r.run("optimization: pso converges (2D continuous)", test_pso_converges);
    r.run("optimization: cmaes converges (2D continuous)", test_cmaes_converges);
    r.run("optimization: tpe converges (mixed space)", test_tpe_converges_mixed);
    r.run("optimization: gp_ei converges (2D continuous)", test_gp_ei_converges);
    r.run("optimization: gp_ei maximize (EI sign)", test_gp_ei_maximize);
    r.run("optimization: gp_ei edge cases (no-cont / all-equal)", test_gp_ei_edge_cases);
    r.run("optimization: enqueue out-of-range rejected", test_enqueue_out_of_range_rejected);
    r.run("optimization: median pruner decisions", test_median_pruner);
    r.run("optimization: asha pruner decisions", test_asha_pruner);
    r.run("optimization: racing pruner decisions", test_racing_pruner);
    r.run("optimization: hyperband bracket decisions", test_hyperband_brackets);
    r.run("optimization: hyperband edges (require R, no prune above R)", test_hyperband_edges);
    r.run("optimization: pruned trial is terminal", test_pruner_lifecycle);
    r.run("optimization: terminal status + intermediate step validation",
          test_terminal_status_and_intermediate_step_validation);
    r.run("optimization: cancelled + structured error lifecycle",
          test_cancelled_structured_error_and_idempotence);
    r.run("optimization: trace UTF-8 boundaries", test_trace_utf8_boundaries);
    r.run("optimization: rich trace since_id + ownership",
          test_rich_trial_trace_since_id_and_ownership);
    r.run("optimization: invalid pruner + NaN rejected", test_invalid_pruner_and_nan);
    r.run("optimization: lhs stratifies startup batch", test_lhs_stratifies);
}
