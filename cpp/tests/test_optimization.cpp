// SPDX-License-Identifier: CECILL-2.1
//
// F0 — native ask/tell hyperparameter optimizer smoke tests. Drives the
// public C ABI end-to-end (search space, optimizer, ask/tell, best, trials,
// and the pure-native n4m_finetune_estimator over a real PLS cross-validation).

#include <cmath>
#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "harness.hpp"

namespace {

n4m_optimizer_options_t default_opts() {
    n4m_optimizer_options_t o;
    n4m_optimizer_options_init(&o);
    return o;
}

void test_options_init() {
    n4m_optimizer_options_t o;
    n4m_optimizer_options_init(&o);
    N4M_TEST_REQUIRE(o.struct_size == sizeof(n4m_optimizer_options_t));
    N4M_TEST_REQUIRE(o.sampler == N4M_SAMPLER_RANDOM);
    N4M_TEST_REQUIRE(o.pruner == N4M_PRUNER_NONE);
    N4M_TEST_REQUIRE(o.metric == N4M_METRIC_RMSE);
    N4M_TEST_REQUIRE(o.n_startup_trials == 10);
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
    o.sampler = N4M_SAMPLER_TPE;  // reserved for F4
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
    // a second condition with a different parent for the same child is rejected
    n4m_search_space_add_float(sp, "other", 0.0, 1.0, 0.0, 0);
    const char* refs2[2] = {"gamma", "other"};
    const char* labs2[2] = {"", "x"};
    N4M_TEST_REQUIRE(
        n4m_search_space_add_constraint(sp, N4M_CONSTRAINT_CONDITION_IN, refs2, labs2, 2)
        == N4M_ERR_UNSUPPORTED);
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

}  // namespace

void register_optimization_tests(n4m_testing::Runner& r) {
    r.run("optimization: options init", test_options_init);
    r.run("optimization: search space build", test_space_build);
    r.run("optimization: reserved sampler NOT_IMPLEMENTED", test_reserved_sampler_not_implemented);
    r.run("optimization: random quadratic converges", test_quadratic_converges);
    r.run("optimization: determinism given seed", test_determinism);
    r.run("optimization: ask_batch distinct trials", test_ask_batch);
    r.run("optimization: finetune_estimator PLS CV", test_finetune_estimator);
    r.run("optimization: invalid ranges rejected", test_invalid_ranges);
    r.run("optimization: struct_size guard", test_struct_size_guard);
    r.run("optimization: enqueue warm-start", test_enqueue_warm_start);
    r.run("optimization: conditional activation", test_conditional_activation);
    r.run("optimization: finetune rejects unsupported param", test_finetune_rejects_unsupported_param);
    r.run("optimization: auto direction maximizes R2", test_auto_direction_maximizes_r2);
}
