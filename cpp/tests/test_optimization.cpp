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
    o.sampler = static_cast<n4m_sampler_kind_t>(99);  // out-of-range → NOT_IMPLEMENTED
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
    const double ok[1] = {5.0};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, ok, 1) == N4M_OK);
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
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_float(sp, "x", 0.0, 1.0, 0.0, 0);
    n4m_optimizer_options_t o = default_opts();
    o.sampler = N4M_SAMPLER_GA;
    o.seed = 1;
    n4m_optimizer_t* opt = nullptr;
    n4m_optimizer_create(ctx, sp, &o, &opt);
    // ask_batch beyond the population (16) returns a partial batch at the boundary
    n4m_trial_t* buf[20] = {nullptr};
    int32_t count = 0;
    const n4m_status_t st = n4m_optimizer_ask_batch(opt, 20, buf, &count);
    N4M_TEST_REQUIRE(count == 16);   // stops at the generation boundary
    N4M_TEST_REQUIRE(st != N4M_OK);  // the 17th ask cannot advance without scores
    for (int i = 0; i < 16; ++i) {
        int64_t id = 0;
        n4m_trial_get_id(buf[i], &id);
        n4m_optimizer_tell(opt, id, 0.5);
    }
    n4m_trial_t* t = nullptr;
    N4M_TEST_REQUIRE(n4m_optimizer_ask(opt, &t) == N4M_OK);  // next generation now available
    // enqueue/warm-start is unsupported for population samplers
    const char* names[1] = {"x"};
    const double v[1] = {0.3};
    N4M_TEST_REQUIRE(n4m_optimizer_enqueue(opt, names, v, 1) == N4M_ERR_UNSUPPORTED);
    n4m_optimizer_destroy(opt);
    n4m_search_space_destroy(sp);
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

void test_invalid_pruner_and_nan() {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_search_space_t* sp = nullptr;
    n4m_search_space_create(&sp);
    n4m_search_space_add_int(sp, "k", 1, 10, 1, 0);
    n4m_optimizer_options_t o = default_opts();
    o.pruner = static_cast<n4m_pruner_kind_t>(99);  // out-of-range → NOT_IMPLEMENTED
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

}  // namespace

void register_optimization_tests(n4m_testing::Runner& r) {
    r.run("optimization: options init", test_options_init);
    r.run("optimization: search space build", test_space_build);
    r.run("optimization: reserved sampler NOT_IMPLEMENTED", test_reserved_sampler_not_implemented);
    r.run("optimization: random quadratic converges", test_quadratic_converges);
    r.run("optimization: sobol sequence parity (vs scipy)", test_sobol_sequence_parity);
    r.run("optimization: determinism given seed", test_determinism);
    r.run("optimization: ask_batch distinct trials", test_ask_batch);
    r.run("optimization: finetune_estimator PLS CV", test_finetune_estimator);
    r.run("optimization: invalid ranges rejected", test_invalid_ranges);
    r.run("optimization: struct_size guard", test_struct_size_guard);
    r.run("optimization: enqueue warm-start", test_enqueue_warm_start);
    r.run("optimization: conditional activation", test_conditional_activation);
    r.run("optimization: conditional deep nesting (E2)", test_conditional_deep_nesting);
    r.run("optimization: finetune rejects unsupported param", test_finetune_rejects_unsupported_param);
    r.run("optimization: auto direction maximizes R2", test_auto_direction_maximizes_r2);
    r.run("optimization: ternary converges (unimodal int)", test_ternary_converges);
    r.run("optimization: ternary respects step + batch reservations", test_ternary_respects_step_and_batch);
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
    r.run("optimization: invalid pruner + NaN rejected", test_invalid_pruner_and_nan);
    r.run("optimization: lhs stratifies startup batch", test_lhs_stratifies);
}
