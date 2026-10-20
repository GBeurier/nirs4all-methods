// SPDX-License-Identifier: CECILL-2.1
//
// Consumers of fitted PLS / PCR models with null components (components
// flagged in N4M_MODEL_UNRESOLVED_COMPONENTS with zero W, P, R, T columns):
// the PLS diagnostics and monitoring, the PLS-LDA / PLS-logistic heads, the
// T2 variable selection, and the scale invariance of VIP and the selectivity
// ratio. Every expected value is written down
// analytically with the centred, orthogonal a = (-3, -1, 1, 3),
// b = (1, -3, 3, -1) (a'a = b'b = 20, a'b = 0), and the statistics of a model
// with null components are compared with those of the model truncated to its
// resolved components.

#include "n4m/n4m.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

void register_pls_null_component_tests(n4m_testing::Runner& r);

namespace {

constexpr double kA[4] = {-3.0, -1.0, 1.0, 3.0};
constexpr double kB[4] = {1.0, -3.0, 3.0, -1.0};
constexpr double kTol = 1e-12;

struct Fit {
    n4m_context_t* ctx{nullptr};
    n4m_model_t* model{nullptr};
    Fit() = default;
    Fit(const Fit&) = delete;
    Fit& operator=(const Fit&) = delete;
    ~Fit() {
        n4m_model_destroy(model);
        n4m_context_destroy(ctx);
    }
};

n4m_matrix_view_t view(std::vector<double>& values, std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t v{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&v, values.data(), rows, cols,
                                                   N4M_DTYPE_F64) == N4M_OK);
    return v;
}

// Fits (centred, store_scores = 1, X and Y scaled when `scale`) and keeps the
// model; refuses the test on a failed fit.
void fit(Fit& out, n4m_algorithm_t algorithm, n4m_solver_t solver, std::vector<double> x,
         std::int64_t p, std::vector<double> y, std::int64_t q, std::int32_t components,
         bool scale = false) {
    N4M_TEST_REQUIRE(n4m_context_create(&out.ctx) == N4M_OK);
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, components) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_algorithm(cfg, algorithm) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, solver) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, scale ? 1 : 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_y(cfg, scale ? 1 : 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_store_scores(cfg, 1) == N4M_OK);
    const n4m_matrix_view_t xv = view(x, 4, p);
    const n4m_matrix_view_t yv = view(y, 4, q);
    const n4m_status_t st = n4m_model_fit(out.ctx, cfg, &xv, &yv, &out.model);
    n4m_config_destroy(cfg);
    if (st != N4M_OK) throw std::runtime_error("fit refused with status " + std::to_string(st));
}

std::vector<double> model_array(const Fit& f, n4m_model_array_t which) {
    n4m_array_t* arr = nullptr;
    N4M_TEST_REQUIRE(n4m_model_get_array(f.ctx, f.model, which, &arr) == N4M_OK);
    n4m_matrix_view_t v{};
    N4M_TEST_REQUIRE(n4m_array_view(arr, &v) == N4M_OK);
    const auto* data = static_cast<const double*>(v.data);
    std::vector<double> out(data, data + v.rows * v.cols);
    n4m_array_free(arr);
    return out;
}

std::vector<double> result_matrix(const n4m_method_result_t* result, const char* name) {
    const double* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(result, name, &data, &rows, &cols) ==
                     N4M_OK);
    return std::vector<double>(data, data + rows * cols);
}

double result_scalar(const n4m_method_result_t* result, const char* name) {
    double value = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, name, &value) == N4M_OK);
    return value;
}

void expect_close(const std::string& where, const std::vector<double>& got,
                  const std::vector<double>& want, double abs_floor = 0.0) {
    if (got.size() != want.size()) throw std::runtime_error(where + ": wrong size");
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (!(std::fabs(got[i] - want[i]) <= kTol * std::fabs(want[i]) + abs_floor))
            throw std::runtime_error(where + "[" + std::to_string(i) + "] = " +
                                     std::to_string(got[i]) + ", expected " +
                                     std::to_string(want[i]));
    }
}

struct Diagnostics {
    std::vector<double> t2, q, dmodx;
    double directions{0.0};
    double components{0.0};
};

Diagnostics diagnostics(const Fit& f, std::vector<double> x, std::int64_t p,
                        bool with_reference) {
    const n4m_matrix_view_t xv = view(x, 4, p);
    n4m_method_result_t* result = nullptr;
    const n4m_status_t st = n4m_metrics_pls_diagnostics_compute(
        f.ctx, f.model, &xv, with_reference ? &xv : nullptr, &result);
    if (st != N4M_OK)
        throw std::runtime_error(std::string("diagnostics refused: ") +
                                 n4m_context_last_error(f.ctx));
    Diagnostics out;
    out.t2 = result_matrix(result, "t2");
    out.q = result_matrix(result, "q");
    out.dmodx = result_matrix(result, "dmodx");
    out.directions = result_scalar(result, "n_latent_directions");
    out.components = result_scalar(result, "n_components");
    n4m_method_result_destroy(result);
    return out;
}

std::vector<double> rows_of(double (*f)(std::size_t)) {
    std::vector<double> out(4);
    for (std::size_t i = 0; i < 4; ++i) out[i] = f(i);
    return out;
}

constexpr struct {
    const char* tag;
    n4m_solver_t solver;
} kSolvers[] = {
    {"nipals", N4M_SOLVER_NIPALS},
    {"simpls", N4M_SOLVER_SIMPLS},
    {"orthogonal scores", N4M_SOLVER_ORTHOGONAL_SCORES},
    {"kernel", N4M_SOLVER_KERNEL_ALGORITHM},
    {"wide kernel", N4M_SOLVER_WIDE_KERNEL},
    {"svd", N4M_SOLVER_SVD},
    {"power", N4M_SOLVER_POWER},
    {"randomized svd", N4M_SOLVER_RANDOMIZED_SVD},
};

// X = [a, a, b], Y = a, two components: one direction w = (1, 1, 0) / sqrt 2,
// t = sqrt 2 a (variance 40 / 3), and a null second component. T2 = 3 a^2 / 20,
// Q = b^2 (the b column is not reconstructed), DModX = |b| / sqrt(3 - 1)
// without reference and |b| / sqrt 10 with X as reference
// (sigma^2 = sum Q / ((3 - 1)(4 - 1 - 1)) = 5). Bit-identical to the
// one-component model, for every solver. The second-component null scores
// used to make T2 refuse ("reference variance is zero at component 1").
void test_diagnostics_null_component() {
    std::vector<double> x(12), y(4);
    for (std::size_t i = 0; i < 4; ++i) {
        x[i * 3] = x[i * 3 + 1] = kA[i];
        x[i * 3 + 2] = kB[i];
        y[i] = kA[i];
    }
    const std::vector<double> t2 = rows_of([](std::size_t i) { return 3.0 * kA[i] * kA[i] / 20.0; });
    const std::vector<double> q = rows_of([](std::size_t i) { return kB[i] * kB[i]; });
    const std::vector<double> dmodx = rows_of([](std::size_t i) { return std::fabs(kB[i]) / std::sqrt(2.0); });
    const std::vector<double> dmodx_ref = rows_of([](std::size_t i) { return std::fabs(kB[i]) / std::sqrt(10.0); });
    for (const auto& s : kSolvers) {
        Fit two, one;
        fit(two, N4M_ALGO_PLS_REGRESSION, s.solver, x, 3, y, 1, 2);
        fit(one, N4M_ALGO_PLS_REGRESSION, s.solver, x, 3, y, 1, 1);
        if (model_array(two, N4M_MODEL_UNRESOLVED_COMPONENTS) != std::vector<double>{0.0, 1.0})
            throw std::runtime_error(std::string(s.tag) + ": expected a null second component");
        for (const bool reference : {false, true}) {
            const std::string where = std::string(s.tag) + (reference ? " (reference X)" : "");
            const Diagnostics d = diagnostics(two, x, 3, reference);
            const Diagnostics d1 = diagnostics(one, x, 3, reference);
            expect_close(where + " t2", d.t2, t2);
            expect_close(where + " q", d.q, q);
            expect_close(where + " dmodx", d.dmodx, reference ? dmodx_ref : dmodx);
            if (d.directions != 1.0 || d.components != 2.0)
                throw std::runtime_error(where + ": n_latent_directions / n_components");
            if (d.t2 != d1.t2 || d.q != d1.q || d.dmodx != d1.dmodx)
                throw std::runtime_error(where + ": differs from the one-component model");
        }
    }
}

// PCR keeps the X direction of a component flagged unresolved only for its Y
// side. X = [2 a, b, 0], Y = [a + 2^-48 b, 3 a - 2^-48 b], two components: the
// second (Y't_2 below the resolution) is flagged, yet both directions
// t_1 = 2 a (variance 80 / 3) and t_2 = b (variance 20 / 3) are kept:
// T2 = 3 (a^2 + b^2) / 20 = 3 / 2 on every row, Q = 0, DModX = 0 with
// n_features - 2 = 1 degree of freedom.
void test_diagnostics_pcr_kept_direction() {
    std::vector<double> x(12, 0.0), y(8);
    for (std::size_t i = 0; i < 4; ++i) {
        x[i * 3] = 2.0 * kA[i];
        x[i * 3 + 1] = kB[i];
        y[i * 2] = kA[i] + std::ldexp(kB[i], -48);
        y[i * 2 + 1] = 3.0 * kA[i] - std::ldexp(kB[i], -48);
    }
    Fit f;
    fit(f, N4M_ALGO_PCR, N4M_SOLVER_SVD, x, 3, y, 2, 2);
    if (model_array(f, N4M_MODEL_UNRESOLVED_COMPONENTS) != std::vector<double>{0.0, 1.0})
        throw std::runtime_error("pcr: expected the second component flagged");
    const Diagnostics d = diagnostics(f, x, 3, false);
    expect_close("pcr t2", d.t2, {1.5, 1.5, 1.5, 1.5});
    expect_close("pcr q", d.q, {0.0, 0.0, 0.0, 0.0}, 1e-24);
    expect_close("pcr dmodx", d.dmodx, {0.0, 0.0, 0.0, 0.0}, 1e-12);
    if (d.directions != 2.0) throw std::runtime_error("pcr: n_latent_directions != 2");
}

// No resolved component at all: X = a, Y = b + 2^-50 a, one component (a
// null component). There is no latent direction: T2 = 0 on every row (the
// model space has dimension 0), Q = a^2 and DModX = |a| (one degree of
// freedom). Monitoring on the same model runs (thresholds 0 for T2).
void test_diagnostics_no_direction() {
    std::vector<double> x(4), y(4);
    for (std::size_t i = 0; i < 4; ++i) {
        x[i] = kA[i];
        y[i] = kB[i] + std::ldexp(kA[i], -50);
    }
    for (const auto& s : kSolvers) {
        Fit f;
        fit(f, N4M_ALGO_PLS_REGRESSION, s.solver, x, 1, y, 1, 1);
        const Diagnostics d = diagnostics(f, x, 1, false);
        const std::string where = s.tag;
        expect_close(where + " t2", d.t2, {0.0, 0.0, 0.0, 0.0});
        expect_close(where + " q", d.q, {9.0, 1.0, 1.0, 9.0});
        expect_close(where + " dmodx", d.dmodx, {3.0, 1.0, 1.0, 3.0});
        if (d.directions != 0.0) throw std::runtime_error(where + ": n_latent_directions != 0");
        const n4m_matrix_view_t xv = view(x, 4, 1);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_metrics_pls_monitoring_run(f.ctx, f.model, &xv, &xv, 0.05,
                                                        &result) == N4M_OK);
        expect_close(where + " monitoring t2", result_matrix(result, "t2"), {0.0, 0.0, 0.0, 0.0});
        n4m_method_result_destroy(result);
    }
}

// Monitoring with a null component: X = [a, a, b], Y = a, two components;
// reference and monitored rows X. Its T2 and Q are the diagnostics above and
// bit-identical to the one-component model's.
void test_monitoring_null_component() {
    std::vector<double> x(12), y(4);
    for (std::size_t i = 0; i < 4; ++i) {
        x[i * 3] = x[i * 3 + 1] = kA[i];
        x[i * 3 + 2] = kB[i];
        y[i] = kA[i];
    }
    Fit two, one;
    fit(two, N4M_ALGO_PLS_REGRESSION, N4M_SOLVER_NIPALS, x, 3, y, 1, 2);
    fit(one, N4M_ALGO_PLS_REGRESSION, N4M_SOLVER_NIPALS, x, 3, y, 1, 1);
    const n4m_matrix_view_t xv = view(x, 4, 3);
    std::vector<double> got[2][2];
    const Fit* fits[2] = {&two, &one};
    for (std::size_t k = 0; k < 2; ++k) {
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_metrics_pls_monitoring_run(fits[k]->ctx, fits[k]->model, &xv, &xv,
                                                        0.05, &result) == N4M_OK);
        got[k][0] = result_matrix(result, "t2");
        got[k][1] = result_matrix(result, "q");
        n4m_method_result_destroy(result);
    }
    expect_close("monitoring t2", got[0][0],
                 rows_of([](std::size_t i) { return 3.0 * kA[i] * kA[i] / 20.0; }));
    expect_close("monitoring q", got[0][1], rows_of([](std::size_t i) { return kB[i] * kB[i]; }));
    if (got[0][0] != got[1][0] || got[0][1] != got[1][1])
        throw std::runtime_error("monitoring differs from the one-component model");
}

// Autoscaled model: X = [a, 2 b], Y = a, one component, scale on. The model
// X space is X / s with s = sqrt(20 / 3) (sample standard deviation of a and
// of 2 b / 2): w = (1, 0), t = a / s (variance 1), T2 = 3 a^2 / 20 and
// Q = 3 b^2 / 20. The diagnostics used to project the centred but unscaled
// rows (T2 = a^2, Q mixed units).
void test_diagnostics_autoscaled() {
    std::vector<double> x(8), y(4);
    for (std::size_t i = 0; i < 4; ++i) {
        x[i * 2] = kA[i];
        x[i * 2 + 1] = 2.0 * kB[i];
        y[i] = kA[i];
    }
    Fit f;
    fit(f, N4M_ALGO_PLS_REGRESSION, N4M_SOLVER_NIPALS, x, 2, y, 1, 1, true);
    const Diagnostics d = diagnostics(f, x, 2, false);
    expect_close("autoscaled t2", d.t2,
                 rows_of([](std::size_t i) { return 3.0 * kA[i] * kA[i] / 20.0; }));
    expect_close("autoscaled q", d.q,
                 rows_of([](std::size_t i) { return 3.0 * kB[i] * kB[i] / 20.0; }));
}

struct HeadFit {
    std::vector<double> decision, probabilities, coefficients;
    std::vector<std::int32_t> predictions;
};

// n4m_estimators_pls_{lda,logistic}_fit on centred, unscaled X.
HeadFit fit_head(bool logistic, n4m_solver_t solver, std::vector<double> x, std::int64_t p,
                 const std::vector<std::int32_t>& labels, std::int32_t components) {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, components) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, solver) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_y(cfg, 0) == N4M_OK);
    const n4m_matrix_view_t xv = view(x, 4, p);
    n4m_method_result_t* result = nullptr;
    const n4m_status_t st =
        logistic ? n4m_estimators_pls_logistic_fit(ctx, cfg, &xv, labels.data(), 4, 2, &result)
                 : n4m_estimators_pls_lda_fit(ctx, cfg, &xv, labels.data(), 4, 2, &result);
    if (st != N4M_OK) {
        const std::string message = n4m_context_last_error(ctx);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        throw std::runtime_error("head fit refused: " + message);
    }
    HeadFit out;
    out.decision = result_matrix(result, "decision_scores");
    if (logistic) {
        out.probabilities = result_matrix(result, "probabilities");
        out.coefficients = result_matrix(result, "coefficients");
    }
    const std::int32_t* pred = nullptr;
    std::int32_t size = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(result, "predictions", &pred, &size) ==
                     N4M_OK);
    out.predictions.assign(pred, pred + size);
    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return out;
}

// PLS-LDA on X = [a, a], classes (0, 0, 1, 1), two components: the second
// PLS-DA component is null and its zero score column made the pooled
// covariance singular ("failed to invert PLS-LDA pooled covariance"). The
// head now uses the one resolved direction t ~ a: class means -+2 sqrt 2 (in
// unit-weight units), pooled variance 4, so the decision is
// log(1/2) - 1 - a_i for class 0 and log(1/2) - 1 + a_i for class 1 (any
// weight normalisation), predictions (0, 0, 1, 1), bit-identical to the
// one-component head. PLS-logistic (ridge-regularised Newton) already
// handled it; its two-component fit equals the one-component one with a
// zero coefficient for the null component.
void test_score_heads_null_component() {
    std::vector<double> x(8);
    for (std::size_t i = 0; i < 4; ++i) x[i * 2] = x[i * 2 + 1] = kA[i];
    const std::vector<std::int32_t> labels = {0, 0, 1, 1};
    std::vector<double> decision(8);
    for (std::size_t i = 0; i < 4; ++i) {
        decision[i * 2] = std::log(0.5) - 1.0 - kA[i];
        decision[i * 2 + 1] = std::log(0.5) - 1.0 + kA[i];
    }
    for (const n4m_solver_t solver : {N4M_SOLVER_NIPALS, N4M_SOLVER_SIMPLS}) {
        const std::string tag = solver == N4M_SOLVER_NIPALS ? "nipals" : "simpls";
        const HeadFit two = fit_head(false, solver, x, 2, labels, 2);
        const HeadFit one = fit_head(false, solver, x, 2, labels, 1);
        expect_close("pls-lda " + tag + " decision", two.decision, decision);
        if (two.predictions != std::vector<std::int32_t>{0, 0, 1, 1})
            throw std::runtime_error("pls-lda " + tag + ": predictions");
        if (two.decision != one.decision)
            throw std::runtime_error("pls-lda " + tag + ": differs from the one-component head");

        const HeadFit ltwo = fit_head(true, solver, x, 2, labels, 2);
        const HeadFit lone = fit_head(true, solver, x, 2, labels, 1);
        if (ltwo.decision != lone.decision || ltwo.probabilities != lone.probabilities ||
            ltwo.predictions != lone.predictions || ltwo.coefficients.size() != 2 ||
            ltwo.coefficients[0] != lone.coefficients[0] || ltwo.coefficients[1] != 0.0)
            throw std::runtime_error("pls-logistic " + tag +
                                     ": differs from the one-component head");
    }
}

// No resolved direction: X = a, classes (0, 1, 1, 0) (the centred class
// indicators are orthogonal to a, so the only component is null). PLS-LDA
// decides by the class priors: decision log(1/2) everywhere, predictions the
// lowest class id on the tie (0).
void test_lda_no_direction() {
    std::vector<double> x(kA, kA + 4);
    const HeadFit fit = fit_head(false, N4M_SOLVER_NIPALS, x, 1, {0, 1, 1, 0}, 1);
    expect_close("pls-lda no direction decision", fit.decision,
                 std::vector<double>(8, std::log(0.5)));
    if (fit.predictions != std::vector<std::int32_t>{0, 0, 0, 0})
        throw std::runtime_error("pls-lda no direction: predictions");
}

struct T2Selection {
    std::vector<double> t2, ucl, rmse;
    std::vector<std::int64_t> mask;
    double directions{0.0};
};

T2Selection t2_select(std::vector<double> x, std::vector<double> y, std::int32_t components,
                      const std::vector<double>& alphas) {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    n4m_validation_plan_t* plan = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, components) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_y(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_validation_plan_create(&plan) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_validation_plan_set_n_samples(plan, 4) == N4M_OK);
    for (std::int64_t fold = 0; fold < 4; ++fold) {
        std::vector<std::int64_t> train;
        for (std::int64_t i = 0; i < 4; ++i)
            if (i != fold) train.push_back(i);
        N4M_TEST_REQUIRE(n4m_validation_plan_add_fold(plan, train.data(), 3, &fold, 1) == N4M_OK);
    }
    const n4m_matrix_view_t xv = view(x, 4, 4);
    const n4m_matrix_view_t yv = view(y, 4, 1);
    n4m_method_result_t* result = nullptr;
    const n4m_status_t st = n4m_feature_selection_t2_select(
        ctx, cfg, &xv, &yv, plan, alphas.data(), static_cast<std::int64_t>(alphas.size()),
        2, &result);
    if (st != N4M_OK) {
        const std::string message = n4m_context_last_error(ctx);
        n4m_validation_plan_destroy(plan);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        throw std::runtime_error("T2 selection refused: " + message);
    }
    T2Selection out;
    out.t2 = result_matrix(result, "t2_scores");
    out.ucl = result_matrix(result, "ucl_by_alpha");
    out.rmse = result_matrix(result, "rmse_by_alpha");
    out.directions = result_scalar(result, "n_latent_directions");
    const std::int64_t* mask = nullptr;
    std::int64_t size = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(result, "selected_mask", &mask, &size) ==
                     N4M_OK);
    out.mask.assign(mask, mask + size);
    n4m_method_result_destroy(result);
    n4m_validation_plan_destroy(plan);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return out;
}

// T2 variable selection on X = [a, 2 a, 3 a, 4 a], Y = a, two components:
// the second weight column is zero (null component) and the weight
// covariance was refused as singular (N4M_ERR_INVALID_ARGUMENT). With the
// one direction w = (1, 2, 3, 4) / sqrt 30 (variance 1 / 18 over the four
// weights): T2 = (1.35, 0.15, 0.15, 1.35), and the limit
// (p - 1)^2 / p Beta_{1-alpha}(1/2, 1) = 2.25 (1 - alpha)^2 (the CDF of
// Beta(1/2, 1) is sqrt x), rounded to 0.01: 2.03 at alpha = 0.05, 1.82 at
// 0.1. Everything (T2, limits, masks, CV RMSE; min_selected = 2) equals
// the one-component selection.
void test_t2_selection_null_component() {
    std::vector<double> x(16), y(4);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) x[i * 4 + j] = static_cast<double>(j + 1) * kA[i];
        y[i] = kA[i];
    }
    const std::vector<double> alphas = {0.05, 0.1};
    const T2Selection two = t2_select(x, y, 2, alphas);
    const T2Selection one = t2_select(x, y, 1, alphas);
    expect_close("t2 selection scores", two.t2, {1.35, 0.15, 0.15, 1.35});
    expect_close("t2 selection limits", two.ucl, {2.03, 1.82});
    if (two.directions != 1.0) throw std::runtime_error("t2 selection: n_latent_directions != 1");
    if (two.t2 != one.t2 || two.ucl != one.ucl || two.rmse != one.rmse || two.mask != one.mask)
        throw std::runtime_error("t2 selection differs from the one-component selection");
}

// Variable-importance scores (method 0 = VIP, 2 = selectivity ratio) of a
// fitted model on its training X; the status is returned, the scores in `out`.
n4m_status_t variable_scores(const Fit& f, std::vector<double> x, std::int64_t p,
                             std::int32_t method, std::vector<double>& out) {
    const n4m_matrix_view_t xv = view(x, 4, p);
    n4m_method_result_t* result = nullptr;
    const n4m_status_t st =
        n4m_feature_selection_variable_select_rank(f.ctx, f.model, &xv, method, 1, &result);
    if (st == N4M_OK) {
        out = result_matrix(result, "scores");
        n4m_method_result_destroy(result);
    }
    return st;
}

// VIP is invariant to the units of X and Y. X = 2^e [a, b], Y = 2^f a, one
// component: w = (1, 0) for every solver (SIMPLS stores w / ||X w||, of
// magnitude 2^-e), VIP = (sqrt 2, 0). The weight-norm and explained-variance
// tests were absolute eps: SIMPLS returned (0, 0) from e = 30 on (also at
// 2^254), NIPALS refused Y = 2^-30 a ("Y variance explained by components
// vanished"). X = 2^e [a, b], Y = 2^f [2 a, b], two components:
// SSY = (||t||^2 ||q||^2) = (80, 20) 2^2f, VIP = (sqrt 1.6, sqrt 0.4).
// X = [a, a], Y = a, two components (null second component, flagged): the
// VIP of the one-component model, (1, 1). Randomized SVD stops on its
// default tolerance (weights within ~1e-9 here), hence its 1e-8 margin.
void test_vip_scale_invariance() {
    const double r2 = std::sqrt(2.0);
    for (const auto& s : kSolvers) {
        const double margin = s.solver == N4M_SOLVER_RANDOMIZED_SVD ? 1e-8 : kTol;
        for (const int e : {0, 30, 254, -200}) {
            for (const int f : {0, -30, -200, 200}) {
                std::vector<double> x(8), y(4), y2(8);
                for (std::size_t i = 0; i < 4; ++i) {
                    x[i * 2] = std::ldexp(kA[i], e);
                    x[i * 2 + 1] = std::ldexp(kB[i], e);
                    y[i] = std::ldexp(kA[i], f);
                    y2[i * 2] = std::ldexp(2.0 * kA[i], f);
                    y2[i * 2 + 1] = std::ldexp(kB[i], f);
                }
                const std::string where = std::string(s.tag) + " VIP X = 2^" + std::to_string(e) +
                                          " [a, b], Y = 2^" + std::to_string(f);
                Fit one, two;
                fit(one, N4M_ALGO_PLS_REGRESSION, s.solver, x, 2, y, 1, 1);
                std::vector<double> vip;
                if (variable_scores(one, x, 2, 0, vip) != N4M_OK)
                    throw std::runtime_error(where + " a: refused: " + n4m_context_last_error(one.ctx));
                expect_close(where + " a", vip, {r2, 0.0}, margin);
                fit(two, N4M_ALGO_PLS_REGRESSION, s.solver, x, 2, y2, 2, 2);
                if (variable_scores(two, x, 2, 0, vip) != N4M_OK)
                    throw std::runtime_error(where + " [2a, b]: refused: " +
                                             n4m_context_last_error(two.ctx));
                expect_close(where + " [2a, b]", vip, {std::sqrt(1.6), std::sqrt(0.4)}, margin);
            }
        }
        std::vector<double> x(8), y(4);
        for (std::size_t i = 0; i < 4; ++i) {
            x[i * 2] = x[i * 2 + 1] = kA[i];
            y[i] = kA[i];
        }
        Fit two;
        fit(two, N4M_ALGO_PLS_REGRESSION, s.solver, x, 2, y, 1, 2);
        if (model_array(two, N4M_MODEL_UNRESOLVED_COMPONENTS) != std::vector<double>{0.0, 1.0})
            throw std::runtime_error(std::string(s.tag) + " VIP [a, a]: expected a null component");
        std::vector<double> vip;
        N4M_TEST_REQUIRE(variable_scores(two, x, 2, 0, vip) == N4M_OK);
        expect_close(std::string(s.tag) + " VIP X = [a, a]", vip, {1.0, 1.0});
    }
}

// The selectivity ratio is invariant to the units of X. X = s [a + b, a - b],
// Y = a, one component: t = s sqrt 2 a, T P' = s [a, a], so the explained
// and residual sums are both 20 s^2 and SR = (1, 1) for every s != 0. The
// residual was floored at eps: (0.078125, 0.078125) at s = 2^-30, ~6.7e-138
// at s = 2^-256. A residual that is exactly zero is published explicitly:
// NIPALS on X = [a, 0], Y = a reconstructs the first column exactly
// (w = p = (1, 0), t = a), SR = +inf; the second column is identically
// zero, SR = 0 (nothing to explain).
void test_selectivity_ratio_scale_invariance() {
    for (const auto& s : kSolvers) {
        for (const int e : {0, -30, -256, 30, 250}) {
            std::vector<double> x(8), y(4);
            for (std::size_t i = 0; i < 4; ++i) {
                x[i * 2] = std::ldexp(kA[i] + kB[i], e);
                x[i * 2 + 1] = std::ldexp(kA[i] - kB[i], e);
                y[i] = kA[i];
            }
            const std::string where =
                std::string(s.tag) + " SR X = 2^" + std::to_string(e) + " [a + b, a - b]";
            Fit f;
            fit(f, N4M_ALGO_PLS_REGRESSION, s.solver, x, 2, y, 1, 1);
            std::vector<double> sr;
            if (variable_scores(f, x, 2, 2, sr) != N4M_OK)
                throw std::runtime_error(where + ": refused: " + n4m_context_last_error(f.ctx));
            expect_close(where, sr, {1.0, 1.0}, 1e-11);
        }
    }
    std::vector<double> x(8, 0.0), y(4);
    for (std::size_t i = 0; i < 4; ++i) {
        x[i * 2] = kA[i];
        y[i] = kA[i];
    }
    Fit f;
    fit(f, N4M_ALGO_PLS_REGRESSION, N4M_SOLVER_NIPALS, x, 2, y, 1, 1);
    std::vector<double> sr;
    N4M_TEST_REQUIRE(variable_scores(f, x, 2, 2, sr) == N4M_OK);
    if (!(sr.size() == 2 && std::isinf(sr[0]) && sr[0] > 0.0 && sr[1] == 0.0))
        throw std::runtime_error("SR X = [a, 0]: expected (+inf, 0)");
}

// DModX against a reference whose residual vanishes. Fit on X_ref = 2^e [a, 0],
// Y = a, one component (w = p = (1, 0)), evaluate X = 2^e [a, b] with X_ref as
// reference: every reference row is reconstructed (sum Q_ref = 0), so the
// normalized distance is undefined. It used to replace sigma_ref = 0 by 1 and
// publish DModX = |b| 2^e = (1, 3, 3, 1) 2^e, which depends on the units; it
// is now NaN on every row with dmodx_reference_degenerate = 1, while
// Q = 2^2e b^2 and T2 = 3 a^2 / 20 stay available. With
// X_ref = 2^e [a, 2^-20 b] the reference residual 2^(e-20) b is resolved:
// sigma_ref^2 = sum Q_ref / ((2 - 1)(4 - 1 - 1)) = 10 2^(2e-40) and
// DModX = |b| 2^20 / sqrt 10 for every e, flag 0.
void test_dmodx_degenerate_reference() {
    for (const auto& s : kSolvers) {
        for (const int e : {0, -30, 200}) {
            for (const bool degenerate : {true, false}) {
                std::vector<double> x_ref(8), x(8), y(4);
                for (std::size_t i = 0; i < 4; ++i) {
                    x_ref[i * 2] = x[i * 2] = std::ldexp(kA[i], e);
                    x_ref[i * 2 + 1] = degenerate ? 0.0 : std::ldexp(kB[i], e - 20);
                    x[i * 2 + 1] = std::ldexp(kB[i], e);
                    y[i] = kA[i];
                }
                const std::string where = std::string(s.tag) + " DModX X_ref = 2^" +
                                          std::to_string(e) +
                                          (degenerate ? " [a, 0]" : " [a, 2^-20 b]");
                Fit f;
                fit(f, N4M_ALGO_PLS_REGRESSION, s.solver, x_ref, 2, y, 1, 1);
                const n4m_matrix_view_t xv = view(x, 4, 2);
                const n4m_matrix_view_t rv = view(x_ref, 4, 2);
                n4m_method_result_t* result = nullptr;
                if (n4m_metrics_pls_diagnostics_compute(f.ctx, f.model, &xv, &rv, &result) !=
                    N4M_OK)
                    throw std::runtime_error(where + ": refused: " + n4m_context_last_error(f.ctx));
                const std::vector<double> dmodx = result_matrix(result, "dmodx");
                const double flag = result_scalar(result, "dmodx_reference_degenerate");
                std::vector<double> q(4);
                for (std::size_t i = 0; i < 4; ++i) q[i] = std::ldexp(kB[i] * kB[i], 2 * e);
                expect_close(where + " q", result_matrix(result, "q"), q);
                expect_close(where + " t2", result_matrix(result, "t2"),
                             rows_of([](std::size_t i) { return 3.0 * kA[i] * kA[i] / 20.0; }));
                n4m_method_result_destroy(result);
                if (degenerate) {
                    if (flag != 1.0) throw std::runtime_error(where + ": flag not set");
                    for (const double d : dmodx)
                        if (!std::isnan(d)) throw std::runtime_error(where + ": DModX is not NaN");
                } else {
                    if (flag != 0.0) throw std::runtime_error(where + ": flag set");
                    expect_close(where + " dmodx", dmodx, rows_of([](std::size_t i) {
                                     return std::fabs(kB[i]) * std::ldexp(1.0, 20) / std::sqrt(10.0);
                                 }));
                }
            }
        }
    }
}

}  // namespace

void register_pls_null_component_tests(n4m_testing::Runner& r) {
    r.run("pls_null_components/diagnostics", test_diagnostics_null_component);
    r.run("pls_null_components/diagnostics_pcr_kept_direction", test_diagnostics_pcr_kept_direction);
    r.run("pls_null_components/diagnostics_no_direction", test_diagnostics_no_direction);
    r.run("pls_null_components/monitoring", test_monitoring_null_component);
    r.run("pls_null_components/diagnostics_autoscaled", test_diagnostics_autoscaled);
    r.run("pls_null_components/score_heads", test_score_heads_null_component);
    r.run("pls_null_components/lda_no_direction", test_lda_no_direction);
    r.run("pls_null_components/t2_selection", test_t2_selection_null_component);
    r.run("pls_null_components/vip_scale_invariance", test_vip_scale_invariance);
    r.run("pls_null_components/selectivity_ratio_scale_invariance",
          test_selectivity_ratio_scale_invariance);
    r.run("pls_null_components/dmodx_degenerate_reference", test_dmodx_degenerate_reference);
}
