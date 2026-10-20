// SPDX-License-Identifier: CECILL-2.1
//
// Consumers of fitted PLS / PCR models with null components (components
// flagged in N4M_MODEL_UNRESOLVED_COMPONENTS with zero W, P, R, T columns):
// the PLS diagnostics and monitoring. Every expected value is written down
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

}  // namespace

void register_pls_null_component_tests(n4m_testing::Runner& r) {
    r.run("pls_null_components/diagnostics", test_diagnostics_null_component);
    r.run("pls_null_components/diagnostics_pcr_kept_direction", test_diagnostics_pcr_kept_direction);
    r.run("pls_null_components/diagnostics_no_direction", test_diagnostics_no_direction);
    r.run("pls_null_components/monitoring", test_monitoring_null_component);
    r.run("pls_null_components/diagnostics_autoscaled", test_diagnostics_autoscaled);
}
