// SPDX-License-Identifier: CECILL-2.1
//
// Phase 19 — NIRS regression metrics + Hotelling T² / Q-residuals tests.
//
// Smoke:  hand-built tiny inputs verify each metric and the multivariate
//         statistics on trivial structures (perfect prediction → RMSE = 0,
//         and constant-mean centring → T² indistinguishable across rows).
// Parity: load `nirs_metrics_v1.json`, `hotelling_t2_v1.json`, and
//         `q_residuals_v1.json`, replay each case, and assert tolerances
//         from the brief.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef N4M_PARITY_FIXTURE_DIR
#  error "N4M_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Smoke tests.
// ---------------------------------------------------------------------------

void test_metric_smoke_perfect() {
    const double y_true[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y_pred[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double rmse = -1.0, mae = -1.0, bias = -1.0, sep = -1.0;
    N4M_TEST_REQUIRE(n4m_metric_rmse(y_true, y_pred, 5, &rmse) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_metric_mae(y_true, y_pred, 5, &mae) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_metric_bias(y_true, y_pred, 5, &bias) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_metric_sep(y_true, y_pred, 5, &sep) == N4M_OK);
    N4M_TEST_REQUIRE(rmse == 0.0);
    N4M_TEST_REQUIRE(mae == 0.0);
    N4M_TEST_REQUIRE(bias == 0.0);
    N4M_TEST_REQUIRE(sep == 0.0);
    double r2 = -1.0;
    N4M_TEST_REQUIRE(n4m_metric_r2(y_true, y_pred, 5, &r2) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(r2 - 1.0) < 1e-15);
}

void test_metric_smoke_constant_bias() {
    const double y_true[4] = {1.0, 2.0, 3.0, 4.0};
    const double y_pred[4] = {1.5, 2.5, 3.5, 4.5};   // y_pred = y_true + 0.5
    double bias = 0.0;
    N4M_TEST_REQUIRE(n4m_metric_bias(y_true, y_pred, 4, &bias) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(bias - 0.5) < 1e-15);
    double rmse = 0.0;
    N4M_TEST_REQUIRE(n4m_metric_rmse(y_true, y_pred, 4, &rmse) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(rmse - 0.5) < 1e-15);
    double mae = 0.0;
    N4M_TEST_REQUIRE(n4m_metric_mae(y_true, y_pred, 4, &mae) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(mae - 0.5) < 1e-15);
    // SEP = std of (y_pred - y_true) = std([0.5, 0.5, 0.5, 0.5]) = 0.
    double sep = -1.0;
    N4M_TEST_REQUIRE(n4m_metric_sep(y_true, y_pred, 4, &sep) == N4M_OK);
    N4M_TEST_REQUIRE(sep == 0.0);
    // RPD = std(y_true) / SEP → +inf when SEP == 0.
    double rpd = 0.0;
    N4M_TEST_REQUIRE(n4m_metric_rpd(y_true, y_pred, 4, &rpd) == N4M_OK);
    N4M_TEST_REQUIRE(std::isinf(rpd));
}

void test_metric_smoke_invalid_n() {
    const double y[1] = {0.0};
    double out = 0.0;
    N4M_TEST_REQUIRE(n4m_metric_rmse(y, y, 0, &out)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_metric_mae(y, y, -3, &out)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_metric_rmse(nullptr, y, 1, &out)
                     == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(n4m_metric_rmse(y, y, 1, nullptr)
                     == N4M_ERR_NULL_POINTER);
}

void test_hotelling_t2_smoke() {
    // 6x3 well-conditioned data, k=2 components, alpha=0.05.
    double X[6 * 3] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        2.0, 3.0, 4.0,
        5.0, 6.0, 7.0,
        3.0, 4.0, 5.0,
    };
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 6, 3);
    double t2[6] = {0};
    double ucl = -1.0;
    const n4m_status_t st = n4m_util_hotelling_t2(
        Xv, /*n_components=*/2, /*alpha=*/0.05, t2, 6, &ucl);
    N4M_TEST_REQUIRE(st == N4M_OK);
    for (int i = 0; i < 6; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(t2[i]));
    }
    N4M_TEST_REQUIRE(ucl > 0.0);

    // Invalid args.
    N4M_TEST_REQUIRE(n4m_util_hotelling_t2(Xv, 0, 0.05, t2, 6, &ucl)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_util_hotelling_t2(Xv, 2, 1.5, t2, 6, &ucl)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_util_hotelling_t2(Xv, 2, 0.05, t2, 5, &ucl)
                     == N4M_ERR_SHAPE_MISMATCH);
}

void test_q_residuals_smoke() {
    // Same 6x3 inputs as the T² smoke.
    double X[6 * 3] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        2.0, 3.0, 4.0,
        5.0, 6.0, 7.0,
        3.0, 4.0, 5.0,
    };
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 6, 3);
    double q[6] = {0};
    double ucl = -1.0;
    const n4m_status_t st = n4m_util_q_residuals(
        Xv, /*n_components=*/2, /*alpha=*/0.05, q, 6, &ucl);
    N4M_TEST_REQUIRE(st == N4M_OK);
    for (int i = 0; i < 6; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(q[i]));
        N4M_TEST_REQUIRE(q[i] >= -1e-10);  // residual energy is non-negative
    }
    N4M_TEST_REQUIRE(std::isfinite(ucl));
}

// ---------------------------------------------------------------------------
// Metric parity tests.
// ---------------------------------------------------------------------------

struct MetricCase {
    std::string         name;
    std::int64_t        n = 0;
    std::vector<double> y_true;
    std::vector<double> y_pred;
    // expected metric values keyed by name
    double exp_rmse = 0.0, exp_mae = 0.0, exp_bias = 0.0, exp_sep = 0.0;
    double exp_rpd = 0.0,  exp_rpiq = 0.0, exp_r2 = 0.0, exp_nrmse = 0.0;
};

double decode_hex_double_value(const std::string& body, const std::string& key,
                                std::size_t lo, std::size_t hi) {
    std::size_t pos = ::n4m_testing::find_value_for_key(body, key, lo, hi);
    return ::n4m_testing::parse_hex_double(body, pos);
}

std::vector<MetricCase> load_metric_cases() {
    const std::string body = ::n4m_testing::slurp(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/nirs_metrics_v1.json");
    auto [cases_lo, cases_hi] = ::n4m_testing::find_cases_array(body);
    auto spans = ::n4m_testing::list_case_object_spans(body, cases_lo,
                                                        cases_hi);
    std::vector<MetricCase> out;
    for (const auto& [lo, hi] : spans) {
        MetricCase c;
        std::size_t np = ::n4m_testing::find_value_for_key(body, "name",
                                                            lo, hi);
        c.name = ::n4m_testing::parse_string(body, np);
        c.n = ::n4m_testing::parse_int64(body,
            ::n4m_testing::find_value_for_key(body, "n", lo, hi));
        std::size_t ytp = ::n4m_testing::find_value_for_key(body, "y_true_hex",
                                                              lo, hi);
        ::n4m_testing::parse_hex_double_array(body, ytp, c.y_true);
        std::size_t ypp = ::n4m_testing::find_value_for_key(body, "y_pred_hex",
                                                              lo, hi);
        ::n4m_testing::parse_hex_double_array(body, ypp, c.y_pred);
        // metrics_hex is a nested object — locate its span and decode by key.
        std::size_t mp = ::n4m_testing::find_value_for_key(body, "metrics_hex",
                                                            lo, hi);
        auto [ml, mh] = ::n4m_testing::read_value_span(body, mp);
        c.exp_rmse  = decode_hex_double_value(body, "rmse",  ml, mh);
        c.exp_mae   = decode_hex_double_value(body, "mae",   ml, mh);
        c.exp_bias  = decode_hex_double_value(body, "bias",  ml, mh);
        c.exp_sep   = decode_hex_double_value(body, "sep",   ml, mh);
        c.exp_rpd   = decode_hex_double_value(body, "rpd",   ml, mh);
        c.exp_rpiq  = decode_hex_double_value(body, "rpiq",  ml, mh);
        c.exp_r2    = decode_hex_double_value(body, "r2",    ml, mh);
        c.exp_nrmse = decode_hex_double_value(body, "nrmse", ml, mh);
        out.push_back(std::move(c));
    }
    return out;
}

void check_one(double got, double want, const std::string& tag) {
    if (std::isinf(want)) {
        if (!std::isinf(got)) {
            throw std::runtime_error(tag + " expected +inf, got " +
                                      std::to_string(got));
        }
        return;
    }
    const double abs_diff = std::fabs(got - want);
    const double ref      = std::max(std::fabs(got), std::fabs(want));
    if (abs_diff <= 1e-13) return;
    if (ref > 0.0 && abs_diff / ref <= 1e-13) return;
    throw std::runtime_error(tag + " got=" + std::to_string(got) +
                              " want=" + std::to_string(want) +
                              " |diff|=" + std::to_string(abs_diff));
}

void verify_metrics_parity() {
    auto cases = load_metric_cases();
    N4M_TEST_REQUIRE(!cases.empty());
    for (const auto& c : cases) {
        double v = 0.0;
        N4M_TEST_REQUIRE(n4m_metric_rmse(c.y_true.data(), c.y_pred.data(),
                                          c.n, &v) == N4M_OK);
        check_one(v, c.exp_rmse,  c.name + "/rmse");
        N4M_TEST_REQUIRE(n4m_metric_mae(c.y_true.data(), c.y_pred.data(),
                                         c.n, &v) == N4M_OK);
        check_one(v, c.exp_mae,   c.name + "/mae");
        N4M_TEST_REQUIRE(n4m_metric_bias(c.y_true.data(), c.y_pred.data(),
                                          c.n, &v) == N4M_OK);
        check_one(v, c.exp_bias,  c.name + "/bias");
        N4M_TEST_REQUIRE(n4m_metric_sep(c.y_true.data(), c.y_pred.data(),
                                         c.n, &v) == N4M_OK);
        check_one(v, c.exp_sep,   c.name + "/sep");
        N4M_TEST_REQUIRE(n4m_metric_rpd(c.y_true.data(), c.y_pred.data(),
                                         c.n, &v) == N4M_OK);
        check_one(v, c.exp_rpd,   c.name + "/rpd");
        N4M_TEST_REQUIRE(n4m_metric_rpiq(c.y_true.data(), c.y_pred.data(),
                                          c.n, &v) == N4M_OK);
        check_one(v, c.exp_rpiq,  c.name + "/rpiq");
        N4M_TEST_REQUIRE(n4m_metric_r2(c.y_true.data(), c.y_pred.data(),
                                        c.n, &v) == N4M_OK);
        check_one(v, c.exp_r2,    c.name + "/r2");
        N4M_TEST_REQUIRE(n4m_metric_nrmse(c.y_true.data(), c.y_pred.data(),
                                           c.n, &v) == N4M_OK);
        check_one(v, c.exp_nrmse, c.name + "/nrmse");
    }
}

// ---------------------------------------------------------------------------
// T² / Q parity tests.
// ---------------------------------------------------------------------------

struct PCAStatCase {
    std::string         name;
    std::int64_t        rows = 0;
    std::int64_t        cols = 0;
    std::int32_t        n_components = 0;
    double              alpha = 0.05;
    std::vector<double> input;
    std::vector<double> stat;   // either t2 or q
    double              ucl = 0.0;
};

std::vector<PCAStatCase> load_pca_cases(const std::string& filename,
                                         const std::string& stat_key) {
    const std::string body = ::n4m_testing::slurp(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/" + filename);
    auto [cases_lo, cases_hi] = ::n4m_testing::find_cases_array(body);
    auto spans = ::n4m_testing::list_case_object_spans(body, cases_lo,
                                                        cases_hi);
    std::vector<PCAStatCase> out;
    for (const auto& [lo, hi] : spans) {
        PCAStatCase c;
        std::size_t np = ::n4m_testing::find_value_for_key(body, "name",
                                                            lo, hi);
        c.name = ::n4m_testing::parse_string(body, np);
        c.rows = ::n4m_testing::parse_int64(body,
            ::n4m_testing::find_value_for_key(body, "rows", lo, hi));
        c.cols = ::n4m_testing::parse_int64(body,
            ::n4m_testing::find_value_for_key(body, "cols", lo, hi));
        c.n_components = static_cast<int32_t>(
            ::n4m_testing::parse_int64(body,
                ::n4m_testing::find_value_for_key(body, "n_components",
                                                   lo, hi)));
        c.alpha = std::strtod(
            body.c_str() + ::n4m_testing::find_value_for_key(body, "alpha",
                                                              lo, hi),
            nullptr);
        std::size_t ip = ::n4m_testing::find_value_for_key(body, "input_hex",
                                                            lo, hi);
        ::n4m_testing::parse_hex_double_array(body, ip, c.input);
        std::size_t sp = ::n4m_testing::find_value_for_key(body, stat_key,
                                                            lo, hi);
        ::n4m_testing::parse_hex_double_array(body, sp, c.stat);
        std::size_t up = ::n4m_testing::find_value_for_key(body, "ucl_hex",
                                                            lo, hi);
        c.ucl = ::n4m_testing::parse_hex_double(body, up);
        out.push_back(std::move(c));
    }
    return out;
}

void verify_hotelling_t2_parity() {
    auto cases = load_pca_cases("hotelling_t2_v1.json", "t2_hex");
    N4M_TEST_REQUIRE(!cases.empty());
    for (const auto& c : cases) {
        std::vector<double> X = c.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), c.rows, c.cols);
        std::vector<double> t2(static_cast<std::size_t>(c.rows), 0.0);
        double ucl = -1.0;
        N4M_TEST_REQUIRE(n4m_util_hotelling_t2(
            Xv, c.n_components, c.alpha, t2.data(), c.rows, &ucl)
            == N4M_OK);
        ::n4m_testing::assert_close(t2, c.stat,
                                     "hotelling_t2/" + c.name,
                                     /*abs_tol=*/1e-10,
                                     /*rel_tol=*/1e-11);
        const double ducl = std::fabs(ucl - c.ucl);
        const double refu = std::max(std::fabs(ucl), std::fabs(c.ucl));
        // The UCL involves a numerical F-quantile inversion; use a slightly
        // looser tolerance (~1e-6 rel) than the per-sample T².
        if (ducl > 1e-6 && (refu == 0.0 || ducl / refu > 1e-6)) {
            throw std::runtime_error(
                "hotelling_t2/" + c.name + " UCL mismatch: got=" +
                std::to_string(ucl) + " want=" + std::to_string(c.ucl) +
                " |diff|=" + std::to_string(ducl));
        }
    }
}

void verify_q_residuals_parity() {
    auto cases = load_pca_cases("q_residuals_v1.json", "q_hex");
    N4M_TEST_REQUIRE(!cases.empty());
    for (const auto& c : cases) {
        std::vector<double> X = c.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), c.rows, c.cols);
        std::vector<double> q(static_cast<std::size_t>(c.rows), 0.0);
        double ucl = -1.0;
        N4M_TEST_REQUIRE(n4m_util_q_residuals(
            Xv, c.n_components, c.alpha, q.data(), c.rows, &ucl)
            == N4M_OK);
        ::n4m_testing::assert_close(q, c.stat,
                                     "q_residuals/" + c.name,
                                     /*abs_tol=*/1e-10,
                                     /*rel_tol=*/1e-11);
        const double ducl = std::fabs(ucl - c.ucl);
        const double refu = std::max(std::fabs(ucl), std::fabs(c.ucl));
        // UCL uses an inverse-normal approximation; tolerance ~1e-5 rel.
        if (ducl > 1e-6 && (refu == 0.0 || ducl / refu > 1e-5)) {
            throw std::runtime_error(
                "q_residuals/" + c.name + " UCL mismatch: got=" +
                std::to_string(ucl) + " want=" + std::to_string(c.ucl) +
                " |diff|=" + std::to_string(ducl));
        }
    }
}

}  // namespace

void register_metrics_tests(n4m_testing::Runner& r);
void register_metrics_tests(n4m_testing::Runner& r) {
    r.run("metric_smoke_perfect",          test_metric_smoke_perfect);
    r.run("metric_smoke_constant_bias",    test_metric_smoke_constant_bias);
    r.run("metric_smoke_invalid_n",        test_metric_smoke_invalid_n);
    r.run("hotelling_t2_smoke",            test_hotelling_t2_smoke);
    r.run("q_residuals_smoke",             test_q_residuals_smoke);
    r.run("metrics_parity",                verify_metrics_parity);
    r.run("hotelling_t2_parity",           verify_hotelling_t2_parity);
    r.run("q_residuals_parity",            verify_q_residuals_parity);
}
