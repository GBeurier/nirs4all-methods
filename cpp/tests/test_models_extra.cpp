// SPDX-License-Identifier: CECILL-2.1
//
// Model-fixture parity tests for the extra-PLS model families that are
// reachable through dedicated public `n4m_*_fit` entry points returning a
// n4m_method_result_t (rather than the n4m_model_fit / n4m_model_get_array
// path used by the core solver families in test_models_pls.cpp):
//
//   - MB-PLS        n4m_estimators_mb_pls_fit       (block-weighted multi-block PLS)
//   - PLS-LDA       n4m_estimators_pls_lda_fit      (LDA on PLS scores)
//   - PLS-Logistic  n4m_estimators_pls_logistic_fit (multinomial logistic on PLS scores)
//   - LW-PLS        n4m_estimators_lw_pls_fit       (Gaussian-weighted local PLS)
//
// Fixtures + expected arrays come from the generated headers under
// cpp/tests/fixtures/. Per parity/tolerances.md the relevant rows
// (`sklearn/PLSRegression-block-weighted-MBPLS`,
//  `sklearn/PLSRegression-scores-plus-numpy-LDA`,
//  `sklearn/PLSRegression-scores-plus-numpy-logistic`) gate at abs/rel 1e-8.
//
// LW-PLS gates against synthetic_lw_pls_local_window_v1, regenerated from the
// Gaussian-weighted local-PLS algorithm the engine implements (the reference
// mirrors nirs4all lwpls.py::_lwpls_predict bit-for-bit; the generated
// fixture agrees with the C++ core to ~7e-16). The default NIPALS solver
// selects the Gaussian path; neighbor indices are the rows sorted by raw
// squared Euclidean distance and are checked exactly.

#include "n4m/n4m.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "fixtures/lw_pls_fixtures.hpp"
#include "fixtures/mb_pls_fixtures.hpp"
#include "fixtures/pls_lda_fixtures.hpp"
#include "fixtures/pls_logistic_fixtures.hpp"
#include "harness.hpp"

void register_models_extra_tests(n4m_testing::Runner& r);

namespace {

using ::n4m::test::fixtures::MatrixRef;

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

n4m_matrix_view_t make_view(const MatrixRef& ref) {
    n4m_matrix_view_t v{};
    const n4m_status_t st = n4m_matrix_view_init_rowmajor(
        &v, const_cast<double*>(ref.values), ref.rows, ref.cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

void check_close(const char* tag, const double* actual, std::size_t n,
                 const MatrixRef& expected) {
    N4M_TEST_REQUIRE(n == expected.size);
    for (std::size_t i = 0; i < expected.size; ++i) {
        const double diff  = std::fabs(actual[i] - expected.values[i]);
        const double scale = std::max(std::max(std::fabs(actual[i]),
                                               std::fabs(expected.values[i])), 1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            throw std::runtime_error(std::string(tag) + " mismatch at i=" +
                std::to_string(i) + " got=" + std::to_string(actual[i]) +
                " want=" + std::to_string(expected.values[i]) +
                " diff=" + std::to_string(diff));
        }
    }
}

// Read a named double matrix from a method result and compare to a MatrixRef.
void check_result_matrix(const n4m_method_result_t* result, const char* key,
                         const MatrixRef& expected) {
    const double* data = nullptr;
    std::int64_t rows = 0, cols = 0;
    N4M_TEST_REQUIRE(
        n4m_method_result_get_double_matrix(result, key, &data, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(data != nullptr);
    N4M_TEST_REQUIRE(rows == expected.rows);
    N4M_TEST_REQUIRE(cols == expected.cols);
    check_close(key, data, static_cast<std::size_t>(rows * cols), expected);
}

void test_mb_pls() {
    for (const auto& fx : ::n4m::test::fixtures::kMbPlsFixtures) {
        n4m_context_t* ctx = nullptr;
        n4m_config_t*  cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, fx.n_components) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_matrix_view_t Y = make_view(fx.Y);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_mb_pls_fit(
            ctx, cfg, &X, &Y, fx.block_sizes.values,
            static_cast<std::int64_t>(fx.block_sizes.size), &result) == N4M_OK);
        N4M_TEST_REQUIRE(result != nullptr);

        double n_blocks = 0.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, "n_blocks", &n_blocks) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<std::size_t>(n_blocks) == fx.block_sizes.size);

        check_result_matrix(result, "predictions",   fx.predictions);
        check_result_matrix(result, "coefficients",  fx.coefficients);
        check_result_matrix(result, "intercept",     fx.intercept);
        check_result_matrix(result, "x_mean",        fx.x_mean);
        check_result_matrix(result, "x_scale",       fx.x_scale);
        check_result_matrix(result, "block_weights", fx.block_weights);

        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
}

void test_pls_lda() {
    for (const auto& fx : ::n4m::test::fixtures::kPlsLdaFixtures) {
        n4m_context_t* ctx = nullptr;
        n4m_config_t*  cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, fx.n_components) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_pls_lda_fit(
            ctx, cfg, &X, fx.labels.values,
            static_cast<std::int64_t>(fx.labels.size), fx.n_classes, &result) == N4M_OK);
        N4M_TEST_REQUIRE(result != nullptr);

        double n_classes = 0.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, "n_classes", &n_classes) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<std::int32_t>(n_classes) == fx.n_classes);

        check_result_matrix(result, "decision_scores", fx.decision_scores);

        // Integer class predictions — exact match.
        const std::int32_t* pred = nullptr;
        std::int32_t pred_size = 0;
        N4M_TEST_REQUIRE(
            n4m_method_result_get_int_vector(result, "predictions", &pred, &pred_size) == N4M_OK);
        N4M_TEST_REQUIRE(pred != nullptr);
        N4M_TEST_REQUIRE(static_cast<std::size_t>(pred_size) == fx.predictions.size);
        for (std::size_t i = 0; i < fx.predictions.size; ++i) {
            N4M_TEST_REQUIRE(pred[i] == fx.predictions.values[i]);
        }

        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
}

void test_pls_logistic() {
    for (const auto& fx : ::n4m::test::fixtures::kPlsLogisticFixtures) {
        n4m_context_t* ctx = nullptr;
        n4m_config_t*  cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, fx.n_components) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_pls_logistic_fit(
            ctx, cfg, &X, fx.labels.values,
            static_cast<std::int64_t>(fx.labels.size), fx.n_classes, &result) == N4M_OK);
        N4M_TEST_REQUIRE(result != nullptr);

        double n_classes = 0.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, "n_classes", &n_classes) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<std::int32_t>(n_classes) == fx.n_classes);

        check_result_matrix(result, "decision_scores", fx.decision_scores);
        check_result_matrix(result, "probabilities",   fx.probabilities);
        check_result_matrix(result, "intercepts",      fx.intercepts);
        check_result_matrix(result, "coefficients",    fx.coefficients);

        const std::int32_t* pred = nullptr;
        std::int32_t pred_size = 0;
        N4M_TEST_REQUIRE(
            n4m_method_result_get_int_vector(result, "predictions", &pred, &pred_size) == N4M_OK);
        N4M_TEST_REQUIRE(pred != nullptr);
        N4M_TEST_REQUIRE(static_cast<std::size_t>(pred_size) == fx.predictions.size);
        for (std::size_t i = 0; i < fx.predictions.size; ++i) {
            N4M_TEST_REQUIRE(pred[i] == fx.predictions.values[i]);
        }

        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
}

void test_lw_pls() {
    for (const auto& fx : ::n4m::test::fixtures::kLwPlsFixtures) {
        n4m_context_t* ctx = nullptr;
        n4m_config_t*  cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, fx.n_components) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION) == N4M_OK);
        // The default NIPALS solver selects the Gaussian-weighted local PLS the
        // fixture encodes; SIMPLS would opt into the legacy k-NN cutoff variant.
        N4M_TEST_REQUIRE(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_matrix_view_t Y = make_view(fx.Y);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_lw_pls_fit(
            ctx, cfg, &X, &Y, fx.n_neighbors, &result) == N4M_OK);
        N4M_TEST_REQUIRE(result != nullptr);

        double n_neighbors = 0.0;
        N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, "n_neighbors", &n_neighbors) == N4M_OK);
        N4M_TEST_REQUIRE(static_cast<std::int32_t>(n_neighbors) == fx.n_neighbors);

        check_result_matrix(result, "predictions", fx.predictions);

        // Neighbor indices — exact int64 match.
        const std::int64_t* nbr = nullptr;
        std::int64_t nbr_size = 0;
        N4M_TEST_REQUIRE(n4m_method_result_get_int64_vector(
            result, "neighbor_indices_i64", &nbr, &nbr_size) == N4M_OK);
        N4M_TEST_REQUIRE(nbr != nullptr);
        N4M_TEST_REQUIRE(static_cast<std::size_t>(nbr_size) == fx.neighbor_indices.size);
        for (std::size_t i = 0; i < fx.neighbor_indices.size; ++i) {
            N4M_TEST_REQUIRE(nbr[i] == fx.neighbor_indices.values[i]);
        }

        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
}

void test_group_sparse_penalty() {
    // Two targets and non-contiguous group ids exercise the actual C ABI
    // output, including prediction from the penalized coefficient matrix.
    double x_data[] = {
        1, 0, 2, 3,  2, 1, 0, 2,  3, 2, 1, 0,  4, 1, 3, 1,
        5, 3, 2, 4,  6, 2, 4, 2,  7, 4, 1, 5,  8, 3, 5, 3};
    double y_data[16]{};
    for (std::size_t i = 0; i < 8; ++i) {
        y_data[2 * i] = 2 * x_data[4 * i] + 0.3 * x_data[4 * i + 1] -
                        x_data[4 * i + 2] + 1;
        y_data[2 * i + 1] = -x_data[4 * i] + 0.5 * x_data[4 * i + 2] +
                            0.7 * x_data[4 * i + 3] - 2;
    }
    n4m_matrix_view_t X{}, Y{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&X, x_data, 8, 4, N4M_DTYPE_F64) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(&Y, y_data, 8, 2, N4M_DTYPE_F64) == N4M_OK);
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, 2) == N4M_OK);
    const std::int32_t groups[] = {2, 2, 9, 9};
    auto fit = [&](double lambda) {
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_group_sparse_pls_fit(
            ctx, cfg, &X, &Y, groups, 4, lambda, &result) == N4M_OK);
        N4M_TEST_REQUIRE(result != nullptr);
        return result;
    };
    n4m_method_result_t* baseline = fit(0.0);
    n4m_method_result_t* shrunk = fit(0.2);
    n4m_method_result_t* zero = fit(1e6);
    auto matrix = [](n4m_method_result_t* result, const char* key) {
        const double* data = nullptr;
        std::int64_t rows = 0, cols = 0;
        N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
            result, key, &data, &rows, &cols) == N4M_OK);
        return std::vector<double>(data, data + rows * cols);
    };
    const auto b0 = matrix(baseline, "coefficients");
    const auto b1 = matrix(shrunk, "coefficients");
    const auto bz = matrix(zero, "coefficients");
    const auto p0 = matrix(baseline, "predictions");
    const auto p1 = matrix(shrunk, "predictions");
    const auto pz = matrix(zero, "predictions");
    const auto xm = matrix(shrunk, "x_mean");
    const auto ym = matrix(shrunk, "y_mean");
    double n_groups = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(shrunk, "n_groups", &n_groups) == N4M_OK);
    N4M_TEST_REQUIRE(n_groups == 2.0);
    double max_change = 0.0;
    double max_prediction_change = 0.0;
    for (std::size_t g = 0; g < 2; ++g) {
        double norm_sq = 0.0;
        for (std::size_t f = 2 * g; f < 2 * g + 2; ++f)
            for (std::size_t t = 0; t < 2; ++t)
                norm_sq += b0[f * 2 + t] * b0[f * 2 + t];
        const double norm = std::sqrt(norm_sq);
        const double factor = norm > 0.2 ? 1.0 - 0.2 / norm : 0.0;
        for (std::size_t f = 2 * g; f < 2 * g + 2; ++f) {
            for (std::size_t t = 0; t < 2; ++t) {
                const auto k = f * 2 + t;
                N4M_TEST_REQUIRE(std::fabs(b1[k] - b0[k] * factor) < 1e-10);
                N4M_TEST_REQUIRE(bz[k] == 0.0);
                max_change = std::max(max_change, std::fabs(b1[k] - b0[k]));
            }
        }
    }
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t t = 0; t < 2; ++t) {
            double expected = ym[t];
            for (std::size_t f = 0; f < 4; ++f)
                expected += (x_data[i * 4 + f] - xm[f]) * b1[f * 2 + t];
            const auto k = i * 2 + t;
            N4M_TEST_REQUIRE(std::fabs(p1[k] - expected) < 1e-10);
            N4M_TEST_REQUIRE(std::fabs(pz[k] - ym[t]) < 1e-10);
            max_prediction_change = std::max(max_prediction_change, std::fabs(p1[k] - p0[k]));
        }
    }
    N4M_TEST_REQUIRE(max_change > 1e-6);
    N4M_TEST_REQUIRE(max_prediction_change > 1e-6);
    n4m_method_result_t* fused_raw = nullptr;
    n4m_method_result_t* fused_shrunk = nullptr;
    N4M_TEST_REQUIRE(n4m_estimators_fused_sparse_pls_fit(
        ctx, cfg, &X, &Y, 0.0, 0.0, &fused_raw) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_estimators_fused_sparse_pls_fit(
        ctx, cfg, &X, &Y, 0.2, 0.0, &fused_shrunk) == N4M_OK);
    const auto fused_b0 = matrix(fused_raw, "coefficients");
    const auto fused_b1 = matrix(fused_shrunk, "coefficients");
    double fused_change = 0.0;
    for (std::size_t k = 0; k < fused_b0.size(); ++k)
        fused_change = std::max(fused_change, std::fabs(fused_b1[k] - fused_b0[k]));
    N4M_TEST_REQUIRE(fused_change > 1e-6);
    n4m_method_result_t* invalid = nullptr;
    N4M_TEST_REQUIRE(n4m_estimators_group_sparse_pls_fit(
        ctx, cfg, &X, &Y, groups, 4, -0.1, &invalid) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(invalid == nullptr);
    N4M_TEST_REQUIRE(n4m_estimators_group_sparse_pls_fit(
        ctx, cfg, &X, &Y, groups, 4, std::numeric_limits<double>::quiet_NaN(),
        &invalid) == N4M_ERR_INVALID_ARGUMENT);
    const std::int32_t bad_groups[] = {2, -1, 9, 9};
    N4M_TEST_REQUIRE(n4m_estimators_group_sparse_pls_fit(
        ctx, cfg, &X, &Y, bad_groups, 4, 0.2, &invalid) == N4M_ERR_INVALID_ARGUMENT);
    n4m_method_result_destroy(zero);
    n4m_method_result_destroy(shrunk);
    n4m_method_result_destroy(baseline);
    n4m_method_result_destroy(fused_shrunk);
    n4m_method_result_destroy(fused_raw);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_models_extra_tests(n4m_testing::Runner& r) {
    r.run("models_extra/mb_pls",       test_mb_pls);
    r.run("models_extra/pls_lda",      test_pls_lda);
    r.run("models_extra/pls_logistic", test_pls_logistic);
    r.run("models_extra/lw_pls",       test_lw_pls);
    r.run("models_extra/group_sparse_penalty", test_group_sparse_penalty);
}
