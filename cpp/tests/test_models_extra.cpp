// SPDX-License-Identifier: CECILL-2.1
//
// Model-fixture parity tests for the extra-PLS model families that are
// reachable through dedicated public `n4m_*_fit` entry points returning a
// n4m_method_result_t (rather than the n4m_model_fit / n4m_model_get_array
// path used by the core solver families in test_models_pls.cpp):
//
//   - MB-PLS        n4m_mb_pls_fit       (block-weighted multi-block PLS)
//   - PLS-LDA       n4m_pls_lda_fit      (LDA on PLS scores)
//   - PLS-Logistic  n4m_pls_logistic_fit (multinomial logistic on PLS scores)
//
// Fixtures + expected arrays come from the generated headers under
// cpp/tests/fixtures/. Per parity/tolerances.md the relevant rows
// (`sklearn/PLSRegression-block-weighted-MBPLS`,
//  `sklearn/PLSRegression-scores-plus-numpy-LDA`,
//  `sklearn/PLSRegression-scores-plus-numpy-logistic`) gate at abs/rel 1e-8.
//
// LW-PLS (`n4m_lw_pls_fit`) is intentionally NOT wired here: the committed
// synthetic_lw_pls_local_window_v1 fixture predates a 2026-05-23 change to
// cpp/src/core/lw_pls.cpp and no longer matches the engine — neither the
// predictions nor the kNN neighbor indices agree (verified directly against
// both the public ABI shim and the internal core fit_predict_lw_pls). That is
// a pre-existing engine/fixture drift, not a property of the C ABI, so per the
// "correctness over coverage" rule the family is skipped and flagged for a
// separate fixture-regeneration / parity fix.

#include "n4m/n4m.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fixtures/mb_pls_fixtures.hpp"
#include "fixtures/pls_lda_fixtures.hpp"
#include "fixtures/pls_logistic_fixtures.hpp"
#include "harness.hpp"

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
        N4M_TEST_REQUIRE(n4m_mb_pls_fit(
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
        N4M_TEST_REQUIRE(n4m_pls_lda_fit(
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
        N4M_TEST_REQUIRE(n4m_pls_logistic_fit(
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

}  // namespace

void register_models_extra_tests(n4m_testing::Runner& r) {
    r.run("models_extra/mb_pls",       test_mb_pls);
    r.run("models_extra/pls_lda",      test_pls_lda);
    r.run("models_extra/pls_logistic", test_pls_logistic);
}
