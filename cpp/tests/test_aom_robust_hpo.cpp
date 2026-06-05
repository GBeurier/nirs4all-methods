// SPDX-License-Identifier: CECILL-2.1
//
// Smoke tests for native AOM robust-HPO MethodResult ABI.

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"

void register_aom_robust_hpo_tests(n4m_testing::Runner& r);

namespace {

n4m_matrix_view_t view(std::vector<double>& values,
                       std::int64_t rows,
                       std::int64_t cols) {
    n4m_matrix_view_t out{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
                         &out, values.data(), rows, cols, N4M_DTYPE_F64) ==
                     N4M_OK);
    return out;
}

double get_scalar(const n4m_method_result_t* result, const char* key) {
    double value = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, key, &value) == N4M_OK);
    return value;
}

void make_dataset(std::vector<double>& X,
                  std::vector<double>& Y,
                  std::int64_t n,
                  std::int64_t p) {
    X.assign(static_cast<std::size_t>(n * p), 0.0);
    Y.assign(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < p; ++j) {
            const double x =
                std::sin(0.17 * static_cast<double>((i + 1) * (j + 1))) +
                0.03 * static_cast<double>(i) -
                0.01 * static_cast<double>(j);
            X[static_cast<std::size_t>(i * p + j)] = x;
        }
        Y[static_cast<std::size_t>(i)] =
            1.4 * X[static_cast<std::size_t>(i * p + 2)] -
            0.7 * X[static_cast<std::size_t>(i * p + 5)] +
            0.2 * static_cast<double>(i % 3);
    }
}

void test_aom_robust_hpo_compact_smoke() {
    constexpr std::int64_t n = 18;
    constexpr std::int64_t p = 16;
    std::vector<double> X;
    std::vector<double> Y;
    make_dataset(X, Y, n, p);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X, n, p);
    n4m_matrix_view_t Yv = view(Y, n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_robust_hpo_fit(
                         ctx, cfg, &Xv, &Yv,
                         /*profile=*/0,
                         /*cv=*/3,
                         /*heads_mask=*/3,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);

    const double* predictions = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "predictions", &predictions, &rows, &cols) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(predictions != nullptr);
    N4M_TEST_REQUIRE(rows == n);
    N4M_TEST_REQUIRE(cols == 1);
    for (std::int64_t i = 0; i < n; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(predictions[static_cast<std::size_t>(i)]));
    }

    const double* input_coefficients = nullptr;
    std::int64_t coef_rows = 0;
    std::int64_t coef_cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "input_coefficients",
                         &input_coefficients, &coef_rows, &coef_cols) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(input_coefficients != nullptr);
    N4M_TEST_REQUIRE(coef_rows == p);
    N4M_TEST_REQUIRE(coef_cols == 1);

    const double* intercept = nullptr;
    std::int64_t intercept_rows = 0;
    std::int64_t intercept_cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "intercept", &intercept,
                         &intercept_rows, &intercept_cols) == N4M_OK);
    N4M_TEST_REQUIRE(intercept_rows == 1);
    N4M_TEST_REQUIRE(intercept_cols == 1);
    for (std::int64_t i = 0; i < n; ++i) {
        double replay = intercept[0];
        for (std::int64_t j = 0; j < p; ++j) {
            replay += X[static_cast<std::size_t>(i * p + j)] *
                      input_coefficients[static_cast<std::size_t>(j)];
        }
        N4M_TEST_REQUIRE(std::fabs(
            replay - predictions[static_cast<std::size_t>(i)]) <= 1e-8);
    }

    const double* scores = nullptr;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "candidate_scores", &scores, &rows, &cols) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(scores != nullptr);
    N4M_TEST_REQUIRE(rows > 0);
    N4M_TEST_REQUIRE(cols == 4);
    bool saw_finite = false;
    for (std::int64_t i = 0; i < rows; ++i) {
        const double score = scores[static_cast<std::size_t>(i * cols + 3)];
        saw_finite = saw_finite || std::isfinite(score);
    }
    N4M_TEST_REQUIRE(saw_finite);

    double selected_cv_rmse = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(
                         result, "selected_cv_rmse", &selected_cv_rmse) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(std::isfinite(selected_cv_rmse));
    const double n_chains = get_scalar(result, "n_chains");
    N4M_TEST_REQUIRE(std::fabs(n_chains - 12.0) <= 1e-12);
    const double n_features = get_scalar(result, "n_features");
    N4M_TEST_REQUIRE(std::fabs(n_features - static_cast<double>(p)) <= 1e-12);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_robust_hpo_wide_includes_fck_moment_bank() {
    constexpr std::int64_t n = 24;
    constexpr std::int64_t p = 16;
    constexpr std::int64_t n_chains = 31;
    constexpr std::int64_t n_ridge_lambdas = 4;
    constexpr std::int64_t n_candidates = n_chains * n_ridge_lambdas;
    std::vector<double> X;
    std::vector<double> Y;
    make_dataset(X, Y, n, p);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X, n, p);
    n4m_matrix_view_t Yv = view(Y, n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_robust_hpo_fit(
                         ctx, cfg, &Xv, &Yv,
                         /*profile=*/1,
                         /*cv=*/3,
                         /*heads_mask=*/1,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    N4M_TEST_REQUIRE(get_scalar(result, "n_chains") ==
                     static_cast<double>(n_chains));
    N4M_TEST_REQUIRE(get_scalar(result, "n_candidates") ==
                     static_cast<double>(n_candidates));

    const double* scores = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, "candidate_scores", &scores, &rows, &cols) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(scores != nullptr);
    N4M_TEST_REQUIRE(rows == n_candidates);
    N4M_TEST_REQUIRE(cols == 4);
    bool saw_last_chain = false;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto off = static_cast<std::size_t>(row * cols);
        saw_last_chain = saw_last_chain ||
                         (std::llround(scores[off]) == (n_chains - 1));
    }
    N4M_TEST_REQUIRE(saw_last_chain);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_aom_robust_hpo_tests(n4m_testing::Runner& r) {
    r.run("aom_robust_hpo_compact_smoke", test_aom_robust_hpo_compact_smoke);
    r.run("aom_robust_hpo/wide_includes_fck_moment_bank",
          test_aom_robust_hpo_wide_includes_fck_moment_bank);
}
