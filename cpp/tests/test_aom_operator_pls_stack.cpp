// SPDX-License-Identifier: CECILL-2.1
//
// Public ABI tests for native AOM operator PLS score stack.

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"

void register_aom_operator_pls_stack_tests(n4m_testing::Runner& r);

namespace {

constexpr double kTol = 1e-9;

n4m_matrix_view_t view(double* data, std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t out{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
                         &out, data, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return out;
}

std::vector<double> get_matrix(const n4m_method_result_t* result,
                               const char* key,
                               std::int64_t rows,
                               std::int64_t cols) {
    const double* data = nullptr;
    std::int64_t got_rows = 0;
    std::int64_t got_cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, key, &data, &got_rows, &got_cols) == N4M_OK);
    N4M_TEST_REQUIRE(got_rows == rows);
    N4M_TEST_REQUIRE(got_cols == cols);
    const auto n = static_cast<std::size_t>(rows * cols);
    return std::vector<double>(data, data + n);
}

std::vector<std::int32_t> get_i32(const n4m_method_result_t* result,
                                  const char* key,
                                  std::int64_t n) {
    const std::int32_t* data = nullptr;
    std::int32_t got_n = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         result, key, &data, &got_n) == N4M_OK);
    N4M_TEST_REQUIRE(static_cast<std::int64_t>(got_n) == n);
    return std::vector<std::int32_t>(data, data + static_cast<std::size_t>(n));
}

double get_scalar(const n4m_method_result_t* result, const char* key) {
    double value = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, key, &value) == N4M_OK);
    return value;
}

void make_dataset(std::vector<double>& X, std::vector<double>& y,
                  std::int64_t n, std::int64_t p) {
    X.assign(static_cast<std::size_t>(n * p), 0.0);
    y.assign(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < p; ++j) {
            const double ii = static_cast<double>(i + 1);
            const double jj = static_cast<double>(j + 1);
            X[static_cast<std::size_t>(i * p + j)] =
                std::sin(0.07 * ii * jj) +
                0.35 * std::cos(0.03 * (ii + 2.0 * jj)) +
                0.01 * (ii - jj);
        }
        y[static_cast<std::size_t>(i)] =
            0.65 * X[static_cast<std::size_t>(i * p + 1)] -
            0.30 * X[static_cast<std::size_t>(i * p + 5)] +
            0.20 * X[static_cast<std::size_t>(i * p + 11)] +
            0.02 * std::sin(0.31 * static_cast<double>(i));
    }
}

void test_operator_pls_stack_compact_contract() {
    constexpr std::int64_t n = 24;
    constexpr std::int64_t p = 18;
    constexpr std::int32_t cv = 4;
    constexpr std::int64_t n_operators = 12;
    constexpr std::int64_t n_specs = 4;
    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t components[2] = {1, 2};
    const double alphas[2] = {0.01, 1.0};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X.data(), n, p);
    n4m_matrix_view_t Yv = view(y.data(), n, 1);

    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_ensemble_aom_operator_pls_stack_fit(
                         ctx, cfg, &Xv, &Yv,
                         /*profile=*/0,
                         cv,
                         folds,
                         n,
                         components,
                         2,
                         alphas,
                         2,
                         /*std_penalty=*/0.0,
                         /*gap_penalty=*/0.0,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);

    const auto scores = get_matrix(result, "candidate_scores", n_specs, 7);
    const auto fold_scores = get_matrix(result, "fold_scores", n_specs, cv);
    (void)fold_scores;
    const auto oof = get_matrix(result, "oof_predictions", n, 1);
    const auto preds = get_matrix(result, "predictions", n, 1);
    const auto folds_out = get_i32(result, "fold_ids", n);
    for (std::int64_t i = 0; i < n; ++i) {
        N4M_TEST_REQUIRE(folds_out[static_cast<std::size_t>(i)] == folds[i]);
        N4M_TEST_REQUIRE(std::isfinite(oof[static_cast<std::size_t>(i)]));
    }

    std::int64_t best_spec = 0;
    for (std::int64_t s = 0; s < n_specs; ++s) {
        N4M_TEST_REQUIRE(scores[static_cast<std::size_t>(s * 7 + 0)] ==
                         static_cast<double>(s));
        N4M_TEST_REQUIRE(std::isfinite(scores[static_cast<std::size_t>(s * 7 + 3)]));
        N4M_TEST_REQUIRE(std::isfinite(scores[static_cast<std::size_t>(s * 7 + 6)]));
        if (scores[static_cast<std::size_t>(s * 7 + 6)] <
            scores[static_cast<std::size_t>(best_spec * 7 + 6)]) {
            best_spec = s;
        }
    }
    N4M_TEST_REQUIRE(static_cast<std::int64_t>(
                         std::llround(get_scalar(result, "selected_spec_id"))) ==
                     best_spec);
    N4M_TEST_REQUIRE(get_scalar(result, "n_operators") ==
                     static_cast<double>(n_operators));
    N4M_TEST_REQUIRE(get_scalar(result, "n_specs") ==
                     static_cast<double>(n_specs));

    const auto n_features = static_cast<std::int64_t>(
        std::llround(get_scalar(result, "n_operator_features")));
    N4M_TEST_REQUIRE(n_features >= n_operators);
    const auto features = get_matrix(result, "stack_features", n, n_features);
    const auto coef = get_matrix(result, "coefficients", n_features, 1);
    const auto intercept = get_matrix(result, "intercept", 1, 1);
    const auto input_coef = get_matrix(result, "input_coefficients", p, 1);
    const auto input_intercept = get_matrix(result, "input_intercept", 1, 1);
    const auto offsets = get_i32(result, "operator_feature_offsets",
                                 n_operators + 1);
    N4M_TEST_REQUIRE(offsets.front() == 0);
    N4M_TEST_REQUIRE(offsets.back() == n_features);

    for (std::int64_t r = 0; r < n; ++r) {
        double acc = intercept[0];
        for (std::int64_t c = 0; c < n_features; ++c) {
            acc += features[static_cast<std::size_t>(r * n_features + c)] *
                   coef[static_cast<std::size_t>(c)];
        }
        N4M_TEST_REQUIRE(std::fabs(acc - preds[static_cast<std::size_t>(r)]) <= kTol);
        double input_acc = input_intercept[0];
        for (std::int64_t c = 0; c < p; ++c) {
            input_acc += X[static_cast<std::size_t>(r * p + c)] *
                         input_coef[static_cast<std::size_t>(c)];
        }
        N4M_TEST_REQUIRE(std::fabs(input_acc -
                                   preds[static_cast<std::size_t>(r)]) <= 1e-8);
    }

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_operator_pls_stack_rejects_multi_output_y() {
    constexpr std::int64_t n = 12;
    constexpr std::int64_t p = 12;
    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::vector<double> Y2(static_cast<std::size_t>(n * 2), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        Y2[static_cast<std::size_t>(i * 2 + 0)] = y[static_cast<std::size_t>(i)];
        Y2[static_cast<std::size_t>(i * 2 + 1)] = 0.5 * y[static_cast<std::size_t>(i)];
    }
    const std::int32_t components[1] = {1};
    const double alphas[1] = {0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X.data(), n, p);
    n4m_matrix_view_t Yv = view(Y2.data(), n, 2);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_ensemble_aom_operator_pls_stack_fit(
                         ctx, cfg, &Xv, &Yv, 0, 3,
                         nullptr, 0, components, 1, alphas, 1,
                         0.0, 0.0, &result) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_operator_pls_stack_wide_includes_fck_moment_bank() {
    constexpr std::int64_t n = 24;
    constexpr std::int64_t p = 18;
    constexpr std::int32_t cv = 4;
    constexpr std::int64_t n_operators = 31;
    constexpr std::int64_t n_specs = 1;
    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t components[1] = {1};
    const double alphas[1] = {0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X.data(), n, p);
    n4m_matrix_view_t Yv = view(y.data(), n, 1);

    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_ensemble_aom_operator_pls_stack_fit(
                         ctx, cfg, &Xv, &Yv,
                         /*profile=*/1,
                         cv,
                         folds,
                         n,
                         components,
                         1,
                         alphas,
                         1,
                         /*std_penalty=*/0.0,
                         /*gap_penalty=*/0.0,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    N4M_TEST_REQUIRE(get_scalar(result, "n_operators") ==
                     static_cast<double>(n_operators));
    N4M_TEST_REQUIRE(get_scalar(result, "n_specs") ==
                     static_cast<double>(n_specs));
    const auto n_features = static_cast<std::int64_t>(
        std::llround(get_scalar(result, "n_operator_features")));
    N4M_TEST_REQUIRE(n_features >= n_operators);
    const auto offsets = get_i32(result, "operator_feature_offsets",
                                 n_operators + 1);
    N4M_TEST_REQUIRE(offsets.front() == 0);
    N4M_TEST_REQUIRE(offsets.back() == n_features);
    const auto scores = get_matrix(result, "candidate_scores", n_specs, 7);
    N4M_TEST_REQUIRE(std::isfinite(scores[6]));

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_aom_operator_pls_stack_tests(n4m_testing::Runner& r) {
    r.run("aom_operator_pls_stack/compact_contract",
          test_operator_pls_stack_compact_contract);
    r.run("aom_operator_pls_stack/rejects_multi_output_y",
          test_operator_pls_stack_rejects_multi_output_y);
    r.run("aom_operator_pls_stack/wide_includes_fck_moment_bank",
          test_operator_pls_stack_wide_includes_fck_moment_bank);
}
