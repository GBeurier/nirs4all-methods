// SPDX-License-Identifier: CECILL-2.1
//
// Public ABI tests for native AOM Ridge simplex blender.

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"

void register_aom_ridge_blender_tests(n4m_testing::Runner& r);

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
                std::sin(0.11 * ii * jj) +
                0.2 * std::cos(0.05 * (ii + jj)) +
                0.015 * (ii - 0.5 * jj);
        }
        y[static_cast<std::size_t>(i)] =
            0.9 * X[static_cast<std::size_t>(i * p + 0)] -
            0.35 * X[static_cast<std::size_t>(i * p + 4)] +
            0.15 * X[static_cast<std::size_t>(i * p + 9)] +
            0.02 * std::sin(0.4 * static_cast<double>(i));
    }
}

void test_aom_ridge_blender_compact_contract_and_weighted_predictions() {
    constexpr std::int64_t n = 18;
    constexpr std::int64_t p = 16;
    constexpr std::int32_t cv = 3;
    constexpr std::int64_t n_chains = 12;
    constexpr std::int64_t n_lambdas = 2;
    constexpr std::int64_t n_candidates = n_chains * n_lambdas;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 1.0};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X.data(), n, p);
    n4m_matrix_view_t Yv = view(y.data(), n, 1);

    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_ensemble_aom_ridge_blender_fit(
                         ctx, cfg, &Xv, &Yv,
                         /*profile=*/0,
                         cv,
                         folds,
                         n,
                         lambdas,
                         n_lambdas,
                         /*regularizer=*/0.01,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);

    const auto scores = get_matrix(result, "candidate_scores", n_candidates, 5);
    const auto weights = get_matrix(result, "weights", 1, n_candidates);
    const auto preds = get_matrix(result, "predictions", n, 1);
    const auto oof = get_matrix(result, "oof_predictions", n, 1);
    const auto input_coefficients =
        get_matrix(result, "input_coefficients", p, 1);
    const auto intercept = get_matrix(result, "intercept", 1, 1);
    const auto candidate_preds =
        get_matrix(result, "candidate_predictions", n, n_candidates);
    const auto candidate_oof =
        get_matrix(result, "oof_candidate_predictions", n, n_candidates);

    double weight_sum = 0.0;
    std::int64_t selected_by_weight = 0;
    for (std::int64_t c = 0; c < n_candidates; ++c) {
        const double weight = weights[static_cast<std::size_t>(c)];
        N4M_TEST_REQUIRE(weight >= -1e-12);
        weight_sum += weight;
        if (weight > weights[static_cast<std::size_t>(selected_by_weight)]) {
            selected_by_weight = c;
        }
        N4M_TEST_REQUIRE(std::fabs(scores[static_cast<std::size_t>(c * 5 + 4)] -
                                   weight) <= 1e-12);
    }
    N4M_TEST_REQUIRE(std::fabs(weight_sum - 1.0) <= 1e-10);

    for (std::int64_t row = 0; row < n; ++row) {
        double final_acc = 0.0;
        double oof_acc = 0.0;
        for (std::int64_t c = 0; c < n_candidates; ++c) {
            const auto off = static_cast<std::size_t>(row * n_candidates + c);
            const double weight = weights[static_cast<std::size_t>(c)];
            final_acc += candidate_preds[off] * weight;
            oof_acc += candidate_oof[off] * weight;
        }
        N4M_TEST_REQUIRE(std::fabs(final_acc -
                                   preds[static_cast<std::size_t>(row)]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(oof_acc -
                                   oof[static_cast<std::size_t>(row)]) <= kTol);
        double replay = intercept[0];
        for (std::int64_t c = 0; c < p; ++c) {
            replay += X[static_cast<std::size_t>(row * p + c)] *
                      input_coefficients[static_cast<std::size_t>(c)];
        }
        N4M_TEST_REQUIRE(std::fabs(replay -
                                   preds[static_cast<std::size_t>(row)]) <= 1e-8);
    }

    const double selected_candidate_id =
        get_scalar(result, "selected_candidate_id");
    N4M_TEST_REQUIRE(static_cast<std::int64_t>(std::llround(selected_candidate_id)) ==
                     selected_by_weight);
    const double blend_rmse = get_scalar(result, "blend_oof_rmse");
    N4M_TEST_REQUIRE(std::isfinite(blend_rmse));
    N4M_TEST_REQUIRE(get_scalar(result, "n_chains") == static_cast<double>(n_chains));
    N4M_TEST_REQUIRE(get_scalar(result, "n_candidates") ==
                     static_cast<double>(n_candidates));

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_ridge_blender_rejects_non_positive_lambda() {
    constexpr std::int64_t n = 12;
    constexpr std::int64_t p = 10;
    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    const double lambdas[1] = {0.0};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X.data(), n, p);
    n4m_matrix_view_t Yv = view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_ensemble_aom_ridge_blender_fit(
                         ctx, cfg, &Xv, &Yv, 0, 3,
                         nullptr, 0, lambdas, 1, 0.0, &result) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_ridge_blender_wide_includes_fck_moment_bank() {
    constexpr std::int64_t n = 18;
    constexpr std::int64_t p = 16;
    constexpr std::int32_t cv = 3;
    constexpr std::int64_t n_chains = 31;
    constexpr std::int64_t n_lambdas = 1;
    constexpr std::int64_t n_candidates = n_chains * n_lambdas;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = view(X.data(), n, p);
    n4m_matrix_view_t Yv = view(y.data(), n, 1);

    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_ensemble_aom_ridge_blender_fit(
                         ctx, cfg, &Xv, &Yv,
                         /*profile=*/1,
                         cv,
                         folds,
                         n,
                         lambdas,
                         n_lambdas,
                         /*regularizer=*/0.01,
                         &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    N4M_TEST_REQUIRE(get_scalar(result, "n_chains") ==
                     static_cast<double>(n_chains));
    N4M_TEST_REQUIRE(get_scalar(result, "n_candidates") ==
                     static_cast<double>(n_candidates));
    const auto weights = get_matrix(result, "weights", 1, n_candidates);
    double weight_sum = 0.0;
    for (double weight : weights) {
        N4M_TEST_REQUIRE(std::isfinite(weight));
        N4M_TEST_REQUIRE(weight >= -1e-12);
        weight_sum += weight;
    }
    N4M_TEST_REQUIRE(std::fabs(weight_sum - 1.0) <= 1e-10);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_aom_ridge_blender_tests(n4m_testing::Runner& r) {
    r.run("aom_ridge_blender/compact_contract_and_weighted_predictions",
          test_aom_ridge_blender_compact_contract_and_weighted_predictions);
    r.run("aom_ridge_blender/rejects_non_positive_lambda",
          test_aom_ridge_blender_rejects_non_positive_lambda);
    r.run("aom_ridge_blender/wide_includes_fck_moment_bank",
          test_aom_ridge_blender_wide_includes_fck_moment_bank);
}
