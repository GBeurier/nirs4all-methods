// SPDX-License-Identifier: CECILL-2.1

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"

namespace {

n4m_matrix_view_t make_view(double* data, std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t v{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &v, data, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return v;
}

void add_fold(n4m_validation_plan_t* plan,
              const std::vector<std::int64_t>& train,
              const std::vector<std::int64_t>& test) {
    N4M_TEST_REQUIRE(n4m_validation_plan_add_fold(
        plan,
        train.data(), static_cast<std::int64_t>(train.size()),
        test.data(), static_cast<std::int64_t>(test.size())) == N4M_OK);
}

void add_three_contiguous_folds(n4m_validation_plan_t* plan, std::int64_t n) {
    N4M_TEST_REQUIRE(n4m_validation_plan_set_n_samples(plan, n) == N4M_OK);
    for (std::int64_t fold = 0; fold < 3; ++fold) {
        std::vector<std::int64_t> train;
        std::vector<std::int64_t> test;
        for (std::int64_t row = 0; row < n; ++row) {
            if ((row * 3) / n == fold) {
                test.push_back(row);
            } else {
                train.push_back(row);
            }
        }
        add_fold(plan, train, test);
    }
}

void setup_selector(n4m_context_t** ctx,
                    n4m_config_t** cfg,
                    n4m_operator_bank_t** bank,
                    n4m_validation_plan_t** plan,
                    std::int64_t n) {
    N4M_TEST_REQUIRE(n4m_context_create(ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_algorithm(
        *cfg, N4M_ALGO_PLS_REGRESSION) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_solver(*cfg, N4M_SOLVER_SIMPLS) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_deflation(
        *cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(*cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_operator_bank_create(bank) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_operator_bank_add(
        *bank, N4M_OP_IDENTITY, nullptr, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_validation_plan_create(plan) == N4M_OK);
    add_three_contiguous_folds(*plan, n);
}

void teardown_selector(n4m_context_t* ctx,
                       n4m_config_t* cfg,
                       n4m_operator_bank_t* bank,
                       n4m_validation_plan_t* plan) {
    n4m_validation_plan_destroy(plan);
    n4m_operator_bank_destroy(bank);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void check_linear_replay(const std::vector<double>& X,
                         std::int64_t n,
                         std::int64_t p,
                         const double* predictions,
                         const double* coefficients,
                         const double* input_coefficients,
                         const double* intercept) {
    for (std::int64_t feature = 0; feature < p; ++feature) {
        N4M_TEST_REQUIRE(std::fabs(coefficients[feature] -
                                   input_coefficients[feature]) < 1e-10);
    }
    for (std::int64_t row = 0; row < n; ++row) {
        double replay = intercept[0];
        for (std::int64_t feature = 0; feature < p; ++feature) {
            replay += X[static_cast<std::size_t>(row * p + feature)] *
                      input_coefficients[feature];
        }
        N4M_TEST_REQUIRE(std::fabs(replay - predictions[row]) < 1e-9);
    }
}

void test_global_selector_exports_replayable_coefficients() {
    constexpr std::int64_t n = 18;
    constexpr std::int64_t p = 6;
    std::vector<double> X(static_cast<std::size_t>(n * p), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t row = 0; row < n; ++row) {
        for (std::int64_t col = 0; col < p; ++col) {
            X[static_cast<std::size_t>(row * p + col)] =
                std::sin(0.17 * static_cast<double>(row + 1) *
                         static_cast<double>(col + 1)) +
                0.03 * static_cast<double>(row - col);
        }
        y[static_cast<std::size_t>(row)] =
            0.7 * X[static_cast<std::size_t>(row * p)] -
            0.4 * X[static_cast<std::size_t>(row * p + 2)] +
            0.2 * X[static_cast<std::size_t>(row * p + 5)];
    }

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    n4m_operator_bank_t* bank = nullptr;
    n4m_validation_plan_t* plan = nullptr;
    setup_selector(&ctx, &cfg, &bank, &plan, n);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_aom_global_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_global_select(
        ctx, cfg, bank, &Xv, &Yv, plan, 2, &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);

    const double* predictions = nullptr;
    const double* coefficients = nullptr;
    const double* input_coefficients = nullptr;
    const double* intercept = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_aom_global_result_get_predictions(
        result, &predictions, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == n && cols == 1);
    N4M_TEST_REQUIRE(n4m_aom_global_result_get_coefficients(
        result, &coefficients, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == p && cols == 1);
    N4M_TEST_REQUIRE(n4m_aom_global_result_get_input_coefficients(
        result, &input_coefficients, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == p && cols == 1);
    N4M_TEST_REQUIRE(n4m_aom_global_result_get_intercept(
        result, &intercept, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == 1 && cols == 1);

    check_linear_replay(X, n, p, predictions, coefficients,
                        input_coefficients, intercept);

    n4m_aom_global_result_destroy(result);
    teardown_selector(ctx, cfg, bank, plan);
}

void test_pop_selector_exports_replayable_coefficients() {
    constexpr std::int64_t n = 18;
    constexpr std::int64_t p = 6;
    std::vector<double> X(static_cast<std::size_t>(n * p), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t row = 0; row < n; ++row) {
        for (std::int64_t col = 0; col < p; ++col) {
            X[static_cast<std::size_t>(row * p + col)] =
                std::cos(0.11 * static_cast<double>(row + 2) *
                         static_cast<double>(col + 1)) +
                0.02 * static_cast<double>(row + col);
        }
        y[static_cast<std::size_t>(row)] =
            0.5 * X[static_cast<std::size_t>(row * p + 1)] -
            0.3 * X[static_cast<std::size_t>(row * p + 3)];
    }

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    n4m_operator_bank_t* bank = nullptr;
    n4m_validation_plan_t* plan = nullptr;
    setup_selector(&ctx, &cfg, &bank, &plan, n);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_aom_per_component_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_per_component_select(
        ctx, cfg, bank, &Xv, &Yv, plan, 2, &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);

    const double* predictions = nullptr;
    const double* coefficients = nullptr;
    const double* input_coefficients = nullptr;
    const double* intercept = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    N4M_TEST_REQUIRE(n4m_aom_per_component_result_get_predictions(
        result, &predictions, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == n && cols == 1);
    N4M_TEST_REQUIRE(n4m_aom_per_component_result_get_coefficients(
        result, &coefficients, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == p && cols == 1);
    N4M_TEST_REQUIRE(n4m_aom_per_component_result_get_input_coefficients(
        result, &input_coefficients, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == p && cols == 1);
    N4M_TEST_REQUIRE(n4m_aom_per_component_result_get_intercept(
        result, &intercept, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(rows == 1 && cols == 1);

    check_linear_replay(X, n, p, predictions, coefficients,
                        input_coefficients, intercept);

    n4m_aom_per_component_result_destroy(result);
    teardown_selector(ctx, cfg, bank, plan);
}

}  // namespace

void register_aom_selection_tests(n4m_testing::Runner& r) {
    r.run("aom_global_selector_exports_replayable_coefficients",
          test_global_selector_exports_replayable_coefficients);
    r.run("pop_selector_exports_replayable_coefficients",
          test_pop_selector_exports_replayable_coefficients);
}
