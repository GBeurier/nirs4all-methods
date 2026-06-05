// SPDX-License-Identifier: CECILL-2.1
//
// Public ABI tests for n4m_aom_sweep_run. The result shape is intentionally
// checked because this surface is meant to feed downstream preprocessing-rank
// analysis.

#include "n4m/n4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

void register_aom_sweep_tests(n4m_testing::Runner& r);

namespace {

constexpr double kTol = 1e-9;

n4m_matrix_view_t make_view(double* data, std::int64_t rows,
                            std::int64_t cols) {
    n4m_matrix_view_t v{};
    N4M_TEST_REQUIRE(
        n4m_matrix_view_init_rowmajor(
            &v, data, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return v;
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

std::vector<std::int32_t> get_int_vector(const n4m_method_result_t* result,
                                         const char* key) {
    const std::int32_t* data = nullptr;
    std::int32_t size = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         result, key, &data, &size) == N4M_OK);
    return std::vector<std::int32_t>(
        data, data + static_cast<std::size_t>(size));
}

void assert_route_partitions(const n4m_method_result_t* result) {
    N4M_TEST_REQUIRE(
        std::fabs(get_scalar(result, "n_ridge_operator_moment_candidates") +
                  get_scalar(result, "n_pls_operator_moment_candidates") -
                  get_scalar(result, "n_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(
        std::fabs(get_scalar(result, "n_ridge_materialized_candidates") +
                  get_scalar(result, "n_pls_materialized_candidates") -
                  get_scalar(result, "n_materialized_candidates")) <= kTol);
    const auto routes = get_int_vector(result, "candidate_routes");
    const auto n_candidates =
        static_cast<std::int64_t>(std::llround(get_scalar(result, "n_candidates")));
    N4M_TEST_REQUIRE(routes.size() == static_cast<std::size_t>(n_candidates));
    std::int64_t n_materialized = 0;
    std::int64_t n_dense = 0;
    std::int64_t n_banded = 0;
    std::int64_t n_structured = 0;
    for (const auto route : routes) {
        if (route == 0) {
            ++n_materialized;
        } else if (route == 1) {
            ++n_dense;
        } else if (route == 2) {
            ++n_banded;
        } else if (route == 3) {
            ++n_structured;
        } else {
            throw std::runtime_error("unexpected AOM candidate route code");
        }
    }
    N4M_TEST_REQUIRE(
        std::fabs(static_cast<double>(n_materialized) -
                  get_scalar(result, "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(
        std::fabs(static_cast<double>(n_dense) -
                  get_scalar(result, "n_dense_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(
        std::fabs(static_cast<double>(n_banded) -
                  get_scalar(result, "n_banded_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(
        std::fabs(static_cast<double>(n_structured) -
                  get_scalar(result, "n_structured_operator_moment_candidates")) <= kTol);
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
                std::sin(0.13 * ii * jj) +
                std::cos(0.07 * (ii + 1.0) * (jj + 2.0)) +
                0.01 * (ii - jj);
        }
        y[static_cast<std::size_t>(i)] =
            0.7 * X[static_cast<std::size_t>(i * p + 0)] -
            0.4 * X[static_cast<std::size_t>(i * p + 3)] +
            0.2 * X[static_cast<std::size_t>(i * p + 7)] +
            0.05 * std::sin(0.3 * static_cast<double>(i));
    }
}

std::vector<double> finite_difference_order1(const std::vector<double>& X,
                                             std::int64_t n,
                                             std::int64_t p) {
    std::vector<double> out(static_cast<std::size_t>(n * p), 0.0);
    for (std::int64_t row = 0; row < n; ++row) {
        for (std::int64_t col = 0; col < p; ++col) {
            double value = 0.0;
            if (col > 0) {
                value -= 0.5 * X[static_cast<std::size_t>(row * p + col - 1)];
            }
            if (col + 1 < p) {
                value += 0.5 * X[static_cast<std::size_t>(row * p + col + 1)];
            }
            out[static_cast<std::size_t>(row * p + col)] = value;
        }
    }
    return out;
}

std::vector<double> detrend_degree1(const std::vector<double>& X,
                                    std::int64_t n,
                                    std::int64_t p) {
    std::vector<double> basis(static_cast<std::size_t>(p * 2), 1.0);
    for (std::int64_t col = 0; col < p; ++col) {
        basis[static_cast<std::size_t>(col * 2 + 1)] =
            p > 1
                ? -1.0 + 2.0 * static_cast<double>(col) /
                             static_cast<double>(p - 1)
                : 0.0;
    }

    double g00 = 0.0;
    double g01 = 0.0;
    double g11 = 0.0;
    for (std::int64_t col = 0; col < p; ++col) {
        const double b0 = basis[static_cast<std::size_t>(col * 2)];
        const double b1 = basis[static_cast<std::size_t>(col * 2 + 1)];
        g00 += b0 * b0;
        g01 += b0 * b1;
        g11 += b1 * b1;
    }
    const double det = g00 * g11 - g01 * g01;
    N4M_TEST_REQUIRE(std::fabs(det) > 1e-14);

    std::vector<double> out(static_cast<std::size_t>(n * p), 0.0);
    for (std::int64_t row = 0; row < n; ++row) {
        double rhs0 = 0.0;
        double rhs1 = 0.0;
        for (std::int64_t col = 0; col < p; ++col) {
            const double value = X[static_cast<std::size_t>(row * p + col)];
            rhs0 += value * basis[static_cast<std::size_t>(col * 2)];
            rhs1 += value * basis[static_cast<std::size_t>(col * 2 + 1)];
        }
        const double c0 = (rhs0 * g11 - rhs1 * g01) / det;
        const double c1 = (g00 * rhs1 - g01 * rhs0) / det;
        for (std::int64_t col = 0; col < p; ++col) {
            const double trend =
                c0 * basis[static_cast<std::size_t>(col * 2)] +
                c1 * basis[static_cast<std::size_t>(col * 2 + 1)];
            out[static_cast<std::size_t>(row * p + col)] =
                X[static_cast<std::size_t>(row * p + col)] - trend;
        }
    }
    return out;
}

void test_aom_sweep_compact_contract_and_oof_score() {
    constexpr std::int64_t n = 20;
    constexpr std::int64_t p = 24;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 12;
    constexpr std::int64_t n_lambdas = 1;
    constexpr std::int64_t n_components = 2;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const double lambdas[1] = {0.1};
    const std::int32_t comps[2] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_sweep_run(
                         ctx, cfg, &Xv, &Yv, 0, cv, folds, n,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &result) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_chains") -
                               static_cast<double>(n_chains)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "profile")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "cv") -
                               static_cast<double>(cv)) <= kTol);

    const auto result_chain_offsets = get_int_vector(result, "chain_offsets");
    const auto result_op_kinds = get_int_vector(result, "op_kinds");
    const auto result_param_offsets = get_int_vector(result, "param_offsets");
    N4M_TEST_REQUIRE(result_chain_offsets.size() ==
                     static_cast<std::size_t>(n_chains + 1));
    N4M_TEST_REQUIRE(result_chain_offsets.front() == 0);
    N4M_TEST_REQUIRE(result_chain_offsets.back() ==
                     static_cast<std::int32_t>(result_op_kinds.size()));
    N4M_TEST_REQUIRE(result_param_offsets.size() ==
                     result_op_kinds.size() + 1);
    N4M_TEST_REQUIRE(result_param_offsets.front() == 0);
    N4M_TEST_REQUIRE(result_op_kinds.front() == N4M_OP_IDENTITY);
    const std::vector<double> chain_params =
        get_matrix(result, "chain_params", 1,
                   static_cast<std::int64_t>(result_param_offsets.back()));
    N4M_TEST_REQUIRE(chain_params.size() ==
                     static_cast<std::size_t>(result_param_offsets.back()));

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", n_candidates, 5);
    std::vector<int> chain_seen(static_cast<std::size_t>(n_chains), 0);
    double best = scores[4];
    std::int64_t best_id = 0;
    for (std::int64_t row = 0; row < n_candidates; ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        N4M_TEST_REQUIRE(std::fabs(scores[offset] -
                                   static_cast<double>(row)) <= kTol);
        N4M_TEST_REQUIRE(std::isfinite(scores[offset + 4]));
        const auto chain_id = static_cast<std::int64_t>(scores[offset + 1]);
        N4M_TEST_REQUIRE(chain_id >= 0);
        N4M_TEST_REQUIRE(chain_id < n_chains);
        chain_seen[static_cast<std::size_t>(chain_id)] = 1;
        if (scores[offset + 4] < best) {
            best = scores[offset + 4];
            best_id = row;
        }
    }
    N4M_TEST_REQUIRE(std::all_of(chain_seen.begin(), chain_seen.end(),
                                 [](int v) { return v == 1; }));

    const auto selected_id =
        static_cast<std::int64_t>(get_scalar(result, "selected_candidate_id"));
    N4M_TEST_REQUIRE(selected_id == best_id);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_cv_rmse") -
                               best) <= kTol);
    const auto selected_offset = static_cast<std::size_t>(selected_id * 5);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_chain_id") -
                               scores[selected_offset + 1]) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_head_id") -
                               scores[selected_offset + 2]) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_param") -
                               scores[selected_offset + 3]) <= kTol);

    const std::vector<double> oof = get_matrix(result, "oof_predictions", n, 1);
    double sse = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        const double d = oof[static_cast<std::size_t>(i)] -
                         y[static_cast<std::size_t>(i)];
        sse += d * d;
    }
    const double rmse = std::sqrt(sse / static_cast<double>(n));
    N4M_TEST_REQUIRE(std::fabs(rmse -
                               get_scalar(result, "selected_cv_rmse")) <= kTol);

    const std::int32_t* fold_data = nullptr;
    std::int32_t fold_size = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         result, "fold_ids", &fold_data, &fold_size) == N4M_OK);
    N4M_TEST_REQUIRE(fold_size == n);
    for (std::int64_t i = 0; i < n; ++i) {
        N4M_TEST_REQUIRE(fold_data[static_cast<std::size_t>(i)] ==
                         folds[static_cast<std::size_t>(i)]);
    }

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_custom_descriptor_contract() {
    constexpr std::int64_t n = 20;
    constexpr std::int64_t p = 24;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 3;
    constexpr std::int64_t n_candidates = 9;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[4] = {0, 1, 2, 4};
    const std::int32_t op_kinds[4] = {
        N4M_OP_IDENTITY,
        N4M_OP_DETREND_POLY,
        N4M_OP_SAVGOL_SMOOTH,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[5] = {0, 0, 1, 3, 4};
    const double params[4] = {1.0, 5.0, 2.0, 1.0};
    const double lambdas[2] = {0.01, 0.1};
    const std::int32_t comps[1] = {1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         lambdas, 2, comps, 1, 3, &result) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "profile") + 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_chains") -
                               static_cast<double>(n_chains)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", n_candidates, 5);
    double best = scores[4];
    std::int64_t best_id = 0;
    std::vector<int> chain_seen(static_cast<std::size_t>(n_chains), 0);
    for (std::int64_t row = 0; row < n_candidates; ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        N4M_TEST_REQUIRE(std::fabs(scores[offset] -
                                   static_cast<double>(row)) <= kTol);
        N4M_TEST_REQUIRE(std::isfinite(scores[offset + 4]));
        const auto chain_id = static_cast<std::int64_t>(scores[offset + 1]);
        N4M_TEST_REQUIRE(chain_id >= 0);
        N4M_TEST_REQUIRE(chain_id < n_chains);
        chain_seen[static_cast<std::size_t>(chain_id)] = 1;
        if (scores[offset + 4] < best) {
            best = scores[offset + 4];
            best_id = row;
        }
    }
    N4M_TEST_REQUIRE(std::all_of(chain_seen.begin(), chain_seen.end(),
                                 [](int v) { return v == 1; }));
    const auto selected_id =
        static_cast<std::int64_t>(get_scalar(result, "selected_candidate_id"));
    N4M_TEST_REQUIRE(selected_id == best_id);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_cv_rmse") -
                               best) <= kTol);

    const std::vector<double> oof = get_matrix(result, "oof_predictions", n, 1);
    double sse = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        const double d = oof[static_cast<std::size_t>(i)] -
                         y[static_cast<std::size_t>(i)];
        sse += d * d;
    }
    const double rmse = std::sqrt(sse / static_cast<double>(n));
    N4M_TEST_REQUIRE(std::fabs(rmse -
                               get_scalar(result, "selected_cv_rmse")) <= kTol);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_ridge_operator_moments_match_materialized() {
    constexpr std::int64_t n = 30;
    constexpr std::int64_t p = 12;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 3;
    constexpr std::int64_t n_lambdas = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[4] = {0, 1, 2, 4};
    const std::int32_t op_kinds[4] = {
        N4M_OP_IDENTITY,
        N4M_OP_DETREND_POLY,
        N4M_OP_SAVGOL_SMOOTH,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[5] = {0, 0, 1, 3, 4};
    const double params[4] = {1.0, 5.0, 2.0, 1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};
    const std::int32_t comps[1] = {1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* ridge_only = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &ridge_only) == N4M_OK);

    n4m_method_result_t* materialized = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         lambdas, n_lambdas, comps, 1, 3,
                         &materialized) == N4M_OK);

    const std::vector<double> ridge_scores =
        get_matrix(ridge_only, "candidate_scores", n_chains * n_lambdas, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized, "candidate_scores", n_chains * (n_lambdas + 1), 5);

    for (std::int64_t chain = 0; chain < n_chains; ++chain) {
        for (std::int64_t lambda_ix = 0; lambda_ix < n_lambdas; ++lambda_ix) {
            const auto fast_offset =
                static_cast<std::size_t>((chain * n_lambdas + lambda_ix) * 5);
            const auto mat_offset =
                static_cast<std::size_t>((chain * (n_lambdas + 1) + lambda_ix) * 5);
            N4M_TEST_REQUIRE(std::fabs(ridge_scores[fast_offset + 1] -
                                       materialized_scores[mat_offset + 1]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(ridge_scores[fast_offset + 2]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(materialized_scores[mat_offset + 2]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(ridge_scores[fast_offset + 3] -
                                       materialized_scores[mat_offset + 3]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(ridge_scores[fast_offset + 4] -
                                       materialized_scores[mat_offset + 4]) <= 1e-8);
        }
    }

    n4m_method_result_destroy(materialized);
    n4m_method_result_destroy(ridge_only);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_wide_positive_ridge_operator_moments_match_materialized() {
    constexpr std::int64_t n = 50;
    constexpr std::int64_t p = 32;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 3;
    constexpr std::int64_t n_lambdas_fast = 2;
    constexpr std::int64_t n_lambdas_fallback = 3;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[4] = {0, 1, 2, 4};
    const std::int32_t op_kinds[4] = {
        N4M_OP_IDENTITY,
        N4M_OP_DETREND_POLY,
        N4M_OP_SAVGOL_SMOOTH,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[5] = {0, 0, 1, 3, 4};
    const double params[4] = {1.0, 5.0, 2.0, 1.0};
    const double positive_lambdas[static_cast<std::size_t>(n_lambdas_fast)] =
        {0.01, 0.1};
    const double fallback_lambdas[
        static_cast<std::size_t>(n_lambdas_fallback)] = {0.0, 0.01, 0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* wide_moment = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         positive_lambdas, n_lambdas_fast,
                         nullptr, 0, 1, &wide_moment) == N4M_OK);

    n4m_method_result_t* materialized = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         fallback_lambdas, n_lambdas_fallback,
                         nullptr, 0, 1, &materialized) == N4M_OK);

    const std::vector<double> moment_scores =
        get_matrix(wide_moment, "candidate_scores",
                   n_chains * n_lambdas_fast, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized, "candidate_scores",
                   n_chains * n_lambdas_fallback, 5);

    for (std::int64_t chain = 0; chain < n_chains; ++chain) {
        for (std::int64_t lambda_ix = 0; lambda_ix < n_lambdas_fast; ++lambda_ix) {
            const auto fast_offset = static_cast<std::size_t>(
                (chain * n_lambdas_fast + lambda_ix) * 5);
            const auto mat_offset = static_cast<std::size_t>(
                (chain * n_lambdas_fallback + lambda_ix + 1) * 5);
            N4M_TEST_REQUIRE(std::fabs(moment_scores[fast_offset + 1] -
                                       materialized_scores[mat_offset + 1]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(moment_scores[fast_offset + 2]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(materialized_scores[mat_offset + 2]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(moment_scores[fast_offset + 3] -
                                       materialized_scores[mat_offset + 3]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(moment_scores[fast_offset + 4] -
                                       materialized_scores[mat_offset + 4]) <= 1e-7);
        }
    }

    n4m_method_result_destroy(materialized);
    n4m_method_result_destroy(wide_moment);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_banded_ridge_matches_materialized_sweep() {
    constexpr std::int64_t n = 100;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_lambdas = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::vector<double> Xt = finite_difference_order1(X, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_FINITE_DIFFERENCE};
    const std::int32_t param_offsets[2] = {0, 1};
    const double params[1] = {1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Xtv = make_view(Xt.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* banded = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &banded) == N4M_OK);

    n4m_method_result_t* materialized = nullptr;
    N4M_TEST_REQUIRE(n4m_sweep_run(
                         ctx, cfg, &Xtv, &Yv, cv, folds, n,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &materialized) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_operator_moment_candidates") -
                               static_cast<double>(n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_banded_operator_moment_candidates") -
                               static_cast<double>(n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_dense_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_ridge_operator_moment_candidates") -
                               static_cast<double>(n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_pls_operator_moment_candidates")) <= kTol);
    assert_route_partitions(banded);

    const std::vector<double> banded_scores =
        get_matrix(banded, "candidate_scores", n_lambdas, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized, "candidate_scores", n_lambdas, 4);
    for (std::int64_t row = 0; row < n_lambdas; ++row) {
        const auto fast = static_cast<std::size_t>(row * 5);
        const auto ref = static_cast<std::size_t>(row * 4);
        N4M_TEST_REQUIRE(std::fabs(banded_scores[fast + 2] -
                                   materialized_scores[ref + 1]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(banded_scores[fast + 3] -
                                   materialized_scores[ref + 2]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(banded_scores[fast + 4] -
                                   materialized_scores[ref + 3]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized);
    n4m_method_result_destroy(banded);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_banded_pls_matches_materialized_sweep() {
    constexpr std::int64_t n = 320;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_components = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::vector<double> Xt = finite_difference_order1(X, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_FINITE_DIFFERENCE};
    const std::int32_t param_offsets[2] = {0, 1};
    const double params[1] = {1.0};
    const std::int32_t comps[static_cast<std::size_t>(n_components)] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Xtv = make_view(Xt.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* banded = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         nullptr, 0, comps, n_components, 2,
                         &banded) == N4M_OK);

    n4m_method_result_t* materialized = nullptr;
    N4M_TEST_REQUIRE(n4m_sweep_run(
                         ctx, cfg, &Xtv, &Yv, cv, folds, n,
                         nullptr, 0, comps, n_components, 2,
                         &materialized) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_operator_moment_candidates") -
                               static_cast<double>(n_components)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_banded_operator_moment_candidates") -
                               static_cast<double>(n_components)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_dense_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_pls_operator_moment_candidates") -
                               static_cast<double>(n_components)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(banded, "n_ridge_operator_moment_candidates")) <= kTol);
    assert_route_partitions(banded);

    const std::vector<double> banded_scores =
        get_matrix(banded, "candidate_scores", n_components, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized, "candidate_scores", n_components, 4);
    for (std::int64_t row = 0; row < n_components; ++row) {
        const auto fast = static_cast<std::size_t>(row * 5);
        const auto ref = static_cast<std::size_t>(row * 4);
        N4M_TEST_REQUIRE(std::fabs(banded_scores[fast + 2] -
                                   materialized_scores[ref + 1]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(banded_scores[fast + 3] -
                                   materialized_scores[ref + 2]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(banded_scores[fast + 4] -
                                   materialized_scores[ref + 3]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized);
    n4m_method_result_destroy(banded);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_gaussian_banded_moment_route() {
    constexpr std::int64_t n = 192;
    constexpr std::int64_t p = 32;
    constexpr std::int32_t cv = 4;
    constexpr std::int64_t n_lambdas = 1;
    constexpr std::int64_t n_components = 1;
    constexpr std::int64_t n_candidates = n_lambdas + n_components;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }

    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_GAUSSIAN};
    const std::int32_t param_offsets[2] = {0, 1};
    const double params[1] = {1.0};
    const double lambdas[1] = {0.1};
    const std::int32_t comps[1] = {1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         cfg, N4M_AOM_MOMENT_FORCE_MOMENTS) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &result) == N4M_OK);

    (void)get_matrix(result, "candidate_scores", n_candidates, 5);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_banded_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_materialized_candidates")) <= kTol);
    assert_route_partitions(result);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_structured_detrend_ridge_matches_materialized_sweep() {
    constexpr std::int64_t n = 100;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_lambdas = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::vector<double> Xt = finite_difference_order1(
        detrend_degree1(X, n, p), n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 2};
    const std::int32_t op_kinds[2] = {
        N4M_OP_DETREND_POLY,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[3] = {0, 1, 2};
    const double params[2] = {1.0, 1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Xtv = make_view(Xt.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* structured = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 2,
                         param_offsets, 3, params, 2,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &structured) == N4M_OK);

    n4m_method_result_t* materialized = nullptr;
    N4M_TEST_REQUIRE(n4m_sweep_run(
                         ctx, cfg, &Xtv, &Yv, cv, folds, n,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &materialized) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_operator_moment_candidates") -
                               static_cast<double>(n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_structured_operator_moment_candidates") -
                               static_cast<double>(n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_banded_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_dense_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_materialized_candidates")) <= kTol);

    const std::vector<double> structured_scores =
        get_matrix(structured, "candidate_scores", n_lambdas, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized, "candidate_scores", n_lambdas, 4);
    for (std::int64_t row = 0; row < n_lambdas; ++row) {
        const auto fast = static_cast<std::size_t>(row * 5);
        const auto ref = static_cast<std::size_t>(row * 4);
        N4M_TEST_REQUIRE(std::fabs(structured_scores[fast + 2] -
                                   materialized_scores[ref + 1]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(structured_scores[fast + 3] -
                                   materialized_scores[ref + 2]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(structured_scores[fast + 4] -
                                   materialized_scores[ref + 3]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized);
    n4m_method_result_destroy(structured);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_structured_detrend_pls_matches_materialized_sweep() {
    constexpr std::int64_t n = 320;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_components = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::vector<double> Xt = finite_difference_order1(
        detrend_degree1(X, n, p), n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 2};
    const std::int32_t op_kinds[2] = {
        N4M_OP_DETREND_POLY,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[3] = {0, 1, 2};
    const double params[2] = {1.0, 1.0};
    const std::int32_t comps[static_cast<std::size_t>(n_components)] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Xtv = make_view(Xt.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* structured = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 2,
                         param_offsets, 3, params, 2,
                         nullptr, 0, comps, n_components, 2,
                         &structured) == N4M_OK);

    n4m_method_result_t* materialized = nullptr;
    N4M_TEST_REQUIRE(n4m_sweep_run(
                         ctx, cfg, &Xtv, &Yv, cv, folds, n,
                         nullptr, 0, comps, n_components, 2,
                         &materialized) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_operator_moment_candidates") -
                               static_cast<double>(n_components)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_structured_operator_moment_candidates") -
                               static_cast<double>(n_components)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_banded_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_dense_operator_moment_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(structured, "n_materialized_candidates")) <= kTol);

    const std::vector<double> structured_scores =
        get_matrix(structured, "candidate_scores", n_components, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized, "candidate_scores", n_components, 4);
    for (std::int64_t row = 0; row < n_components; ++row) {
        const auto fast = static_cast<std::size_t>(row * 5);
        const auto ref = static_cast<std::size_t>(row * 4);
        N4M_TEST_REQUIRE(std::fabs(structured_scores[fast + 2] -
                                   materialized_scores[ref + 1]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(structured_scores[fast + 3] -
                                   materialized_scores[ref + 2]) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(structured_scores[fast + 4] -
                                   materialized_scores[ref + 3]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized);
    n4m_method_result_destroy(structured);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_mixed_hybrid_matches_split_runs() {
    constexpr std::int64_t n = 30;
    constexpr std::int64_t p = 12;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 3;
    constexpr std::int64_t n_lambdas = 2;
    constexpr std::int64_t n_components = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[4] = {0, 1, 2, 4};
    const std::int32_t op_kinds[4] = {
        N4M_OP_IDENTITY,
        N4M_OP_DETREND_POLY,
        N4M_OP_SAVGOL_SMOOTH,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[5] = {0, 0, 1, 3, 4};
    const double params[4] = {1.0, 5.0, 2.0, 1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};
    const std::int32_t comps[static_cast<std::size_t>(n_components)] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* mixed = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &mixed) == N4M_OK);

    n4m_method_result_t* ridge_only = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &ridge_only) == N4M_OK);

    n4m_method_result_t* pls_only = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 4, op_kinds, 4,
                         param_offsets, 5, params, 4,
                         nullptr, 0, comps, n_components, 2,
                         &pls_only) == N4M_OK);

    const std::vector<double> mixed_scores =
        get_matrix(mixed, "candidate_scores",
                   n_chains * (n_lambdas + n_components), 5);
    const std::vector<double> ridge_scores =
        get_matrix(ridge_only, "candidate_scores", n_chains * n_lambdas, 5);
    const std::vector<double> pls_scores =
        get_matrix(pls_only, "candidate_scores", n_chains * n_components, 5);

    for (std::int64_t chain = 0; chain < n_chains; ++chain) {
        for (std::int64_t lambda_ix = 0; lambda_ix < n_lambdas; ++lambda_ix) {
            const auto mixed_offset = static_cast<std::size_t>(
                (chain * (n_lambdas + n_components) + lambda_ix) * 5);
            const auto ridge_offset =
                static_cast<std::size_t>((chain * n_lambdas + lambda_ix) * 5);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 1] -
                                       ridge_scores[ridge_offset + 1]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 2] -
                                       ridge_scores[ridge_offset + 2]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 3] -
                                       ridge_scores[ridge_offset + 3]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 4] -
                                       ridge_scores[ridge_offset + 4]) <= 1e-8);
        }
        for (std::int64_t comp_ix = 0; comp_ix < n_components; ++comp_ix) {
            const auto mixed_offset = static_cast<std::size_t>(
                (chain * (n_lambdas + n_components) + n_lambdas + comp_ix) * 5);
            const auto pls_offset =
                static_cast<std::size_t>((chain * n_components + comp_ix) * 5);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 1] -
                                       pls_scores[pls_offset + 1]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 2] -
                                       pls_scores[pls_offset + 2]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 3] -
                                       pls_scores[pls_offset + 3]) <= kTol);
            N4M_TEST_REQUIRE(std::fabs(mixed_scores[mixed_offset + 4] -
                                       pls_scores[pls_offset + 4]) <= 1e-8);
        }
    }

    double best = mixed_scores[4];
    std::int64_t best_id = 0;
    for (std::int64_t row = 0; row < n_chains * (n_lambdas + n_components); ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        if (mixed_scores[offset + 4] < best) {
            best = mixed_scores[offset + 4];
            best_id = row;
        }
    }
    N4M_TEST_REQUIRE(
        static_cast<std::int64_t>(get_scalar(mixed, "selected_candidate_id")) ==
        best_id);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(mixed, "selected_cv_rmse") -
                               best) <= kTol);

    n4m_method_result_destroy(pls_only);
    n4m_method_result_destroy(ridge_only);
    n4m_method_result_destroy(mixed);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_materialized_policy_matches_auto_scores() {
    constexpr std::int64_t n = 100;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 2;
    constexpr std::int64_t n_lambdas = 2;
    constexpr std::int64_t n_components = 2;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[3] = {0, 1, 3};
    const std::int32_t op_kinds[3] = {
        N4M_OP_IDENTITY,
        N4M_OP_DETREND_POLY,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[4] = {0, 0, 1, 2};
    const double params[2] = {1.0, 1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};
    const std::int32_t comps[static_cast<std::size_t>(n_components)] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* auto_cfg = nullptr;
    n4m_config_t* materialized_cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&auto_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&materialized_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(auto_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(materialized_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         materialized_cfg,
                         N4M_AOM_MOMENT_MATERIALIZED) == N4M_OK);
    n4m_aom_moment_policy_t got_policy = N4M_AOM_MOMENT_AUTO;
    N4M_TEST_REQUIRE(n4m_config_get_aom_moment_policy(
                         materialized_cfg, &got_policy) == N4M_OK);
    N4M_TEST_REQUIRE(got_policy == N4M_AOM_MOMENT_MATERIALIZED);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* auto_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, auto_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 3,
                         param_offsets, 4, params, 2,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &auto_result) == N4M_OK);

    n4m_method_result_t* materialized_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, materialized_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 3,
                         param_offsets, 4, params, 2,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &materialized_result) == N4M_OK);

    N4M_TEST_REQUIRE(get_scalar(auto_result,
                                "n_operator_moment_candidates") > 0.0);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(materialized_result,
                                          "n_operator_moment_candidates")) <=
                     kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(materialized_result,
                                          "n_materialized_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(materialized_result,
                                          "n_ridge_materialized_candidates") -
                               static_cast<double>(n_chains * n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(materialized_result,
                                          "n_pls_materialized_candidates") -
                               static_cast<double>(n_chains * n_components)) <= kTol);
    assert_route_partitions(auto_result);
    assert_route_partitions(materialized_result);

    const std::vector<double> auto_scores =
        get_matrix(auto_result, "candidate_scores", n_candidates, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized_result, "candidate_scores", n_candidates, 5);
    for (std::int64_t row = 0; row < n_candidates; ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        for (std::size_t col = 0; col < 4U; ++col) {
            N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + col] -
                                       materialized_scores[offset + col]) <=
                             kTol);
        }
        N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + 4] -
                                   materialized_scores[offset + 4]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized_result);
    n4m_method_result_destroy(auto_result);
    n4m_config_destroy(materialized_cfg);
    n4m_config_destroy(auto_cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_force_moments_policy_accepts_full_moment_grid() {
    constexpr std::int64_t n = 330;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 2;
    constexpr std::int64_t n_lambdas = 1;
    constexpr std::int64_t n_components = 1;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[3] = {0, 1, 2};
    const std::int32_t op_kinds[2] = {
        N4M_OP_IDENTITY,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[3] = {0, 0, 1};
    const double params[1] = {1.0};
    const double lambdas[1] = {0.1};
    const std::int32_t comps[1] = {1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         cfg, N4M_AOM_MOMENT_FORCE_MOMENTS) == N4M_OK);
    n4m_aom_moment_policy_t got_policy = N4M_AOM_MOMENT_AUTO;
    N4M_TEST_REQUIRE(n4m_config_get_aom_moment_policy(
                         cfg, &got_policy) == N4M_OK);
    N4M_TEST_REQUIRE(got_policy == N4M_AOM_MOMENT_FORCE_MOMENTS);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 2,
                         param_offsets, 3, params, 1,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &result) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_ridge_operator_moment_candidates") -
                               static_cast<double>(n_chains * n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_pls_operator_moment_candidates") -
                               static_cast<double>(n_chains * n_components)) <= kTol);
    assert_route_partitions(result);
    (void)get_matrix(result, "input_coefficients", p, 1);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_force_moments_score_only_skips_final_refit() {
    constexpr std::int64_t n = 330;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 2;
    constexpr std::int64_t n_lambdas = 1;
    constexpr std::int64_t n_components = 1;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[3] = {0, 1, 2};
    const std::int32_t op_kinds[2] = {
        N4M_OP_IDENTITY,
        N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[3] = {0, 0, 1};
    const double params[1] = {1.0};
    const double lambdas[1] = {0.1};
    const std::int32_t comps[1] = {1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         cfg, N4M_AOM_MOMENT_FORCE_MOMENTS) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);
    std::int32_t got_score_only = 0;
    N4M_TEST_REQUIRE(n4m_config_get_aom_score_only(
                         cfg, &got_score_only) == N4M_OK);
    N4M_TEST_REQUIRE(got_score_only == 1);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 2,
                         param_offsets, 3, params, 1,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &result) == N4M_OK);

    (void)get_matrix(result, "candidate_scores", n_candidates, 5);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "score_only") - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_ridge_operator_moment_candidates") -
                               static_cast<double>(n_chains * n_lambdas)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result,
                                          "n_pls_operator_moment_candidates") -
                               static_cast<double>(n_chains * n_components)) <= kTol);
    assert_route_partitions(result);
    (void)get_matrix(result, "predictions", 0, 0);
    (void)get_matrix(result, "oof_predictions", 0, 0);
    (void)get_matrix(result, "coefficients", 0, 0);
    (void)get_matrix(result, "input_coefficients", 0, 0);

    const std::int32_t* fold_data = nullptr;
    std::int32_t fold_size = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         result, "fold_ids", &fold_data, &fold_size) == N4M_OK);
    N4M_TEST_REQUIRE(fold_size == n);
    for (std::int64_t i = 0; i < n; ++i) {
        N4M_TEST_REQUIRE(fold_data[static_cast<std::size_t>(i)] ==
                         folds[static_cast<std::size_t>(i)]);
    }

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_materialized_score_only_keeps_scores() {
    constexpr std::int64_t n = 36;
    constexpr std::int64_t p = 18;
    constexpr std::int32_t cv = 4;
    constexpr std::int64_t n_chains = 2;
    constexpr std::int64_t n_lambdas = 2;
    constexpr std::int64_t n_components = 2;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }

    const std::int32_t chain_offsets[3] = {0, 1, 2};
    const std::int32_t op_kinds[2] = {
        N4M_OP_IDENTITY,
        N4M_OP_SAVGOL_SMOOTH,
    };
    const std::int32_t param_offsets[3] = {0, 0, 2};
    const double params[2] = {5.0, 2.0};
    const double lambdas[2] = {0.1, 1.0};
    const std::int32_t comps[2] = {1, 2};
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* full_cfg = nullptr;
    n4m_config_t* score_cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&full_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&score_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(full_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(score_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         full_cfg, N4M_AOM_MOMENT_MATERIALIZED) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         score_cfg, N4M_AOM_MOMENT_MATERIALIZED) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(score_cfg, 1) == N4M_OK);

    n4m_method_result_t* full = nullptr;
    n4m_method_result_t* score_only = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, full_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 2,
                         param_offsets, 3, params, 2,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &full) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, score_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 2,
                         param_offsets, 3, params, 2,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &score_only) == N4M_OK);

    const std::vector<double> full_scores =
        get_matrix(full, "candidate_scores", n_candidates, 5);
    const std::vector<double> score_scores =
        get_matrix(score_only, "candidate_scores", n_candidates, 5);
    for (std::size_t i = 0; i < full_scores.size(); ++i) {
        N4M_TEST_REQUIRE(std::fabs(full_scores[i] - score_scores[i]) <= 1e-8);
    }
    N4M_TEST_REQUIRE(std::fabs(get_scalar(score_only, "score_only") - 1.0) <= kTol);
    (void)get_matrix(score_only, "predictions", 0, 0);
    (void)get_matrix(score_only, "oof_predictions", 0, 0);
    (void)get_matrix(score_only, "coefficients", 0, 0);
    (void)get_matrix(score_only, "input_coefficients", 0, 0);

    n4m_method_result_destroy(score_only);
    n4m_method_result_destroy(full);
    n4m_config_destroy(score_cfg);
    n4m_config_destroy(full_cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_force_moments_policy_rejects_fallback() {
    constexpr std::int64_t n = 40;
    constexpr std::int64_t p = 16;
    constexpr std::int32_t cv = 4;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::vector<double> Y(static_cast<std::size_t>(n * 2), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        Y[static_cast<std::size_t>(i * 2)] = y[static_cast<std::size_t>(i)];
        Y[static_cast<std::size_t>(i * 2 + 1)] =
            0.5 * y[static_cast<std::size_t>(i)] +
            0.1 * X[static_cast<std::size_t>(i * p)];
    }

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_IDENTITY};
    const std::int32_t param_offsets[2] = {0, 0};
    const std::int32_t comps[1] = {1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         cfg, N4M_AOM_MOMENT_FORCE_MOMENTS) == N4M_OK);

    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(Y.data(), n, 2);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, nullptr, 0,
                         nullptr, 0, comps, 1, 2,
                         &result) == N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(result == nullptr);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_moment_prefix_cache_counters() {
    constexpr std::int64_t n = 140;
    constexpr std::int64_t p = 24;
    constexpr std::int32_t cv = 4;
    constexpr std::int64_t n_chains = 5;
    constexpr std::int64_t n_lambdas = 1;
    constexpr std::int64_t n_components = 1;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);
    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }

    const std::int32_t chain_offsets[6] = {0, 1, 3, 5, 6, 8};
    const std::int32_t op_kinds[8] = {
        N4M_OP_DETREND_POLY,
        N4M_OP_DETREND_POLY, N4M_OP_SAVGOL_DERIVATIVE,
        N4M_OP_DETREND_POLY, N4M_OP_FINITE_DIFFERENCE,
        N4M_OP_SAVGOL_SMOOTH,
        N4M_OP_SAVGOL_SMOOTH, N4M_OP_FINITE_DIFFERENCE,
    };
    const std::int32_t param_offsets[9] = {
        0, 1, 2, 5, 6, 7, 9, 11, 12,
    };
    const double params[12] = {
        1.0,
        1.0, 7.0, 2.0, 1.0,
        1.0, 1.0,
        5.0, 2.0,
        5.0, 2.0, 1.0,
    };
    const double lambdas[1] = {0.1};
    const std::int32_t comps[1] = {1};
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         cfg, N4M_AOM_MOMENT_FORCE_MOMENTS) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);

    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 6, op_kinds, 8,
                         param_offsets, 9, params, 12,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &result) == N4M_OK);

    (void)get_matrix(result, "candidate_scores", n_candidates, 5);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(get_scalar(result, "n_moment_prefix_cache_hits") >= 3.0);
    N4M_TEST_REQUIRE(get_scalar(result, "n_moment_prefix_cache_misses") >= 5.0);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_wide_ridge_auto_route_is_backend_aware() {
    constexpr std::int64_t n = 30;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_lambdas = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_FINITE_DIFFERENCE};
    const std::int32_t param_offsets[2] = {0, 1};
    const double params[1] = {1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* auto_cfg = nullptr;
    n4m_config_t* materialized_cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&auto_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&materialized_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(auto_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(materialized_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         materialized_cfg,
                         N4M_AOM_MOMENT_MATERIALIZED) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* auto_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, auto_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &auto_result) == N4M_OK);

    n4m_method_result_t* materialized_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, materialized_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         lambdas, n_lambdas, nullptr, 0, 1,
                         &materialized_result) == N4M_OK);

    const bool cuda_available =
        n4m_backend_is_available(N4M_BACKEND_CUDA) != 0;
    if (cuda_available) {
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_operator_moment_candidates") -
                                   static_cast<double>(n_lambdas)) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_materialized_candidates")) <= kTol);
    } else {
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_operator_moment_candidates")) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_materialized_candidates") -
                                   static_cast<double>(n_lambdas)) <= kTol);
    }

    const std::vector<double> auto_scores =
        get_matrix(auto_result, "candidate_scores", n_lambdas, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized_result, "candidate_scores", n_lambdas, 5);
    for (std::int64_t row = 0; row < n_lambdas; ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        for (std::size_t col = 0; col < 4U; ++col) {
            N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + col] -
                                       materialized_scores[offset + col]) <=
                             kTol);
        }
        N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + 4] -
                                   materialized_scores[offset + 4]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized_result);
    n4m_method_result_destroy(auto_result);
    n4m_config_destroy(materialized_cfg);
    n4m_config_destroy(auto_cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_wide_pls_auto_route_is_backend_aware() {
    constexpr std::int64_t n = 30;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_components = 2;

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_FINITE_DIFFERENCE};
    const std::int32_t param_offsets[2] = {0, 1};
    const double params[1] = {1.0};
    const std::int32_t comps[static_cast<std::size_t>(n_components)] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* auto_cfg = nullptr;
    n4m_config_t* materialized_cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&auto_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&materialized_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(auto_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(materialized_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         materialized_cfg,
                         N4M_AOM_MOMENT_MATERIALIZED) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* auto_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, auto_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         nullptr, 0, comps, n_components, 2,
                         &auto_result) == N4M_OK);

    n4m_method_result_t* materialized_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, materialized_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, params, 1,
                         nullptr, 0, comps, n_components, 2,
                         &materialized_result) == N4M_OK);

    const bool cuda_available =
        n4m_backend_is_available(N4M_BACKEND_CUDA) != 0;
    if (cuda_available) {
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_operator_moment_candidates") -
                                   static_cast<double>(n_components)) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_materialized_candidates")) <= kTol);
    } else {
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_operator_moment_candidates")) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                              "n_materialized_candidates") -
                                   static_cast<double>(n_components)) <= kTol);
    }

    const std::vector<double> auto_scores =
        get_matrix(auto_result, "candidate_scores", n_components, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized_result, "candidate_scores", n_components, 5);
    for (std::int64_t row = 0; row < n_components; ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        for (std::size_t col = 0; col < 4U; ++col) {
            N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + col] -
                                       materialized_scores[offset + col]) <=
                             kTol);
        }
        N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + 4] -
                                   materialized_scores[offset + 4]) <= 1e-7);
    }

    n4m_method_result_destroy(materialized_result);
    n4m_method_result_destroy(auto_result);
    n4m_config_destroy(materialized_cfg);
    n4m_config_destroy(auto_cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_structured_whittaker_matches_materialized_policy() {
    constexpr std::int64_t n = 330;
    constexpr std::int64_t p = 64;
    constexpr std::int32_t cv = 5;
    constexpr std::int64_t n_chains = 2;
    constexpr std::int64_t n_lambdas = 2;
    constexpr std::int64_t n_components = 2;
    constexpr std::int64_t n_candidates =
        n_chains * (n_lambdas + n_components);

    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    std::int32_t folds[static_cast<std::size_t>(n)]{};
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t chain_offsets[3] = {0, 1, 3};
    const std::int32_t op_kinds[3] = {
        N4M_OP_WHITTAKER,
        N4M_OP_WHITTAKER,
        N4M_OP_SAVGOL_DERIVATIVE,
    };
    const std::int32_t param_offsets[4] = {0, 1, 2, 5};
    const double params[5] = {100.0, 1000.0, 7.0, 2.0, 1.0};
    const double lambdas[static_cast<std::size_t>(n_lambdas)] = {0.01, 0.1};
    const std::int32_t comps[static_cast<std::size_t>(n_components)] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* auto_cfg = nullptr;
    n4m_config_t* materialized_cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&auto_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&materialized_cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(auto_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(materialized_cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_moment_policy(
                         materialized_cfg,
                         N4M_AOM_MOMENT_MATERIALIZED) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);

    n4m_method_result_t* auto_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, auto_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 3,
                         param_offsets, 4, params, 5,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &auto_result) == N4M_OK);

    n4m_method_result_t* materialized_result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, materialized_cfg, &Xv, &Yv, cv, folds, n,
                         chain_offsets, 3, op_kinds, 3,
                         param_offsets, 4, params, 5,
                         lambdas, n_lambdas, comps, n_components, 3,
                         &materialized_result) == N4M_OK);

    N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                          "n_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                          "n_structured_operator_moment_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                          "n_banded_operator_moment_candidates")) <=
                     kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                          "n_dense_operator_moment_candidates")) <=
                     kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(auto_result,
                                          "n_materialized_candidates")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(materialized_result,
                                          "n_operator_moment_candidates")) <=
                     kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(materialized_result,
                                          "n_materialized_candidates") -
                               static_cast<double>(n_candidates)) <= kTol);

    const std::vector<double> auto_scores =
        get_matrix(auto_result, "candidate_scores", n_candidates, 5);
    const std::vector<double> materialized_scores =
        get_matrix(materialized_result, "candidate_scores", n_candidates, 5);
    for (std::int64_t row = 0; row < n_candidates; ++row) {
        const auto offset = static_cast<std::size_t>(row * 5);
        for (std::size_t col = 0; col < 4U; ++col) {
            N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + col] -
                                       materialized_scores[offset + col]) <=
                             kTol);
        }
        N4M_TEST_REQUIRE(std::fabs(auto_scores[offset + 4] -
                                   materialized_scores[offset + 4]) <= 1e-6);
    }

    n4m_method_result_destroy(materialized_result);
    n4m_method_result_destroy(auto_result);
    n4m_config_destroy(materialized_cfg);
    n4m_config_destroy(auto_cfg);
    n4m_context_destroy(ctx);
}

void test_aom_chain_sweep_rejects_non_strict_operator() {
    constexpr std::int64_t n = 20;
    constexpr std::int64_t p = 24;
    std::vector<double> X;
    std::vector<double> y;
    make_dataset(X, y, n, p);

    const std::int32_t chain_offsets[2] = {0, 1};
    const std::int32_t op_kinds[1] = {N4M_OP_SNV};
    const std::int32_t param_offsets[2] = {0, 0};
    const double lambda = 0.1;

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_aom_chain_sweep_run(
                         ctx, cfg, &Xv, &Yv, 5, nullptr, 0,
                         chain_offsets, 2, op_kinds, 1,
                         param_offsets, 2, nullptr, 0,
                         &lambda, 1, nullptr, 0, 1, &result) ==
                     N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(result == nullptr);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_aom_sweep_tests(n4m_testing::Runner& r) {
    r.run("aom_sweep/compact_contract_and_oof_score",
          test_aom_sweep_compact_contract_and_oof_score);
    r.run("aom_chain_sweep/custom_descriptor_contract",
          test_aom_chain_sweep_custom_descriptor_contract);
    r.run("aom_chain_sweep/ridge_operator_moments_match_materialized",
          test_aom_chain_sweep_ridge_operator_moments_match_materialized);
    r.run("aom_chain_sweep/wide_positive_ridge_operator_moments_match_materialized",
          test_aom_chain_sweep_wide_positive_ridge_operator_moments_match_materialized);
    r.run("aom_chain_sweep/banded_ridge_matches_materialized_sweep",
          test_aom_chain_sweep_banded_ridge_matches_materialized_sweep);
    r.run("aom_chain_sweep/banded_pls_matches_materialized_sweep",
          test_aom_chain_sweep_banded_pls_matches_materialized_sweep);
    r.run("aom_chain_sweep/gaussian_banded_moment_route",
          test_aom_chain_sweep_gaussian_banded_moment_route);
    r.run("aom_chain_sweep/structured_detrend_ridge_matches_materialized_sweep",
          test_aom_chain_sweep_structured_detrend_ridge_matches_materialized_sweep);
    r.run("aom_chain_sweep/structured_detrend_pls_matches_materialized_sweep",
          test_aom_chain_sweep_structured_detrend_pls_matches_materialized_sweep);
    r.run("aom_chain_sweep/mixed_hybrid_matches_split_runs",
          test_aom_chain_sweep_mixed_hybrid_matches_split_runs);
    r.run("aom_chain_sweep/materialized_policy_matches_auto_scores",
          test_aom_chain_sweep_materialized_policy_matches_auto_scores);
    r.run("aom_chain_sweep/force_moments_policy_accepts_full_moment_grid",
          test_aom_chain_sweep_force_moments_policy_accepts_full_moment_grid);
    r.run("aom_chain_sweep/force_moments_score_only_skips_final_refit",
          test_aom_chain_sweep_force_moments_score_only_skips_final_refit);
    r.run("aom_chain_sweep/materialized_score_only_keeps_scores",
          test_aom_chain_sweep_materialized_score_only_keeps_scores);
    r.run("aom_chain_sweep/force_moments_policy_rejects_fallback",
          test_aom_chain_sweep_force_moments_policy_rejects_fallback);
    r.run("aom_chain_sweep/moment_prefix_cache_counters",
          test_aom_chain_sweep_moment_prefix_cache_counters);
    r.run("aom_chain_sweep/wide_ridge_auto_route_is_backend_aware",
          test_aom_chain_sweep_wide_ridge_auto_route_is_backend_aware);
    r.run("aom_chain_sweep/wide_pls_auto_route_is_backend_aware",
          test_aom_chain_sweep_wide_pls_auto_route_is_backend_aware);
    r.run("aom_chain_sweep/structured_whittaker_matches_materialized_policy",
          test_aom_chain_sweep_structured_whittaker_matches_materialized_policy);
    r.run("aom_chain_sweep/rejects_non_strict_operator",
          test_aom_chain_sweep_rejects_non_strict_operator);
}
