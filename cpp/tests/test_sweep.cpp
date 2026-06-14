// SPDX-License-Identifier: CECILL-2.1
//
// Public ABI tests for n4m_model_selection_sweep_run. The key test compares the moment-based
// OOF predictions against a materialized fold-by-fold n4m_estimators_ridge_fit reference.

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

void register_sweep_tests(n4m_testing::Runner& r);

namespace {

constexpr double kTol = 1e-9;

const double kX[24] = {
    0.5, -1.0,  2.0,
    1.5,  0.2, -0.5,
   -0.3,  2.2,  1.1,
    2.0, -0.7,  0.3,
    0.9,  1.4, -1.2,
   -1.1,  0.5,  1.7,
    1.2, -1.5,  0.8,
   -0.8,  1.0, -0.4,
};
const double kY[8] = {1.2, 0.4, -0.7, 2.1, 0.1, -1.4, 1.8, -0.2};

n4m_matrix_view_t make_view(const double* data, std::int64_t rows,
                            std::int64_t cols) {
    n4m_matrix_view_t v{};
    N4M_TEST_REQUIRE(
        n4m_matrix_view_init_rowmajor(
            &v, const_cast<double*>(data), rows, cols, N4M_DTYPE_F64) == N4M_OK);
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

bool set_process_env(const char* name, const char* value) noexcept {
#if defined(_WIN32)
    return _putenv_s(name, value == nullptr ? "" : value) == 0;
#else
    if (value == nullptr) {
        return unsetenv(name) == 0;
    } else {
        return setenv(name, value, 1) == 0;
    }
#endif
}

std::string get_process_env(const char* name, bool& found) {
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) {
        found = false;
        return {};
    }
    std::string out(value, len > 0 ? len - 1U : 0U);
    std::free(value);
    found = true;
    return out;
#else
    const char* value = std::getenv(name);
    found = value != nullptr;
    return found ? std::string(value) : std::string{};
#endif
}

class EnvVarOverride {
public:
    EnvVarOverride(const char* name, const char* value)
        : name_(name) {
        previous_ = get_process_env(name, had_previous_);
        N4M_TEST_REQUIRE(set_process_env(name_.c_str(), value));
    }

    ~EnvVarOverride() noexcept {
        if (had_previous_) {
            (void)set_process_env(name_.c_str(), previous_.c_str());
        } else {
            (void)set_process_env(name_.c_str(), nullptr);
        }
    }

    EnvVarOverride(const EnvVarOverride&) = delete;
    EnvVarOverride& operator=(const EnvVarOverride&) = delete;

private:
    std::string name_;
    std::string previous_;
    bool had_previous_{false};
};

std::vector<double> materialized_ridge_oof(const std::int32_t* fold_ids,
                                           std::int32_t cv,
                                           double lambda,
                                           double& out_rmse) {
    constexpr std::int64_t n = 8;
    constexpr std::int64_t p = 3;
    std::vector<double> oof(static_cast<std::size_t>(n), 0.0);
    double sse = 0.0;

    for (std::int32_t fold = 0; fold < cv; ++fold) {
        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<std::int64_t> heldout;
        for (std::int64_t i = 0; i < n; ++i) {
            if (fold_ids[i] == fold) {
                heldout.push_back(i);
            } else {
                for (std::int64_t j = 0; j < p; ++j) {
                    train_x.push_back(kX[static_cast<std::size_t>(i * p + j)]);
                }
                train_y.push_back(kY[static_cast<std::size_t>(i)]);
            }
        }

        n4m_context_t* ctx = nullptr;
        n4m_config_t* cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
        n4m_matrix_view_t Xv = make_view(
            train_x.data(), static_cast<std::int64_t>(train_y.size()), p);
        n4m_matrix_view_t Yv = make_view(
            train_y.data(), static_cast<std::int64_t>(train_y.size()), 1);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_ridge_fit(
                             ctx, cfg, &Xv, &Yv, &lambda, 1, &result) == N4M_OK);
        const std::vector<double> coef = get_matrix(result, "coefficients", p, 1);
        const std::vector<double> intercept = get_matrix(result, "intercept", 1, 1);
        for (const auto row_i64 : heldout) {
            const auto row = static_cast<std::size_t>(row_i64);
            double pred = intercept[0];
            for (std::int64_t j = 0; j < p; ++j) {
                pred += kX[row * static_cast<std::size_t>(p) + static_cast<std::size_t>(j)] *
                        coef[static_cast<std::size_t>(j)];
            }
            oof[row] = pred;
            const double d = pred - kY[row];
            sse += d * d;
        }
        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
    out_rmse = std::sqrt(sse / static_cast<double>(n));
    return oof;
}

double materialized_ridge_rmse_dynamic(const std::vector<double>& X,
                                       const std::vector<double>& y,
                                       std::int64_t n,
                                       std::int64_t p,
                                       const std::int32_t* fold_ids,
                                       std::int32_t cv,
                                       double lambda,
                                       bool scale_x) {
    double sse = 0.0;
    for (std::int32_t fold = 0; fold < cv; ++fold) {
        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<std::int64_t> heldout;
        for (std::int64_t i = 0; i < n; ++i) {
            if (fold_ids[i] == fold) {
                heldout.push_back(i);
            } else {
                for (std::int64_t j = 0; j < p; ++j) {
                    train_x.push_back(
                        X[static_cast<std::size_t>(i * p + j)]);
                }
                train_y.push_back(y[static_cast<std::size_t>(i)]);
            }
        }

        n4m_context_t* ctx = nullptr;
        n4m_config_t* cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, scale_x ? 1 : 0) == N4M_OK);
        n4m_matrix_view_t Xv = make_view(
            train_x.data(), static_cast<std::int64_t>(train_y.size()), p);
        n4m_matrix_view_t Yv = make_view(
            train_y.data(), static_cast<std::int64_t>(train_y.size()), 1);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_estimators_ridge_fit(
                             ctx, cfg, &Xv, &Yv, &lambda, 1, &result) == N4M_OK);
        const std::vector<double> coef = get_matrix(result, "coefficients", p, 1);
        const std::vector<double> intercept = get_matrix(result, "intercept", 1, 1);
        for (const auto row_i64 : heldout) {
            const auto row = static_cast<std::size_t>(row_i64);
            double pred = intercept[0];
            for (std::int64_t j = 0; j < p; ++j) {
                pred += X[row * static_cast<std::size_t>(p) +
                          static_cast<std::size_t>(j)] *
                        coef[static_cast<std::size_t>(j)];
            }
            const double d = pred - y[row];
            sse += d * d;
        }
        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
    return std::sqrt(sse / static_cast<double>(n));
}

std::vector<double> materialized_pls_oof(const std::int32_t* fold_ids,
                                         std::int32_t cv,
                                         std::int32_t n_components,
                                         double& out_rmse,
                                         bool scale_x = false) {
    constexpr std::int64_t n = 8;
    constexpr std::int64_t p = 3;
    std::vector<double> oof(static_cast<std::size_t>(n), 0.0);
    double sse = 0.0;

    for (std::int32_t fold = 0; fold < cv; ++fold) {
        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<double> heldout_x;
        std::vector<std::int64_t> heldout;
        for (std::int64_t i = 0; i < n; ++i) {
            if (fold_ids[i] == fold) {
                heldout.push_back(i);
                for (std::int64_t j = 0; j < p; ++j) {
                    heldout_x.push_back(kX[static_cast<std::size_t>(i * p + j)]);
                }
            } else {
                for (std::int64_t j = 0; j < p; ++j) {
                    train_x.push_back(kX[static_cast<std::size_t>(i * p + j)]);
                }
                train_y.push_back(kY[static_cast<std::size_t>(i)]);
            }
        }

        n4m_context_t* ctx = nullptr;
        n4m_config_t* cfg = nullptr;
        n4m_model_t* model = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, scale_x ? 1 : 0) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_n_components(cfg, n_components) == N4M_OK);
        n4m_matrix_view_t Xv = make_view(
            train_x.data(), static_cast<std::int64_t>(train_y.size()), p);
        n4m_matrix_view_t Yv = make_view(
            train_y.data(), static_cast<std::int64_t>(train_y.size()), 1);
        N4M_TEST_REQUIRE(n4m_model_fit(ctx, cfg, &Xv, &Yv, &model) == N4M_OK);

        std::vector<double> pred(heldout.size(), 0.0);
        n4m_matrix_view_t Xhv = make_view(
            heldout_x.data(), static_cast<std::int64_t>(heldout.size()), p);
        n4m_matrix_view_t Phv = make_view(
            pred.data(), static_cast<std::int64_t>(heldout.size()), 1);
        N4M_TEST_REQUIRE(n4m_model_predict(ctx, model, &Xhv, &Phv) == N4M_OK);
        for (std::size_t i = 0; i < heldout.size(); ++i) {
            const auto row = static_cast<std::size_t>(heldout[i]);
            oof[row] = pred[i];
            const double d = pred[i] - kY[row];
            sse += d * d;
        }
        n4m_model_destroy(model);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
    }
    out_rmse = std::sqrt(sse / static_cast<double>(n));
    return oof;
}

void assert_close_vec(const std::vector<double>& got,
                      const std::vector<double>& expected,
                      const char* label) {
    N4M_TEST_REQUIRE(got.size() == expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(got[i] - expected[i]);
        if (diff > kTol) {
            throw std::runtime_error(
                std::string(label) + " mismatch at " + std::to_string(i) +
                " got=" + std::to_string(got[i]) +
                " expected=" + std::to_string(expected[i]));
        }
    }
}

void test_sweep_ridge_oof_matches_materialized_cv() {
    const std::int32_t folds[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    const double lambda = 0.5;
    double expected_rmse = 0.0;
    const std::vector<double> expected =
        materialized_ridge_oof(folds, 4, lambda, expected_rmse);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 8, 3);
    n4m_matrix_view_t Yv = make_view(kY, 8, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, 4, folds, 8,
                         &lambda, 1, nullptr, 0, 1, &result) == N4M_OK);

    const std::vector<double> got = get_matrix(result, "oof_predictions", 8, 1);
    assert_close_vec(got, expected, "oof_predictions");

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 1, 4);
    N4M_TEST_REQUIRE(std::fabs(scores[3] - expected_rmse) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_cv_rmse") -
                               expected_rmse) <= kTol);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_selects_minimum_candidate_and_generates_folds() {
    const double lambdas[2] = {0.05, 5.0};
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 8, 3);
    n4m_matrix_view_t Yv = make_view(kY, 8, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, 4, nullptr, 0,
                         lambdas, 2, nullptr, 0, 1, &result) == N4M_OK);
    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 2, 4);
    const double selected_id = get_scalar(result, "selected_candidate_id");
    const double selected_rmse = get_scalar(result, "selected_cv_rmse");
    const std::size_t min_id = (scores[3] <= scores[7]) ? 0U : 1U;
    N4M_TEST_REQUIRE(std::fabs(selected_id - static_cast<double>(min_id)) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(selected_rmse - scores[min_id * 4 + 3]) <= kTol);
    const std::vector<double> predictions = get_matrix(result, "predictions", 8, 1);
    for (const double v : predictions) N4M_TEST_REQUIRE(std::isfinite(v));

    const std::int32_t* fold_data = nullptr;
    std::int32_t fold_size = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_int_vector(
                         result, "fold_ids", &fold_data, &fold_size) == N4M_OK);
    N4M_TEST_REQUIRE(fold_size == 8);
    N4M_TEST_REQUIRE(fold_data[0] == 0);
    N4M_TEST_REQUIRE(fold_data[7] == 3);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_wide_dual_ridge_scores_match_materialized_cv() {
    constexpr std::int64_t n = 9;
    constexpr std::int64_t p = 14;
    constexpr std::int32_t cv = 3;
    std::vector<double> X(static_cast<std::size_t>(n * p), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < p; ++j) {
            const double ii = static_cast<double>(i + 1);
            const double jj = static_cast<double>(j + 1);
            X[static_cast<std::size_t>(i * p + j)] =
                std::sin(0.17 * ii * jj) +
                0.3 * std::cos(0.11 * (ii + jj)) +
                0.02 * static_cast<double>(i - j);
        }
        y[static_cast<std::size_t>(i)] =
            0.6 * X[static_cast<std::size_t>(i * p + 0)] -
            0.2 * X[static_cast<std::size_t>(i * p + 5)] +
            0.15 * X[static_cast<std::size_t>(i * p + 11)];
    }
    const std::int32_t folds[static_cast<std::size_t>(n)] =
        {0, 1, 2, 0, 1, 2, 0, 1, 2};
    const double lambdas[2] = {0.01, 0.4};
    const double expected0 =
        materialized_ridge_rmse_dynamic(X, y, n, p, folds, cv, lambdas[0], true);
    const double expected1 =
        materialized_ridge_rmse_dynamic(X, y, n, p, folds, cv, lambdas[1], true);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         lambdas, 2, nullptr, 0, 1, &result) == N4M_OK);

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 2, 4);
    N4M_TEST_REQUIRE(std::fabs(scores[3] - expected0) <= 1e-8);
    N4M_TEST_REQUIRE(std::fabs(scores[7] - expected1) <= 1e-8);
    const double selected_id = get_scalar(result, "selected_candidate_id");
    const std::size_t min_id = (expected0 <= expected1) ? 0U : 1U;
    N4M_TEST_REQUIRE(std::fabs(selected_id - static_cast<double>(min_id)) <= kTol);

    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);
    n4m_method_result_t* score_only = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds, n,
                         lambdas, 2, nullptr, 0, 1, &score_only) == N4M_OK);
    const std::vector<double> score_only_scores =
        get_matrix(score_only, "candidate_scores", 2, 4);
    N4M_TEST_REQUIRE(std::fabs(score_only_scores[3] - scores[3]) <= 1e-8);
    N4M_TEST_REQUIRE(std::fabs(score_only_scores[7] - scores[7]) <= 1e-8);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(score_only, "score_only") - 1.0) <= kTol);
    (void)get_matrix(score_only, "predictions", 0, 0);
    (void)get_matrix(score_only, "oof_predictions", 0, 0);
    (void)get_matrix(score_only, "coefficients", 0, 0);

    n4m_method_result_destroy(score_only);
    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_pls_oof_matches_materialized_cv() {
    const std::int32_t folds[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    const std::int32_t comps[1] = {2};
    double expected_rmse = 0.0;
    const std::vector<double> expected =
        materialized_pls_oof(folds, 4, comps[0], expected_rmse);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 8, 3);
    n4m_matrix_view_t Yv = make_view(kY, 8, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, 4, folds, 8,
                         nullptr, 0, comps, 1, 2, &result) == N4M_OK);

    const std::vector<double> got = get_matrix(result, "oof_predictions", 8, 1);
    assert_close_vec(got, expected, "pls_oof_predictions");

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 1, 4);
    N4M_TEST_REQUIRE(std::fabs(scores[1] - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[2] - 2.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[3] - expected_rmse) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "selected_head_id") -
                               1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_candidates") -
                               1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cv_fits") -
                               4.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_host_cv_fits") -
                               4.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cuda_device_cv_fits")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_final_fits") -
                               1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_host_final_fits") -
                               1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cuda_device_final_fits")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_materialized_cv_fits")) <= kTol);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_pls_component_grid_matches_materialized_cv_scores() {
    const std::int32_t folds[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    const std::int32_t comps[2] = {1, 2};
    double expected_rmse_1 = 0.0;
    double expected_rmse_2 = 0.0;
    (void)materialized_pls_oof(folds, 4, comps[0], expected_rmse_1);
    (void)materialized_pls_oof(folds, 4, comps[1], expected_rmse_2);

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 8, 3);
    n4m_matrix_view_t Yv = make_view(kY, 8, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, 4, folds, 8,
                         nullptr, 0, comps, 2, 2, &result) == N4M_OK);

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 2, 4);
    N4M_TEST_REQUIRE(std::fabs(scores[1] - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[2] - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[3] - expected_rmse_1) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[5] - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[6] - 2.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(scores[7] - expected_rmse_2) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_candidates") -
                               2.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cv_fits") -
                               4.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_final_fits") -
                               1.0) <= kTol);

    const double selected_id = get_scalar(result, "selected_candidate_id");
    const std::size_t min_id = (scores[3] <= scores[7]) ? 0U : 1U;
    N4M_TEST_REQUIRE(std::fabs(selected_id - static_cast<double>(min_id)) <= kTol);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_score_only_keeps_scores_and_skips_outputs() {
    const std::int32_t folds[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    const std::int32_t comps[2] = {1, 2};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 8, 3);
    n4m_matrix_view_t Yv = make_view(kY, 8, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, 4, folds, 8,
                         nullptr, 0, comps, 2, 2, &result) == N4M_OK);

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 2, 4);
    N4M_TEST_REQUIRE(std::isfinite(scores[3]));
    N4M_TEST_REQUIRE(std::isfinite(scores[7]));
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "score_only") - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_candidates") -
                               2.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cv_fits") -
                               4.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_host_cv_fits") -
                               4.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cuda_device_cv_fits")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_final_fits")) <= kTol);
    (void)get_matrix(result, "oof_predictions", 0, 0);
    (void)get_matrix(result, "predictions", 0, 0);
    (void)get_matrix(result, "coefficients", 0, 0);
    (void)get_matrix(result, "intercept", 0, 0);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_wide_pls_score_only_uses_moment_route() {
    constexpr std::int64_t n = 20;
    constexpr std::int64_t p = 1024;
    constexpr std::int32_t cv = 4;
    std::vector<double> X(static_cast<std::size_t>(n * p), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < p; ++j) {
            const double ii = static_cast<double>(i + 1);
            const double jj = static_cast<double>(j + 1);
            const double v =
                std::sin(0.007 * ii * jj) +
                0.15 * std::cos(0.013 * (ii + jj)) +
                0.001 * static_cast<double>((i + j) % 7);
            X[static_cast<std::size_t>(i * p + j)] = v;
        }
        y[static_cast<std::size_t>(i)] =
            0.5 * X[static_cast<std::size_t>(i * p + 3)] -
            0.25 * X[static_cast<std::size_t>(i * p + 127)] +
            0.1 * X[static_cast<std::size_t>(i * p + 701)];
    }
    std::vector<std::int32_t> folds(static_cast<std::size_t>(n), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t comps[3] = {1, 2, 3};

    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(X.data(), n, p);
    n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                         ctx, cfg, &Xv, &Yv, cv, folds.data(), n,
                         nullptr, 0, comps, 3, 2, &result) == N4M_OK);

    const std::vector<double> scores =
        get_matrix(result, "candidate_scores", 3, 4);
    for (std::size_t row = 0; row < 3; ++row) {
        N4M_TEST_REQUIRE(std::fabs(scores[row * 4 + 1] - 1.0) <= kTol);
        N4M_TEST_REQUIRE(std::isfinite(scores[row * 4 + 3]));
    }
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "score_only") - 1.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_candidates") -
                               3.0) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cv_fits") -
                               static_cast<double>(cv)) <= kTol);
    const bool cuda_available =
        n4m_backend_is_available(N4M_BACKEND_CUDA) != 0;
    if (cuda_available) {
        N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_host_cv_fits")) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cuda_device_cv_fits") -
                                   static_cast<double>(cv)) <= kTol);
    } else {
        N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_host_cv_fits") -
                                   static_cast<double>(cv)) <= kTol);
        N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_cuda_device_cv_fits")) <= kTol);
    }
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_materialized_cv_fits")) <= kTol);
    N4M_TEST_REQUIRE(std::fabs(get_scalar(result, "n_pls_moment_final_fits")) <= kTol);
    (void)get_matrix(result, "predictions", 0, 0);
    (void)get_matrix(result, "coefficients", 0, 0);

    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

void test_sweep_cuda_many_batched_opt_in_matches_default() {
    if (n4m_backend_is_available(N4M_BACKEND_CUDA) == 0) {
        return;
    }

    constexpr std::int64_t n = 24;
    constexpr std::int64_t p = 512;
    constexpr std::int32_t cv = 4;
    std::vector<double> X(static_cast<std::size_t>(n * p), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < p; ++j) {
            const double ii = static_cast<double>(i + 1);
            const double jj = static_cast<double>(j + 1);
            X[static_cast<std::size_t>(i * p + j)] =
                std::sin(0.011 * ii * jj) +
                0.2 * std::cos(0.017 * (ii + jj)) +
                0.001 * static_cast<double>((i * 3 + j) % 11);
        }
        y[static_cast<std::size_t>(i)] =
            0.45 * X[static_cast<std::size_t>(i * p + 7)] -
            0.20 * X[static_cast<std::size_t>(i * p + 133)] +
            0.12 * X[static_cast<std::size_t>(i * p + 401)];
    }
    std::vector<std::int32_t> folds(static_cast<std::size_t>(n), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        folds[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % cv);
    }
    const std::int32_t comps[3] = {1, 2, 3};

    auto run_once = [&](bool batched) {
        EnvVarOverride batched_env("N4M_CUDA_PLS_MANY_BATCHED", nullptr);
        EnvVarOverride legacy_env("N4M_CUDA_PLS_MANY_LEGACY", nullptr);
        n4m_context_t* ctx = nullptr;
        n4m_config_t* cfg = nullptr;
        N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, 0) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_aom_score_only(cfg, 1) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_cuda_pls_min_device_features(
                             cfg, 1) == N4M_OK);
        N4M_TEST_REQUIRE(n4m_config_set_cuda_pls_many_batched(
                             cfg, batched ? 1 : 0) == N4M_OK);
        std::int32_t observed_many_batched = -1;
        N4M_TEST_REQUIRE(n4m_config_get_cuda_pls_many_batched(
                             cfg, &observed_many_batched) == N4M_OK);
        N4M_TEST_REQUIRE(observed_many_batched == (batched ? 1 : 0));
        N4M_TEST_REQUIRE(n4m_config_set_cuda_pls_many_batched(cfg, 2) ==
                         N4M_ERR_INVALID_ARGUMENT);
        n4m_matrix_view_t Xv = make_view(X.data(), n, p);
        n4m_matrix_view_t Yv = make_view(y.data(), n, 1);
        n4m_method_result_t* result = nullptr;
        N4M_TEST_REQUIRE(n4m_model_selection_sweep_run(
                             ctx, cfg, &Xv, &Yv, cv, folds.data(), n,
                             nullptr, 0, comps, 3, 2, &result) == N4M_OK);
        N4M_TEST_REQUIRE(
            std::fabs(get_scalar(result, "n_pls_moment_cuda_device_cv_fits") -
                      static_cast<double>(cv)) <= kTol);
        N4M_TEST_REQUIRE(
            std::fabs(get_scalar(result, "n_pls_moment_host_cv_fits")) <= kTol);
        std::vector<double> scores =
            get_matrix(result, "candidate_scores", 3, 4);
        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        return scores;
    };

    const std::vector<double> default_scores = run_once(false);
    const std::vector<double> batched_scores = run_once(true);
    assert_close_vec(batched_scores, default_scores,
                     "cuda_many_batched_candidate_scores");
}

}  // namespace

void register_sweep_tests(n4m_testing::Runner& r) {
    r.run("sweep/ridge_oof_matches_materialized_cv",
          test_sweep_ridge_oof_matches_materialized_cv);
    r.run("sweep/selects_minimum_candidate_and_generates_folds",
          test_sweep_selects_minimum_candidate_and_generates_folds);
    r.run("sweep/wide_dual_ridge_scores_match_materialized_cv",
          test_sweep_wide_dual_ridge_scores_match_materialized_cv);
    r.run("sweep/pls_oof_matches_materialized_cv",
          test_sweep_pls_oof_matches_materialized_cv);
    r.run("sweep/pls_component_grid_matches_materialized_cv_scores",
          test_sweep_pls_component_grid_matches_materialized_cv_scores);
    r.run("sweep/score_only_keeps_scores_and_skips_outputs",
          test_sweep_score_only_keeps_scores_and_skips_outputs);
    r.run("sweep/wide_pls_score_only_uses_moment_route",
          test_sweep_wide_pls_score_only_uses_moment_route);
    r.run("sweep/cuda_many_batched_opt_in_matches_default",
          test_sweep_cuda_many_batched_opt_in_matches_default);
}
