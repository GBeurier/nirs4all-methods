// SPDX-License-Identifier: CECILL-2.1
//
// Public ABI tests for n4m_moments_* raw/centered sufficient statistics.

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

void register_moments_tests(n4m_testing::Runner& r);

namespace {

constexpr double kTol = 1e-12;

const double kX[12] = {
    1.0,  2.0, 3.0,
    4.0,  5.0, 6.0,
    7.0,  8.0, 9.0,
    2.0, -1.0, 0.5,
};
const double kY[8] = {
    1.0, -1.0,
    0.5,  2.0,
    3.0,  1.0,
   -2.0,  0.25,
};

struct Expected {
    std::int64_t n;
    std::int64_t p;
    std::int64_t q;
    std::vector<double> x_sum, y_sum, xtx, xty, yty;
    std::vector<double> x_mean, y_mean, cxx, cxy, cyy;
};

n4m_matrix_view_t make_view(const double* data, std::int64_t rows,
                            std::int64_t cols) {
    n4m_matrix_view_t v{};
    N4M_TEST_REQUIRE(
        n4m_matrix_view_init_rowmajor(
            &v, const_cast<double*>(data), rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return v;
}

Expected expected_for_rows(const std::vector<std::int64_t>& rows) {
    Expected e{};
    e.n = static_cast<std::int64_t>(rows.size());
    e.p = 3;
    e.q = 2;
    e.x_sum.assign(3, 0.0);
    e.y_sum.assign(2, 0.0);
    e.xtx.assign(9, 0.0);
    e.xty.assign(6, 0.0);
    e.yty.assign(4, 0.0);
    for (const std::int64_t r : rows) {
        const double* x = kX + r * 3;
        const double* y = kY + r * 2;
        for (std::size_t i = 0; i < 3; ++i) e.x_sum[i] += x[i];
        for (std::size_t j = 0; j < 2; ++j) e.y_sum[j] += y[j];
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                e.xtx[i * 3 + j] += x[i] * x[j];
            }
            for (std::size_t j = 0; j < 2; ++j) {
                e.xty[i * 2 + j] += x[i] * y[j];
            }
        }
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                e.yty[i * 2 + j] += y[i] * y[j];
            }
        }
    }
    const double inv_n = 1.0 / static_cast<double>(e.n);
    e.x_mean.assign(3, 0.0);
    e.y_mean.assign(2, 0.0);
    for (std::size_t i = 0; i < 3; ++i) e.x_mean[i] = e.x_sum[i] * inv_n;
    for (std::size_t i = 0; i < 2; ++i) e.y_mean[i] = e.y_sum[i] * inv_n;
    e.cxx.assign(9, 0.0);
    e.cxy.assign(6, 0.0);
    e.cyy.assign(4, 0.0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            e.cxx[i * 3 + j] =
                e.xtx[i * 3 + j] -
                e.x_sum[i] * e.x_sum[j] * inv_n;
        }
        for (std::size_t j = 0; j < 2; ++j) {
            e.cxy[i * 2 + j] =
                e.xty[i * 2 + j] -
                e.x_sum[i] * e.y_sum[j] * inv_n;
        }
    }
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            e.cyy[i * 2 + j] =
                e.yty[i * 2 + j] -
                e.y_sum[i] * e.y_sum[j] * inv_n;
        }
    }
    return e;
}

void check_scalar(const n4m_method_result_t* result,
                  const char* key,
                  double expected) {
    double got = 0.0;
    N4M_TEST_REQUIRE(n4m_method_result_get_scalar(result, key, &got) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(got - expected) <= kTol);
}

void check_matrix(const n4m_method_result_t* result,
                  const char* key,
                  const std::vector<double>& expected,
                  std::int64_t rows,
                  std::int64_t cols) {
    const double* data = nullptr;
    std::int64_t got_rows = 0;
    std::int64_t got_cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, key, &data, &got_rows, &got_cols) == N4M_OK);
    N4M_TEST_REQUIRE(got_rows == rows);
    N4M_TEST_REQUIRE(got_cols == cols);
    N4M_TEST_REQUIRE(data != nullptr);
    const auto n = static_cast<std::size_t>(rows * cols);
    N4M_TEST_REQUIRE(expected.size() == n);
    for (std::size_t i = 0; i < n; ++i) {
        const double diff = std::fabs(data[i] - expected[i]);
        if (diff > kTol) {
            throw std::runtime_error(
                std::string(key) + " mismatch at " + std::to_string(i) +
                " got=" + std::to_string(data[i]) +
                " expected=" + std::to_string(expected[i]));
        }
    }
}

void check_result(const n4m_method_result_t* result, const Expected& e) {
    check_scalar(result, "n_samples", static_cast<double>(e.n));
    check_scalar(result, "n_features", static_cast<double>(e.p));
    check_scalar(result, "n_targets", static_cast<double>(e.q));
    check_matrix(result, "x_sum", e.x_sum, 1, e.p);
    check_matrix(result, "y_sum", e.y_sum, 1, e.q);
    check_matrix(result, "xtx", e.xtx, e.p, e.p);
    check_matrix(result, "xty", e.xty, e.p, e.q);
    check_matrix(result, "yty", e.yty, e.q, e.q);
    check_matrix(result, "x_mean", e.x_mean, 1, e.p);
    check_matrix(result, "y_mean", e.y_mean, 1, e.q);
    check_matrix(result, "cxx", e.cxx, e.p, e.p);
    check_matrix(result, "cxy", e.cxy, e.p, e.q);
    check_matrix(result, "cyy", e.cyy, e.q, e.q);
}

void test_moments_compute_all_rows() {
    n4m_context_t* ctx = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 4, 3);
    n4m_matrix_view_t Yv = make_view(kY, 4, 2);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(n4m_moments_compute(ctx, &Xv, &Yv, &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);
    check_result(result, expected_for_rows({0, 1, 2, 3}));
    n4m_method_result_destroy(result);
    n4m_context_destroy(ctx);
}

void test_moments_subset_and_subtract_are_fold_exact() {
    n4m_context_t* ctx = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX, 4, 3);
    n4m_matrix_view_t Yv = make_view(kY, 4, 2);

    n4m_method_result_t* all = nullptr;
    N4M_TEST_REQUIRE(n4m_moments_compute(ctx, &Xv, &Yv, &all) == N4M_OK);

    const std::int64_t heldout_rows[2] = {0, 2};
    n4m_method_result_t* heldout = nullptr;
    N4M_TEST_REQUIRE(n4m_moments_subset_compute(
                         ctx, &Xv, &Yv, heldout_rows, 2, &heldout) == N4M_OK);
    check_result(heldout, expected_for_rows({0, 2}));

    n4m_method_result_t* train = nullptr;
    N4M_TEST_REQUIRE(
        n4m_moments_subtract(ctx, all, heldout, &train) == N4M_OK);
    check_result(train, expected_for_rows({1, 3}));

    n4m_method_result_destroy(train);
    n4m_method_result_destroy(heldout);
    n4m_method_result_destroy(all);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_moments_tests(n4m_testing::Runner& r) {
    r.run("moments_compute_all_rows", test_moments_compute_all_rows);
    r.run("moments_subset_and_subtract_are_fold_exact",
          test_moments_subset_and_subtract_are_fold_exact);
}
