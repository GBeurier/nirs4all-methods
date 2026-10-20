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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
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
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rmse(y_true, y_pred, 5, &rmse) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_mae(y_true, y_pred, 5, &mae) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_bias(y_true, y_pred, 5, &bias) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_sep(y_true, y_pred, 5, &sep) == N4M_OK);
    N4M_TEST_REQUIRE(rmse == 0.0);
    N4M_TEST_REQUIRE(mae == 0.0);
    N4M_TEST_REQUIRE(bias == 0.0);
    N4M_TEST_REQUIRE(sep == 0.0);
    double r2 = -1.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_r2(y_true, y_pred, 5, &r2) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(r2 - 1.0) < 1e-15);
}

void test_metric_smoke_constant_bias() {
    const double y_true[4] = {1.0, 2.0, 3.0, 4.0};
    const double y_pred[4] = {1.5, 2.5, 3.5, 4.5};   // y_pred = y_true + 0.5
    double bias = 0.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_bias(y_true, y_pred, 4, &bias) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(bias - 0.5) < 1e-15);
    double rmse = 0.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rmse(y_true, y_pred, 4, &rmse) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(rmse - 0.5) < 1e-15);
    double mae = 0.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_mae(y_true, y_pred, 4, &mae) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(mae - 0.5) < 1e-15);
    // SEP = std of (y_pred - y_true) = std([0.5, 0.5, 0.5, 0.5]) = 0.
    double sep = -1.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_sep(y_true, y_pred, 4, &sep) == N4M_OK);
    N4M_TEST_REQUIRE(sep == 0.0);
    // RPD = std(y_true) / SEP → +inf when SEP == 0.
    double rpd = 0.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rpd(y_true, y_pred, 4, &rpd) == N4M_OK);
    N4M_TEST_REQUIRE(std::isinf(rpd));
}

void test_metric_smoke_invalid_n() {
    const double y[1] = {0.0};
    double out = 0.0;
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rmse(y, y, 0, &out)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_mae(y, y, -3, &out)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rmse(nullptr, y, 1, &out)
                     == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rmse(y, y, 1, nullptr)
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
    const n4m_status_t st = n4m_outlier_detection_hotelling_t2(
        Xv, /*n_components=*/2, /*alpha=*/0.05, t2, 6, &ucl);
    N4M_TEST_REQUIRE(st == N4M_OK);
    for (int i = 0; i < 6; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(t2[i]));
    }
    N4M_TEST_REQUIRE(ucl > 0.0);

    // Invalid args.
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(Xv, 0, 0.05, t2, 6, &ucl)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(Xv, 2, 1.5, t2, 6, &ucl)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(Xv, 2, 0.05, t2, 5, &ucl)
                     == N4M_ERR_SHAPE_MISMATCH);
}

// Review b8fix3 P0-3: column 3 = column 1 + column 2, so the centred rank is
// 2 and the SVD's third singular value is rounding noise (~3e-17). Only the
// two resolved directions enter T²; references are exact rationals from the
// inverse covariance of the two independent columns, and the UCL uses r = 2:
// 2 * 3 * 5 / (4 * 2) * F^{-1}(0.95; 2, 2) = 3.75 * 19.
void test_hotelling_t2_rank_deficient() {
    double X[4 * 3] = {1, 2, 3, 4, 5, 9, 6, 3, 9, 2, 8, 10};
    const n4m_matrix_view_t Xv = make_rowmajor_view(X, 4, 3);
    const double ref[4] = {5439.0 / 2428.0, 417.0 / 2428.0, 4113.0 / 2428.0, 4599.0 / 2428.0};
    double t2[4] = {0}, ucl = -1.0;
    int32_t rank = -1;
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2_resolved(Xv, 3, 0.05, t2, 4, &ucl,
                                                                  &rank) == N4M_OK);
    N4M_TEST_REQUIRE(rank == 2);
    for (int i = 0; i < 4; ++i) N4M_TEST_REQUIRE(std::abs(t2[i] - ref[i]) <= 1e-12);
    N4M_TEST_REQUIRE(std::abs(ucl - 71.25) <= 1e-9);
    double t2_old[4] = {0}, ucl_old = -1.0;
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(Xv, 3, 0.05, t2_old, 4, &ucl_old) ==
                     N4M_OK);
    for (int i = 0; i < 4; ++i) N4M_TEST_REQUIRE(t2_old[i] == t2[i]);
    N4M_TEST_REQUIRE(ucl_old == ucl);
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2_resolved(Xv, 3, 0.05, t2, 4, &ucl,
                                                                  nullptr) ==
                     N4M_ERR_NULL_POINTER);
    // Constant X: no resolved direction.
    double C[4 * 2] = {5, -1, 5, -1, 5, -1, 5, -1};
    const n4m_matrix_view_t Cv = make_rowmajor_view(C, 4, 2);
    N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2_resolved(Cv, 2, 0.05, t2, 4, &ucl,
                                                                  &rank) == N4M_OK);
    N4M_TEST_REQUIRE(rank == 0 && ucl == 0.0);
    for (int i = 0; i < 4; ++i) N4M_TEST_REQUIRE(t2[i] == 0.0);
}

// Review b8fix4 P0-2: a plain rounded mean left the same residual on every
// observation of a non-dyadic constant column (ten values 0.1, k = 1: rank 1,
// T² = 0.9, UCL 5.63). The corrected two-pass mean centres it to exact zeros,
// at every scale. Next to constant columns, a varying column i * s keeps
// rank 1 and the analytic T²_i = 12 z_i^2 / (n (n + 1)), z_i = i - (n-1)/2.
void test_hotelling_t2_constant_columns() {
    const double constants[] = {0.1, 1.0 / 3.0, -2.7e-7, 12345.678, 314159.26535};
    const double scales[] = {1.0, 3.0, 1e-7, 1e7, 0x1p60};
    for (const int64_t n : {int64_t{3}, int64_t{10}, int64_t{101}, int64_t{1000}}) {
        std::vector<double> x(static_cast<size_t>(n)), t2(static_cast<size_t>(n));
        for (const double c : constants) for (const double s : scales) {
            std::fill(x.begin(), x.end(), c * s);
            double ucl = -1.0;
            int32_t rank = -1;
            const n4m_matrix_view_t Xv = make_rowmajor_view(x.data(), n, 1);
            N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2_resolved(
                                 Xv, 1, 0.05, t2.data(), n, &ucl, &rank) == N4M_OK);
            N4M_TEST_REQUIRE(rank == 0 && ucl == 0.0);
            for (double v : t2) N4M_TEST_REQUIRE(v == 0.0);
        }
    }
    for (const int64_t n : {int64_t{10}, int64_t{101}}) {
        for (const double s : scales) {
            std::vector<double> x(static_cast<size_t>(3 * n)), t2(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                x[static_cast<size_t>(3 * i)] = 0.1 * s;
                x[static_cast<size_t>(3 * i + 1)] = static_cast<double>(i) * s;
                x[static_cast<size_t>(3 * i + 2)] = -s / 3.0;
            }
            double ucl = -1.0;
            int32_t rank = -1;
            const n4m_matrix_view_t Xv = make_rowmajor_view(x.data(), n, 3);
            N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2_resolved(
                                 Xv, 2, 0.05, t2.data(), n, &ucl, &rank) == N4M_OK);
            N4M_TEST_REQUIRE(rank == 1);
            const double dn = static_cast<double>(n);
            for (int64_t i = 0; i < n; ++i) {
                const double z = static_cast<double>(i) - (dn - 1.0) / 2.0;
                const double ref = 12.0 * z * z / (dn * (dn + 1.0));
                N4M_TEST_REQUIRE(std::abs(t2[static_cast<size_t>(i)] - ref) <=
                                 1e-12 * std::max(1.0, ref));
            }
            // Same UCL as the varying column alone (r = 1).
            std::vector<double> col(static_cast<size_t>(n)), t2_col(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) col[static_cast<size_t>(i)] = static_cast<double>(i) * s;
            double ucl_col = -1.0;
            const n4m_matrix_view_t Cv = make_rowmajor_view(col.data(), n, 1);
            N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(Cv, 1, 0.05, t2_col.data(), n,
                                                                 &ucl_col) == N4M_OK);
            N4M_TEST_REQUIRE(ucl == ucl_col && ucl > 0.0);
        }
    }
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
    const n4m_status_t st = n4m_outlier_detection_q_residuals(
        Xv, /*n_components=*/2, /*alpha=*/0.05, q, 6, &ucl);
    N4M_TEST_REQUIRE(st == N4M_OK);
    for (int i = 0; i < 6; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(q[i]));
        N4M_TEST_REQUIRE(q[i] >= -1e-10);  // residual energy is non-negative
    }
    N4M_TEST_REQUIRE(std::isfinite(ucl));
}

// Jackson-Mudholkar UCL for a single residual eigenvalue lambda: theta_a =
// lambda^a, h0 = 1/3, so UCL = lambda (7/9 + ca sqrt(2) / 3)^3. ca are SciPy
// norm.isf values; the kernel's quantile approximation is accurate to about
// 1.15e-9 relative, hence the 1e-8 tolerance.
double jm_single_ucl(double lambda, double ca) {
    const double base = 7.0 / 9.0 + ca * std::sqrt(2.0) / 3.0;
    return lambda * base * base * base;
}
bool rel_close(double got, double want, double tol) {
    return std::abs(got - want) <= tol * std::abs(want);
}

// Review b8fix5 P0: the residual moments theta_1..3 overflowed (2^200) or
// underflowed (2^-200) and the UCL was published as 0 with status OK. The
// columns of [[2,1],[2,-1],[-2,1],[-2,-1]] are orthogonal and centred, so with
// k = 1 the residual is the second column exactly: Q_i = 4^s, one residual
// eigenvalue 4^(s+1) / 3, UCL = 1.29001e121 at s = 200. Results are exactly
// equivariant under power-of-two scaling; values that are not representable
// (Q = 2^1022 next to UCL 2^1024.3, or Q below DBL_MIN) are refused.
void test_q_residuals_scaling() {
    const double base[8] = {2, 1, 2, -1, -2, 1, -2, -1};
    double q0[4] = {0}, ucl0 = -1.0;
    {
        double X[8];
        std::copy(base, base + 8, X);
        N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(X, 4, 2), 1, 0.05,
                                                           q0, 4, &ucl0) == N4M_OK);
    }
    for (int i = 0; i < 4; ++i) N4M_TEST_REQUIRE(q0[i] == 1.0);
    N4M_TEST_REQUIRE(rel_close(ucl0, jm_single_ucl(4.0 / 3.0, 1.6448536269514729), 1e-8));
    for (const int s : {-511, -400, -256, -200, -100, 1, 100, 200, 256, 400, 510}) {
        double X[8], q[4] = {0}, ucl = -1.0;
        for (int i = 0; i < 8; ++i) X[i] = std::ldexp(base[i], s);
        N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(X, 4, 2), 1, 0.05,
                                                           q, 4, &ucl) == N4M_OK);
        for (int i = 0; i < 4; ++i) N4M_TEST_REQUIRE(q[i] == std::ldexp(q0[i], 2 * s));
        N4M_TEST_REQUIRE(ucl == std::ldexp(ucl0, 2 * s));
        N4M_TEST_REQUIRE(rel_close(ucl, jm_single_ucl(std::ldexp(4.0, 2 * s) / 3.0,
                                                      1.6448536269514729), 1e-8));
    }
    {
        double X[8], q[4] = {0}, ucl = -1.0;
        for (int i = 0; i < 8; ++i) X[i] = std::ldexp(base[i], 200);
        N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(X, 4, 2), 1, 0.05,
                                                           q, 4, &ucl) == N4M_OK);
        N4M_TEST_REQUIRE(rel_close(ucl, 1.29001e121, 1e-5) && q[0] < ucl);
    }
    for (const int s : {511, 600, -512, -600}) {
        double X[8], q[4] = {0}, ucl = -1.0;
        for (int i = 0; i < 8; ++i) X[i] = std::ldexp(base[i], s);
        N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(X, 4, 2), 1, 0.05,
                                                           q, 4, &ucl) ==
                         N4M_ERR_NUMERICAL_FAILURE);
    }
    // alpha enters through the upper quantile directly: 1 - 1e-20 rounds to
    // 1 (the former UCL was 0); alpha = 0.5 gives ca = 0; alpha = 0.99 makes
    // the base 7/9 + ca sqrt(2)/3 negative: no limit, explicit failure. For
    // alpha = 0.9 the small base (0.17) amplifies the quantile's 1.15e-9
    // relative error about tenfold.
    const struct { double alpha, ca, tol; } quantiles[] = {
        {0.5, 0.0, 1e-15}, {0.9, -1.2815515655446004, 5e-8}, {1e-20, 9.262340089798409, 1e-8}};
    for (const auto& c : quantiles) {
        double X[8], q[4] = {0}, ucl = -1.0;
        std::copy(base, base + 8, X);
        N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(X, 4, 2), 1,
                                                           c.alpha, q, 4, &ucl) == N4M_OK);
        N4M_TEST_REQUIRE(rel_close(ucl, jm_single_ucl(4.0 / 3.0, c.ca), c.tol));
    }
    double X[8], q[4] = {0}, ucl = -1.0;
    std::copy(base, base + 8, X);
    N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(X, 4, 2), 1, 0.99, q, 4,
                                                       &ucl) == N4M_ERR_NUMERICAL_FAILURE);
}

// Review b8fix5 P1: ten rows of the constant 0.1 gave Q = 1.9e-65 everywhere
// next to UCL 0. With the corrected centring shared with Hotelling T²,
// constant columns centre to exact zeros; when no residual direction is
// resolved (constant X, k = p, or numerical rank <= k) Q = 0 and UCL = 0
// exactly (documented state). Next to non-dyadic constant and translated
// columns, two orthogonal varying columns a = [3,1,-1,-3] s and
// b = [1,-1,-1,1] s with k = 1 leave b as the residual: Q_i = (b_i s)^2 and
// one residual eigenvalue 4 s^2 / 3.
void test_q_residuals_constant_columns() {
    const double constants[] = {0.1, 1.0 / 3.0, -2.7e-7, 12345.678, 314159.26535};
    const double scales[] = {1.0, 3.0, 1e-7, 1e7, 0x1p60, 0x1p-200, 0x1p200};
    for (const int64_t n : {int64_t{3}, int64_t{10}, int64_t{101}}) {
        for (const int64_t p : {int64_t{1}, int64_t{2}}) {
            for (const double c : constants) for (const double s : scales) {
                std::vector<double> x(static_cast<size_t>(n * p), c * s), q(static_cast<size_t>(n));
                double ucl = -1.0;
                N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(
                                     make_rowmajor_view(x.data(), n, p), 1, 0.05, q.data(), n,
                                     &ucl) == N4M_OK);
                N4M_TEST_REQUIRE(ucl == 0.0);
                for (double v : q) N4M_TEST_REQUIRE(v == 0.0);
            }
        }
    }
    const double a[4] = {3, 1, -1, -3}, b[4] = {1, -1, -1, 1};
    for (const double s : scales) {
        for (const double offset : {0.0, 0.1, 12345.678}) {
            double x[4 * 4], q[4] = {0}, ucl = -1.0;
            for (int i = 0; i < 4; ++i) {
                x[4 * i] = 0.1 * s;
                x[4 * i + 1] = (a[i] + offset) * s;
                x[4 * i + 2] = -s / 3.0;
                x[4 * i + 3] = (b[i] + offset) * s;
            }
            N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(x, 4, 4), 1,
                                                               0.05, q, 4, &ucl) == N4M_OK);
            // Exact for dyadic data; the rounding of the translated values
            // bounds the error otherwise.
            const double tol = offset == 0.0 ? 1e-15 : 1e-9;
            for (int i = 0; i < 4; ++i) N4M_TEST_REQUIRE(rel_close(q[i], s * s, tol));
            N4M_TEST_REQUIRE(rel_close(ucl, jm_single_ucl(4.0 * s * s / 3.0, 1.6448536269514729),
                                       1e-8));
            // k = 2 spans both varying columns: residual unresolved.
            N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(make_rowmajor_view(x, 4, 4), 2,
                                                               0.05, q, 4, &ucl) == N4M_OK);
            N4M_TEST_REQUIRE(ucl == 0.0);
            for (double v : q) N4M_TEST_REQUIRE(v == 0.0);
        }
    }
}

// Sylvester Hadamard matrix H_n (n a power of two), row-major: columns
// 1..n-1 are orthogonal, centred and of squared norm n.
std::vector<double> sylvester_hadamard(int n) {
    std::vector<double> h(static_cast<size_t>(n) * static_cast<size_t>(n), 1.0);
    for (int m = 1; m < n; m *= 2) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                const double v = h[static_cast<size_t>(i) * static_cast<size_t>(n) +
                                   static_cast<size_t>(j)];
                h[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j + m)] = v;
                h[static_cast<size_t>(i + m) * static_cast<size_t>(n) + static_cast<size_t>(j)] = v;
                h[static_cast<size_t>(i + m) * static_cast<size_t>(n) +
                  static_cast<size_t>(j + m)] = -v;
            }
        }
    }
    return h;
}

// Columns 1..scales.size() of H_n multiplied by scales: with k = 1 the
// residual eigenvalues are n c_j^2 / (n - 1) for the scales after the first
// (largest) one, so the Jackson-Mudholkar limit is known exactly.
std::vector<double> scaled_hadamard(int n, const std::vector<double>& scales) {
    const std::vector<double> h = sylvester_hadamard(n);
    const size_t p = scales.size();
    std::vector<double> x(static_cast<size_t>(n) * p);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i)
        for (size_t j = 0; j < p; ++j) x[i * p + j] = h[i * static_cast<size_t>(n) + j + 1] * scales[j];
    return x;
}

// Review b8fix6 P0-1: for h0 < 0 the transformation Y = (Q / theta_1)^h0
// decreases, so the upper quantile of Q maps to the LOWER quantile of Y. The
// former |h0| A kept the upper one: columns 1..14 of H_16 times
// [2, 1, 1/4 x 12], k = 1 (h0 = -0.06764684) gave UCL 0.365166 and flagged all
// 16 observations (Q_i = 1.75). References: the Jackson-Mudholkar formula with
// the signed h0 on the exact residual eigenvalues, mpmath at 40 digits; the
// tolerance covers the 1.15e-9 normal-quantile approximation. The second scale
// c / 4 moves h0 through 0 (root near c = 4.5314): the limit is continuous on
// both sides. With residual eigenvalues 3.25^2 and 22 times 1 (H_32, h0 =
// -0.46), the base 1 + g is -0.0144 at alpha = 1e-6: no Jackson-Mudholkar
// quantile, explicit failure; it is positive at 0.05 and 0.1.
void test_q_residuals_negative_h0() {
    const auto ucl_of = [](int n, const std::vector<double>& scales, double alpha,
                           n4m_status_t* status) {
        std::vector<double> x = scaled_hadamard(n, scales);
        std::vector<double> q(static_cast<size_t>(n));
        double ucl = -1.0;
        *status = n4m_outlier_detection_q_residuals(
            make_rowmajor_view(x.data(), n, static_cast<std::int64_t>(scales.size())), 1, alpha,
            q.data(), n, &ucl);
        return std::make_pair(ucl, q);
    };
    std::vector<double> astra = {2.0, 1.0};
    astra.resize(14, 0.25);
    const struct { double alpha, ref; } astra_cases[] = {
        {0.05, 5.2294583820033547}, {0.01, 9.6906288186272182}, {0.9, 0.47906393402987125}};
    for (const auto& c : astra_cases) {
        n4m_status_t st = N4M_ERR_INTERNAL;
        const auto r = ucl_of(16, astra, c.alpha, &st);
        N4M_TEST_REQUIRE(st == N4M_OK);
        N4M_TEST_REQUIRE(rel_close(r.first, c.ref, 5e-9));
        for (double v : r.second) N4M_TEST_REQUIRE(rel_close(v, 1.75, 1e-14));
        if (c.alpha <= 0.05) for (double v : r.second) N4M_TEST_REQUIRE(v < r.first);
    }
    // h0 = -3.66e-3, -2.12e-5, +8.82e-4, +4.94e-2 (alpha = 0.05).
    const struct { double c, ref; } crossing[] = {
        {4.5, 6.3099963890565998}, {4.53125, 6.3814368019494924},
        {4.5390625, 6.3993694611641446}, {5.0, 7.5091436613414903}};
    double previous = 0.0;
    for (const auto& c : crossing) {
        std::vector<double> scales = {2.0, c.c / 4.0};
        scales.resize(14, 0.25);
        n4m_status_t st = N4M_ERR_INTERNAL;
        const double ucl = ucl_of(16, scales, 0.05, &st).first;
        N4M_TEST_REQUIRE(st == N4M_OK);
        N4M_TEST_REQUIRE(rel_close(ucl, c.ref, 5e-9));
        N4M_TEST_REQUIRE(ucl > previous);
        previous = ucl;
    }
    std::vector<double> wide = {8.0, 3.25};
    wide.resize(24, 1.0);
    n4m_status_t st = N4M_OK;
    (void)ucl_of(32, wide, 1e-6, &st);
    N4M_TEST_REQUIRE(st == N4M_ERR_NUMERICAL_FAILURE);
    const struct { double alpha, ref; } wide_cases[] = {
        {0.05, 71.914193025598727}, {0.1, 56.312036502701673}};
    for (const auto& c : wide_cases) {
        const double ucl = ucl_of(32, wide, c.alpha, &st).first;
        N4M_TEST_REQUIRE(st == N4M_OK);
        N4M_TEST_REQUIRE(rel_close(ucl, c.ref, 5e-9));
    }
}

// Review b8fix5 (Hotelling checked for the same pattern): T² no longer goes
// through sigma^2 (NaN with status OK beyond |x| ~ 2^512) and is exactly
// scale-invariant; the F quantile has no fixed bracket (the former
// [1e-10, 1e12] returned about 1.1e12 for alpha = 1e-14, d1 = 1, d2 = 2, whose
// quantile is 2 (1-alpha)^2 / (alpha (2-alpha)) = 1e14 exactly in closed form,
// P(F_{1,2} > f) = 1 - sqrt(f / (f + 2))). UCL = 4/3 F for n = 3, r = 1.
void test_hotelling_t2_extremes() {
    const double base[6] = {1, 0.5, 2, -1, 4, 3};
    double t0[3] = {0}, ucl0 = -1.0;
    {
        double X[6];
        std::copy(base, base + 6, X);
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(make_rowmajor_view(X, 3, 2), 1, 0.05,
                                                            t0, 3, &ucl0) == N4M_OK);
    }
    for (const int s : {-1000, -600, -200, 200, 600, 1000}) {
        double X[6], t2[3] = {0}, ucl = -1.0;
        for (int i = 0; i < 6; ++i) X[i] = std::ldexp(base[i], s);
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(make_rowmajor_view(X, 3, 2), 1, 0.05,
                                                            t2, 3, &ucl) == N4M_OK);
        for (int i = 0; i < 3; ++i) N4M_TEST_REQUIRE(t2[i] == t0[i]);
        N4M_TEST_REQUIRE(ucl == ucl0);
    }
    for (const double alpha : {0.05, 1e-6, 1e-14, 1e-300}) {
        double X[3] = {1, 2, 3}, t2[3] = {0}, ucl = -1.0;
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(make_rowmajor_view(X, 3, 1), 1, alpha,
                                                            t2, 3, &ucl) == N4M_OK);
        const double fq = 2.0 * (1.0 - alpha) * (1.0 - alpha) / (alpha * (2.0 - alpha));
        N4M_TEST_REQUIRE(rel_close(ucl, 4.0 / 3.0 * fq, 1e-12));
        // T² = (n - 1) z_i^2 / sum z^2 for one varying column.
        N4M_TEST_REQUIRE(rel_close(t2[0], 1.0, 1e-15) && rel_close(t2[2], 1.0, 1e-15));
        N4M_TEST_REQUIRE(t2[1] == 0.0);
    }
}

// Review b8fix6 P0-2 / P1. Closed forms: P(F_{1,2} > f) = 1 - sqrt(f / (f + 2)),
// f = 2 beta^2 / (1 - beta^2) with beta = 1 - alpha, and P(F_{2,nu} > f) =
// (1 + 2 f / nu)^(-nu / 2), f = nu/2 expm1(-(2/nu) ln alpha). Near alpha = 1
// the former upper-tail inversion went through d2 / (d2 + d1 F) rounding to 1
// and stopped on that plateau: X = [-1, 2^-28, 1], alpha = 1 - 2^-30 gave
// UCL 2.96e-16 instead of 2.312964635e-18, and the central T² 6.17e-18 was
// not flagged. For alpha = 1e-308 the bracket doubling overflowed before
// reaching F = 1e308: N4M_ERR_NUMERICAL_FAILURE instead of UCL 1.333e308. A
// UCL above DBL_MAX (alpha = 7e-309: 1.9e308) is still refused.
void test_hotelling_t2_quantile_tails() {
    const double tiny_alphas[] = {0.05, 0.3, 1e-6, 1e-100, 1e-300, 1e-308};
    const double one_minus[] = {0.5, 0.1, 0x1p-30, 0x1p-40, 0x1p-53};  // beta = 1 - alpha
    // F_{1,2}: n = 3, r = 1, UCL = 4/3 F.
    for (const double alpha : tiny_alphas) {
        double X[3] = {1, 2, 3}, t2[3] = {0}, ucl = -1.0;
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(make_rowmajor_view(X, 3, 1), 1, alpha,
                                                            t2, 3, &ucl) == N4M_OK);
        const double b = 1.0 - alpha;
        N4M_TEST_REQUIRE(rel_close(ucl, 4.0 / 3.0 * (2.0 * b * b / (alpha * (1.0 + b))), 1e-12));
    }
    for (const double beta : one_minus) {
        double X[3] = {-1, 0x1p-28, 1}, t2[3] = {0}, ucl = -1.0;
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(make_rowmajor_view(X, 3, 1), 1,
                                                            1.0 - beta, t2, 3, &ucl) == N4M_OK);
        N4M_TEST_REQUIRE(rel_close(ucl, 4.0 / 3.0 * (2.0 * beta * beta / ((1.0 - beta) *
                                                                         (1.0 + beta))), 1e-12));
        if (beta == 0x1p-30) {
            N4M_TEST_REQUIRE(rel_close(ucl, 2.312964634635743e-18, 1e-12));
            N4M_TEST_REQUIRE(rel_close(t2[1], 6.167905692361981e-18, 1e-9) && t2[1] > ucl);
        }
    }
    for (const double alpha : {7e-309, 5e-324}) {
        double X[3] = {1, 2, 3}, t2[3] = {0}, ucl = -1.0;
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(make_rowmajor_view(X, 3, 1), 1, alpha,
                                                            t2, 3, &ucl) ==
                         N4M_ERR_NUMERICAL_FAILURE);
    }
    // F_{2,4}: two orthogonal centred columns, n = 6, r = 2, UCL = 35/12 F.
    const double x24[12] = {1, 1, 1, -1, 1, 0, -1, 1, -1, -1, -1, 0};
    const auto f24 = [](double log_alpha) { return 2.0 * std::expm1(-0.5 * log_alpha); };
    const auto ucl24 = [&](double alpha) {
        double X[12], t2[6] = {0}, ucl = -1.0;
        std::copy(x24, x24 + 12, X);
        std::int32_t rank = 0;
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2_resolved(
                             make_rowmajor_view(X, 6, 2), 2, alpha, t2, 6, &ucl, &rank) == N4M_OK);
        N4M_TEST_REQUIRE(rank == 2);
        return ucl;
    };
    for (const double alpha : tiny_alphas) {
        N4M_TEST_REQUIRE(rel_close(ucl24(alpha), 35.0 / 12.0 * f24(std::log(alpha)), 1e-12));
    }
    for (const double beta : one_minus) {
        N4M_TEST_REQUIRE(rel_close(ucl24(1.0 - beta), 35.0 / 12.0 * f24(std::log1p(-beta)), 1e-12));
    }
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
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rmse(c.y_true.data(), c.y_pred.data(),
                                          c.n, &v) == N4M_OK);
        check_one(v, c.exp_rmse,  c.name + "/rmse");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_mae(c.y_true.data(), c.y_pred.data(),
                                         c.n, &v) == N4M_OK);
        check_one(v, c.exp_mae,   c.name + "/mae");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_bias(c.y_true.data(), c.y_pred.data(),
                                          c.n, &v) == N4M_OK);
        check_one(v, c.exp_bias,  c.name + "/bias");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_sep(c.y_true.data(), c.y_pred.data(),
                                         c.n, &v) == N4M_OK);
        check_one(v, c.exp_sep,   c.name + "/sep");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rpd(c.y_true.data(), c.y_pred.data(),
                                         c.n, &v) == N4M_OK);
        check_one(v, c.exp_rpd,   c.name + "/rpd");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_rpiq(c.y_true.data(), c.y_pred.data(),
                                          c.n, &v) == N4M_OK);
        check_one(v, c.exp_rpiq,  c.name + "/rpiq");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_r2(c.y_true.data(), c.y_pred.data(),
                                        c.n, &v) == N4M_OK);
        check_one(v, c.exp_r2,    c.name + "/r2");
        N4M_TEST_REQUIRE(n4m_metrics_regression_metrics_nrmse(c.y_true.data(), c.y_pred.data(),
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
        N4M_TEST_REQUIRE(n4m_outlier_detection_hotelling_t2(
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
        N4M_TEST_REQUIRE(n4m_outlier_detection_q_residuals(
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
    r.run("hotelling_t2_rank_deficient",   test_hotelling_t2_rank_deficient);
    r.run("hotelling_t2_constant_columns", test_hotelling_t2_constant_columns);
    r.run("hotelling_t2_extremes",         test_hotelling_t2_extremes);
    r.run("hotelling_t2_quantile_tails",   test_hotelling_t2_quantile_tails);
    r.run("q_residuals_smoke",             test_q_residuals_smoke);
    r.run("q_residuals_scaling",           test_q_residuals_scaling);
    r.run("q_residuals_constant_columns",  test_q_residuals_constant_columns);
    r.run("q_residuals_negative_h0",       test_q_residuals_negative_h0);
    r.run("metrics_parity",                verify_metrics_parity);
    r.run("hotelling_t2_parity",           verify_hotelling_t2_parity);
    r.run("q_residuals_parity",            verify_q_residuals_parity);
}
