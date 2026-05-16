// SPDX-License-Identifier: CeCILL-2.1
//
// Phase 44 regression: confirm the OMP-annotated deflation, loading,
// and fill_prediction paths produce results numerically equivalent to
// the reference scalar loops.
//
// Same shared fixture as test_blas_backend.cpp (n=50, p=5, q=1, k=3).
// OMP thread scheduling does not change FP summation order *inside*
// each annotated loop iteration — the parallel dimension is over
// independent rows (deflation, fill_prediction) or independent
// features (x_loadings), with the reductions kept thread-local. So
// the rmse_rel against the fixture stays at machine epsilon even
// with multiple threads.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pls4all/p4a.h"
#include "harness.hpp"

namespace {

constexpr std::int32_t kN = 50;
constexpr std::int32_t kP = 5;
constexpr std::int32_t kQ = 1;
constexpr std::int32_t kComponents = 3;

// Fixture constants from bindings/js/test/parity_fixture.json.
constexpr double kExpectedCoefs[] = {
    1.0000027187515323,
    0.499994268955383,
    -0.30000061454539234,
    -3.202142436496356e-06,
    -7.901299650182916e-06,
};
constexpr double kExpectedPreds5[] = {
    0.34282942523393906,
    0.738499878972995,
    1.1420420223581373,
    1.4025309763869322,
    1.3613098478826045,
};
constexpr double kRmseRelTol = 1e-10;

void build_inputs(std::vector<double>& X, std::vector<double>& Y) {
    X.assign(static_cast<std::size_t>(kN * kP), 0.0);
    Y.assign(static_cast<std::size_t>(kN * kQ), 0.0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(kP); ++j) {
            X[i * kP + j] = std::sin(
                static_cast<double>((i + 1U) * (j + 1U)) * 0.3);
        }
        Y[i * kQ + 0] = X[i * kP + 0]
                         + 0.5 * X[i * kP + 1]
                         - 0.3 * X[i * kP + 2];
    }
}

double rmse_rel(const double* actual, const double* expected, std::size_t n) {
    double diff = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = actual[i] - expected[i];
        diff += d * d;
        ref += expected[i] * expected[i];
    }
    const double denom = std::sqrt(ref);
    return std::sqrt(diff) /
           (denom > std::numeric_limits<double>::min()
                ? denom
                : std::numeric_limits<double>::min());
}

}  // namespace

TEST(openmp_backend, p4a_backend_is_available_reports_compile_time_state) {
    const int reported = p4a_backend_is_available(P4A_BACKEND_OPENMP);
#if defined(P4A_USE_OPENMP)
    CHECK_EQ(reported, 1);
#else
    CHECK_EQ(reported, 0);
#endif
}

TEST(openmp_backend, p4a_pls_fit_simple_matches_fixture_within_omp_tolerance) {
    std::vector<double> X;
    std::vector<double> Y;
    build_inputs(X, Y);

    std::vector<double> coefs(static_cast<std::size_t>(kP * kQ), 0.0);
    std::vector<double> x_mean(static_cast<std::size_t>(kP), 0.0);
    std::vector<double> y_mean(static_cast<std::size_t>(kQ), 0.0);
    std::vector<double> preds(static_cast<std::size_t>(kN * kQ), 0.0);

    const p4a_status_t status = p4a_pls_fit_simple(
        X.data(), Y.data(),
        kN, kP, kQ, kComponents,
        coefs.data(), x_mean.data(), y_mean.data(), preds.data());
    CHECK_EQ(status, P4A_OK);

    CHECK(rmse_rel(coefs.data(), kExpectedCoefs,
                    static_cast<std::size_t>(kP * kQ)) <= kRmseRelTol);
    CHECK(rmse_rel(preds.data(), kExpectedPreds5, 5U) <= kRmseRelTol);
}

TEST(openmp_backend, larger_matrix_predictions_remain_finite) {
    // 200x20x2, k=8: exercises the fill_prediction / fill_transform
    // and deflation OMP paths at a realistic size.
    constexpr std::int32_t n = 200;
    constexpr std::int32_t p = 20;
    constexpr std::int32_t q = 2;
    constexpr std::int32_t k = 8;

    std::vector<double> X(static_cast<std::size_t>(n * p));
    std::vector<double> Y(static_cast<std::size_t>(n * q));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(p); ++j) {
            X[i * p + j] =
                std::sin(static_cast<double>((i + 1U) * (j + 1U)) * 0.117)
                + 0.3 * std::cos(static_cast<double>((i + 2U) * (j + 3U)) * 0.041);
        }
        Y[i * q + 0] = X[i * p + 0] + 0.5 * X[i * p + 1] - 0.2 * X[i * p + 2];
        Y[i * q + 1] = X[i * p + 3] - 0.4 * X[i * p + 4] + 0.1 * X[i * p + 5];
    }

    std::vector<double> coefs(static_cast<std::size_t>(p * q), 0.0);
    std::vector<double> x_mean(static_cast<std::size_t>(p), 0.0);
    std::vector<double> y_mean(static_cast<std::size_t>(q), 0.0);
    std::vector<double> preds(static_cast<std::size_t>(n * q), 0.0);

    CHECK_EQ(p4a_pls_fit_simple(X.data(), Y.data(),
                                  n, p, q, k,
                                  coefs.data(), x_mean.data(),
                                  y_mean.data(), preds.data()),
              P4A_OK);
    for (double v : coefs) { CHECK(std::isfinite(v)); }
    for (double v : preds) { CHECK(std::isfinite(v)); }
}
