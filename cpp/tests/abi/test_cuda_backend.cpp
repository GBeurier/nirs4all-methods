// SPDX-License-Identifier: CECILL-2.1
//
// Phase 45 regression: confirm the cuBLAS dispatch path produces
// results numerically equivalent to the reference scalar loops on
// the same shared fixture every other binding consumes.
//
// When P4A_USE_CUDA is defined the cuBLAS dispatch is exercised.
// When it is not, the tests still compile and the parity assertion
// hits the scalar / BLAS path. The smoke and availability tests
// gracefully report whichever runtime state the host produces.
//
// Tolerance: rmse_rel <= 1e-10. cuBLAS reorders FP ops in its tiled
// kernels, so we use the same relaxed gate as the BLAS regression
// (the cross-binding gate stays at 1e-12).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pls4all/p4a.h"
#include "harness.hpp"

#if defined(P4A_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif

namespace {

constexpr std::int32_t kN = 50;
constexpr std::int32_t kP = 5;
constexpr std::int32_t kQ = 1;
constexpr std::int32_t kComponents = 3;

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

#if defined(P4A_USE_CUDA)
bool cuda_runtime_active() {
    return ::pls4all::cuda_dispatch::cuda_runtime_available();
}
#endif

}  // namespace

TEST(cuda_backend, p4a_backend_is_available_reports_compile_and_runtime_state) {
    const int reported = p4a_backend_is_available(P4A_BACKEND_CUDA);
#if defined(P4A_USE_CUDA)
    // Compile-time support present; runtime result depends on whether
    // the host has a working GPU.
    CHECK(reported == 0 || reported == 1);
#else
    CHECK_EQ(reported, 0);
#endif
}

TEST(cuda_backend, p4a_pls_fit_simple_matches_fixture) {
#if defined(P4A_USE_CUDA)
    if (!cuda_runtime_active()) {
        // No GPU at runtime — accept trivial pass so OFF / no-device
        // CI runners stay green.
        CHECK(true);
        return;
    }
#endif
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

TEST(cuda_backend, larger_matrix_predictions_remain_finite) {
#if defined(P4A_USE_CUDA)
    if (!cuda_runtime_active()) {
        CHECK(true);
        return;
    }
#endif
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
