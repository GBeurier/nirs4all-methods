// SPDX-License-Identifier: CeCILL-2.1

#include "core/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "fixtures/metrics_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-12;
constexpr double kRelTol = 1e-12;

void check_metric(int& failures,
                  const char* label,
                  double actual,
                  double expected) {
    const double diff = std::fabs(actual - expected);
    const double scale = std::max(std::max(std::fabs(actual), std::fabs(expected)), 1.0);
    if (diff > kAbsTol && diff > kRelTol * scale) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s: actual %.17g expected %.17g diff %.3g\n",
                     label,
                     actual,
                     expected,
                     diff);
    }
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::MetricsFixture& fixture) {
    p4a_context_t* raw_ctx = nullptr;
    CHECK_EQ(p4a_context_create(&raw_ctx), P4A_OK);
    auto* ctx = static_cast<::pls4all::core::Context*>(raw_ctx);

    p4a_matrix_view_t observed{};
    p4a_matrix_view_t predicted{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&observed,
                                           const_cast<double*>(fixture.observed.values),
                                           fixture.observed.rows,
                                           fixture.observed.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&predicted,
                                           const_cast<double*>(fixture.predicted.values),
                                           fixture.predicted.rows,
                                           fixture.predicted.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    ::pls4all::core::RegressionMetrics metrics{};
    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, observed, predicted, metrics), P4A_OK);
    CHECK_EQ(metrics.rows, fixture.observed.rows);
    CHECK_EQ(metrics.cols, fixture.observed.cols);
    CHECK_EQ(metrics.count, static_cast<std::int64_t>(fixture.observed.size));
    CHECK_EQ(fixture.expected.size, static_cast<std::size_t>(9U));

    const double* expected = fixture.expected.values;
    check_metric(failures, "rmse", metrics.rmse, expected[0]);
    check_metric(failures, "mae", metrics.mae, expected[1]);
    check_metric(failures, "bias", metrics.bias, expected[2]);
    check_metric(failures, "r2", metrics.r2, expected[3]);
    check_metric(failures, "q2", metrics.q2, expected[4]);
    check_metric(failures, "slope", metrics.slope, expected[5]);
    check_metric(failures, "intercept", metrics.intercept, expected[6]);
    check_metric(failures, "rpd", metrics.rpd, expected[7]);
    check_metric(failures, "rpiq", metrics.rpiq, expected[8]);

    p4a_context_destroy(raw_ctx);
}

}  // namespace

TEST(metrics_phase3k, generated_regression_fixture_matches_numpy_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kMetricsFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(metrics_phase3k, rejects_shape_dtype_empty_and_nonfinite_inputs) {
    p4a_context_t* raw_ctx = nullptr;
    CHECK_EQ(p4a_context_create(&raw_ctx), P4A_OK);
    auto* ctx = static_cast<::pls4all::core::Context*>(raw_ctx);

    double observed_values[] = {1.0, 2.0, 3.0, 4.0};
    double predicted_values[] = {1.1, 1.9, 3.2, 3.8};
    p4a_matrix_view_t observed{};
    p4a_matrix_view_t predicted{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&observed, observed_values, 2, 2, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&predicted, predicted_values, 2, 2, P4A_DTYPE_F64), P4A_OK);

    ::pls4all::core::RegressionMetrics metrics{};
    p4a_matrix_view_t mismatched = predicted;
    mismatched.cols = 1;
    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, observed, mismatched, metrics),
             P4A_ERR_SHAPE_MISMATCH);

    std::int32_t int_values[] = {1, 2, 3, 4};
    p4a_matrix_view_t int_view{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&int_view, int_values, 2, 2, P4A_DTYPE_I32), P4A_OK);
    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, int_view, predicted, metrics),
             P4A_ERR_DTYPE_MISMATCH);

    p4a_matrix_view_t empty{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&empty, nullptr, 0, 2, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, empty, predicted, metrics),
             P4A_ERR_INVALID_ARGUMENT);

    double bad_values[] = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0};
    p4a_matrix_view_t bad{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&bad, bad_values, 2, 2, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, bad, predicted, metrics),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_context_destroy(raw_ctx);
}

TEST(metrics_phase3k, handles_constant_observed_values_deterministically) {
    p4a_context_t* raw_ctx = nullptr;
    CHECK_EQ(p4a_context_create(&raw_ctx), P4A_OK);
    auto* ctx = static_cast<::pls4all::core::Context*>(raw_ctx);

    double observed_values[] = {2.0, 2.0, 2.0, 2.0};
    double perfect_values[] = {2.0, 2.0, 2.0, 2.0};
    double imperfect_values[] = {1.5, 2.0, 2.5, 3.0};
    p4a_matrix_view_t observed{};
    p4a_matrix_view_t perfect{};
    p4a_matrix_view_t imperfect{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&observed, observed_values, 4, 1, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&perfect, perfect_values, 4, 1, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&imperfect, imperfect_values, 4, 1, P4A_DTYPE_F64), P4A_OK);

    ::pls4all::core::RegressionMetrics metrics{};
    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, observed, perfect, metrics), P4A_OK);
    CHECK_EQ(metrics.r2, 1.0);
    CHECK_EQ(metrics.q2, 1.0);
    CHECK_EQ(metrics.slope, 0.0);
    CHECK_EQ(metrics.intercept, 2.0);

    CHECK_EQ(::pls4all::core::compute_regression_metrics(*ctx, observed, imperfect, metrics), P4A_OK);
    CHECK_EQ(metrics.r2, 0.0);
    CHECK_EQ(metrics.q2, 0.0);
    CHECK_EQ(metrics.intercept, 2.0);

    p4a_context_destroy(raw_ctx);
}
