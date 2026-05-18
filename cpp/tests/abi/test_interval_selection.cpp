// SPDX-License-Identifier: CECILL-2.1

#include "core/interval_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "fixtures/interval_selection_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

p4a_matrix_view_t matrix_view(const ::pls4all::test::fixtures::MatrixRef& ref) {
    p4a_matrix_view_t view{};
    view.data = const_cast<double*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

void check_close_values(int& failures,
                        const char* label,
                        const std::vector<double>& actual,
                        const ::pls4all::test::fixtures::MatrixRef& expected) {
    if (actual.size() != expected.size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s size: actual %lu expected %lu\n",
                     label,
                     static_cast<unsigned long>(actual.size()),
                     static_cast<unsigned long>(expected.size));
        return;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(actual[i] - expected.values[i]);
        const double scale = std::max(std::max(std::fabs(actual[i]), std::fabs(expected.values[i])), 1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %.17g expected %.17g diff %.3g\n",
                         label,
                         static_cast<unsigned long>(i),
                         actual[i],
                         expected.values[i],
                         diff);
            return;
        }
    }
}

void check_indices(int& failures,
                   const char* label,
                   const std::vector<std::int64_t>& actual,
                   const ::pls4all::test::fixtures::IntervalSelectionIndexRef& expected) {
    if (actual.size() != expected.size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s size: actual %lu expected %lu\n",
                     label,
                     static_cast<unsigned long>(actual.size()),
                     static_cast<unsigned long>(expected.size));
        return;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected.values[i]) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %lld expected %lld\n",
                         label,
                         static_cast<unsigned long>(i),
                         static_cast<long long>(actual[i]),
                         static_cast<long long>(expected.values[i]));
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::IntervalSelectionFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_NIPALS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             P4A_OK);

    ::pls4all::core::IntervalSelectionResult result;
    CHECK_EQ(::pls4all::core::cross_validate_intervals(ctx,
                                                       cfg,
                                                       X,
                                                       Y,
                                                       plan,
                                                       fixture.interval_width,
                                                       fixture.step,
                                                       result),
             P4A_OK);
    CHECK_EQ(result.interval_width, fixture.interval_width);
    CHECK_EQ(result.step, fixture.step);
    CHECK_EQ(result.best_interval_index, fixture.best_index);
    CHECK_EQ(result.best_start, fixture.best_interval.values[0]);
    CHECK_EQ(result.best_length, fixture.best_interval.values[1]);
    check_indices(failures, "intervals", result.intervals, fixture.intervals);
    check_close_values(failures, "rmse", result.rmse, fixture.rmse);
    check_close_values(failures,
                       "metrics_by_interval",
                       result.metrics_matrix,
                       fixture.metrics_by_interval);
}

}  // namespace

TEST(interval_selection_phase5b, generated_fixture_matches_sklearn_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kIntervalSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(interval_selection_phase5b, rejects_invalid_interval_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kIntervalSelectionFixtures[0];
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             P4A_OK);

    ::pls4all::core::IntervalSelectionResult result;
    CHECK_EQ(::pls4all::core::cross_validate_intervals(ctx, cfg, X, Y, plan, 0, 1, result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::cross_validate_intervals(ctx,
                                                       cfg,
                                                       X,
                                                       Y,
                                                       plan,
                                                       static_cast<std::int32_t>(fixture.X.cols + 1),
                                                       1,
                                                       result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::cross_validate_intervals(ctx,
                                                       cfg,
                                                       X,
                                                       Y,
                                                       plan,
                                                       fixture.interval_width,
                                                       0,
                                                       result),
             P4A_ERR_INVALID_ARGUMENT);

    cfg.n_components = fixture.interval_width + 1;
    CHECK_EQ(::pls4all::core::cross_validate_intervals(ctx,
                                                       cfg,
                                                       X,
                                                       Y,
                                                       plan,
                                                       fixture.interval_width,
                                                       fixture.step,
                                                       result),
             P4A_ERR_INVALID_ARGUMENT);

    cfg.n_components = fixture.n_components;
    ::pls4all::core::ValidationPlan empty;
    empty.n_samples = fixture.X.rows;
    CHECK_EQ(::pls4all::core::cross_validate_intervals(ctx,
                                                       cfg,
                                                       X,
                                                       Y,
                                                       empty,
                                                       fixture.interval_width,
                                                       fixture.step,
                                                       result),
             P4A_ERR_INVALID_ARGUMENT);
}
