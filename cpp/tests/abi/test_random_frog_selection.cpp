// SPDX-License-Identifier: CECILL-2.1

#include "core/random_frog_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/random_frog_selection_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

n4m_matrix_view_t matrix_view(const ::n4m::test::fixtures::MatrixRef& ref) {
    n4m_matrix_view_t view{};
    view.data = const_cast<double*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

void check_close_values(int& failures,
                        const char* label,
                        const std::vector<double>& actual,
                        const ::n4m::test::fixtures::MatrixRef& expected) {
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

void check_close_scalar(int& failures, const char* label, double actual, double expected) {
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

void check_indices(int& failures,
                   const char* label,
                   const std::vector<std::int64_t>& actual,
                   const ::n4m::test::fixtures::RandomFrogSelectionIndexRef& expected) {
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
                   const ::n4m::test::fixtures::RandomFrogSelectionFixture& fixture) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_NIPALS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::ValidationPlan plan;
    CHECK_EQ(::n4m::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             N4M_OK);

    ::n4m::core::RandomFrogSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_random_frog(ctx,
                                                    cfg,
                                                    X,
                                                    Y,
                                                    plan,
                                                    fixture.n_iterations,
                                                    fixture.initial_size,
                                                    fixture.min_size,
                                                    fixture.max_size,
                                                    fixture.top_k,
                                                    fixture.seed,
                                                    result),
             N4M_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.n_iterations, fixture.n_iterations);
    CHECK_EQ(result.initial_size, fixture.initial_size);
    CHECK_EQ(result.min_size, fixture.min_size);
    CHECK_EQ(result.max_size, fixture.max_size);
    CHECK_EQ(result.top_k, fixture.top_k);
    CHECK_EQ(result.seed, fixture.seed);
    check_close_scalar(failures, "best_rmse", result.best_rmse, fixture.best_rmse);
    check_close_values(failures, "global_scores", result.global_scores, fixture.global_scores);
    check_close_values(failures, "inclusion_frequencies", result.inclusion_frequencies, fixture.inclusion_frequencies);
    check_close_values(failures, "rmse_by_iteration", result.rmse_by_iteration, fixture.rmse_by_iteration);
    check_indices(failures, "subset_sizes", result.subset_sizes, fixture.subset_sizes);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
    check_indices(failures, "best_indices", result.best_indices, fixture.best_indices);
}

}  // namespace

TEST(random_frog_selection_phase5g, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kRandomFrogSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(random_frog_selection_phase5g, rejects_invalid_random_frog_requests) {
    const auto& fixture = ::n4m::test::fixtures::kRandomFrogSelectionFixtures[0];
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::ValidationPlan plan;
    CHECK_EQ(::n4m::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             N4M_OK);

    ::n4m::core::RandomFrogSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_random_frog(ctx,
                                                    cfg,
                                                    X,
                                                    Y,
                                                    plan,
                                                    0,
                                                    fixture.initial_size,
                                                    fixture.min_size,
                                                    fixture.max_size,
                                                    fixture.top_k,
                                                    fixture.seed,
                                                    result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_random_frog(ctx,
                                                    cfg,
                                                    X,
                                                    Y,
                                                    plan,
                                                    fixture.n_iterations,
                                                    fixture.initial_size,
                                                    cfg.n_components - 1,
                                                    fixture.max_size,
                                                    fixture.top_k,
                                                    fixture.seed,
                                                    result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_random_frog(ctx,
                                                    cfg,
                                                    X,
                                                    Y,
                                                    plan,
                                                    fixture.n_iterations,
                                                    fixture.initial_size,
                                                    fixture.min_size,
                                                    fixture.min_size - 1,
                                                    fixture.top_k,
                                                    fixture.seed,
                                                    result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_random_frog(ctx,
                                                    cfg,
                                                    X,
                                                    Y,
                                                    plan,
                                                    fixture.n_iterations,
                                                    fixture.initial_size,
                                                    fixture.min_size,
                                                    fixture.max_size,
                                                    static_cast<std::int32_t>(fixture.X.cols + 1),
                                                    fixture.seed,
                                                    result),
             N4M_ERR_INVALID_ARGUMENT);

    n4m_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::n4m::core::select_by_random_frog(ctx,
                                                    cfg,
                                                    X,
                                                    mismatched,
                                                    plan,
                                                    fixture.n_iterations,
                                                    fixture.initial_size,
                                                    fixture.min_size,
                                                    fixture.max_size,
                                                    fixture.top_k,
                                                    fixture.seed,
                                                    result),
             N4M_ERR_SHAPE_MISMATCH);
}
