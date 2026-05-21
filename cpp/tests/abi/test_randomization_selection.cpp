// SPDX-License-Identifier: CECILL-2.1

#include "core/randomization_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "fixtures/randomization_selection_fixtures.hpp"
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

void check_indices(int& failures,
                   const char* label,
                   const std::vector<std::int64_t>& actual,
                   const ::n4m::test::fixtures::RandomizationSelectionIndexRef& expected) {
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
                   const ::n4m::test::fixtures::RandomizationSelectionFixture& fixture) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_NIPALS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);

    ::n4m::core::RandomizationSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_randomization_test(ctx,
                                                           cfg,
                                                           X,
                                                           Y,
                                                           fixture.n_permutations,
                                                           fixture.randomization_seed,
                                                           fixture.alpha,
                                                           result),
             N4M_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_permutations, fixture.n_permutations);
    CHECK_EQ(result.selected_count, static_cast<std::int32_t>(fixture.selected_indices.size));
    CHECK_EQ(result.randomization_seed, fixture.randomization_seed);
    check_close_values(failures, "observed_scores", result.observed_scores, fixture.observed_scores);
    check_close_values(failures, "p_values", result.p_values, fixture.p_values);
    check_indices(failures, "exceedance_counts", result.exceedance_counts, fixture.exceedance_counts);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
}

}  // namespace

TEST(randomization_selection_phase5o, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kRandomizationSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(randomization_selection_phase5o, rejects_invalid_randomization_requests) {
    const auto& fixture = ::n4m::test::fixtures::kRandomizationSelectionFixtures[0];
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::RandomizationSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_randomization_test(ctx,
                                                           cfg,
                                                           X,
                                                           Y,
                                                           0,
                                                           fixture.randomization_seed,
                                                           fixture.alpha,
                                                           result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_randomization_test(ctx,
                                                           cfg,
                                                           X,
                                                           Y,
                                                           fixture.n_permutations,
                                                           fixture.randomization_seed,
                                                           -0.01,
                                                           result),
             N4M_ERR_INVALID_ARGUMENT);

    n4m_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::n4m::core::select_by_randomization_test(ctx,
                                                           cfg,
                                                           X,
                                                           mismatched,
                                                           fixture.n_permutations,
                                                           fixture.randomization_seed,
                                                           fixture.alpha,
                                                           result),
             N4M_ERR_SHAPE_MISMATCH);

    cfg.n_components = static_cast<std::int32_t>(fixture.X.cols + 1);
    CHECK_EQ(::n4m::core::select_by_randomization_test(ctx,
                                                           cfg,
                                                           X,
                                                           Y,
                                                           fixture.n_permutations,
                                                           fixture.randomization_seed,
                                                           fixture.alpha,
                                                           result),
             N4M_ERR_INVALID_ARGUMENT);
}

