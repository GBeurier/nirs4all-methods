// SPDX-License-Identifier: CECILL-2.1

#include "core/spa_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "fixtures/spa_selection_fixtures.hpp"
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
                   const ::n4m::test::fixtures::SpaSelectionIndexRef& expected) {
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
                   const ::n4m::test::fixtures::SpaSelectionFixture& fixture) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_NIPALS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::SpaSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_spa(ctx, cfg, X, Y, fixture.top_k, result), N4M_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.top_k, fixture.top_k);
    check_close_values(failures, "coefficient_scores", result.coefficient_scores, fixture.coefficient_scores);
    check_close_values(failures, "selection_scores", result.selection_scores, fixture.selection_scores);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
}

}  // namespace

TEST(spa_selection_phase5e, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kSpaSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(spa_selection_phase5e, rejects_invalid_spa_requests) {
    const auto& fixture = ::n4m::test::fixtures::kSpaSelectionFixtures[0];
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::SpaSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_spa(ctx, cfg, X, Y, 0, result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_spa(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            static_cast<std::int32_t>(fixture.X.cols + 1),
                                            result),
             N4M_ERR_INVALID_ARGUMENT);

    n4m_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::n4m::core::select_by_spa(ctx,
                                            cfg,
                                            X,
                                            mismatched,
                                            fixture.top_k,
                                            result),
             N4M_ERR_SHAPE_MISMATCH);
}
