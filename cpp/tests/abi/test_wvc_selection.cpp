// SPDX-License-Identifier: CECILL-2.1

#include "core/wvc_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/wvc_threshold_selection_fixtures.hpp"
#include "fixtures/wvc_selection_fixtures.hpp"
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

template <typename IndexRef>
void check_indices(int& failures,
                   const char* label,
                   const std::vector<std::int64_t>& actual,
                   const IndexRef& expected) {
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

void check_scalar(int& failures, const char* label, double actual, double expected) {
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
                   const ::n4m::test::fixtures::WvcSelectionFixture& fixture) {
    ::n4m::core::Context ctx;
    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);

    ::n4m::core::WvcSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            fixture.n_components,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             N4M_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.top_k, fixture.top_k);
    CHECK_EQ(result.normalize, fixture.normalize ? 1 : 0);
    check_close_values(failures, "weights_w", result.weights_w, fixture.weights_w);
    check_close_values(failures, "loadings_p", result.loadings_p, fixture.loadings_p);
    check_close_values(failures, "y_loadings_q", result.y_loadings_q, fixture.y_loadings_q);
    check_close_values(failures, "wvc_scores", result.wvc_scores, fixture.wvc_scores);
    check_close_values(failures, "final_scores", result.final_scores, fixture.final_scores);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
}

void check_threshold_fixture(
    int& failures,
    const ::n4m::test::fixtures::WvcThresholdSelectionFixture& fixture) {
    ::n4m::core::Context ctx;
    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);

    ::n4m::core::WvcThresholdSelectionResult result;
    CHECK_EQ(::n4m::core::select_by_wvc_threshold(ctx,
                                                      X,
                                                      Y,
                                                      fixture.n_components,
                                                      fixture.normalize,
                                                      fixture.score_threshold,
                                                      fixture.threshold_factor,
                                                      fixture.min_selected,
                                                      result),
             N4M_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.min_selected, fixture.min_selected);
    CHECK_EQ(result.normalize, fixture.normalize ? 1 : 0);
    check_scalar(failures, "score_threshold", result.score_threshold, fixture.score_threshold);
    check_scalar(failures, "threshold_factor", result.threshold_factor, fixture.threshold_factor);
    check_scalar(failures, "mean_score", result.mean_score, fixture.mean_score);
    check_scalar(failures, "effective_threshold", result.effective_threshold, fixture.effective_threshold);
    check_close_values(failures, "threshold_final_scores", result.final_scores, fixture.final_scores);
    check_indices(failures, "threshold_ranked_indices", result.ranked_indices, fixture.ranked_indices);
    check_indices(failures, "threshold_selected_indices", result.selected_indices, fixture.selected_indices);
}

}  // namespace

TEST(wvc_selection_phase5m, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kWvcSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(wvc_selection_phase5m, rejects_invalid_wvc_requests) {
    const auto& fixture = ::n4m::test::fixtures::kWvcSelectionFixtures[0];
    ::n4m::core::Context ctx;
    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::WvcSelectionResult result;

    CHECK_EQ(::n4m::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            0,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            fixture.n_components,
                                            0,
                                            fixture.normalize,
                                            result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            static_cast<std::int32_t>(fixture.X.cols),
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             N4M_ERR_INVALID_ARGUMENT);

    n4m_matrix_view_t bad_y = Y;
    bad_y.cols = 2;
    CHECK_EQ(::n4m::core::select_by_wvc(ctx,
                                            X,
                                            bad_y,
                                            fixture.n_components,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             N4M_ERR_INVALID_ARGUMENT);

    n4m_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::n4m::core::select_by_wvc(ctx,
                                            X,
                                            mismatched,
                                            fixture.n_components,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             N4M_ERR_SHAPE_MISMATCH);
}

TEST(wvc_threshold_selection_phase5r, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kWvcThresholdSelectionFixtures) {
        check_threshold_fixture(failures, fixture);
    }
}

TEST(wvc_threshold_selection_phase5r, rejects_invalid_threshold_requests) {
    const auto& fixture = ::n4m::test::fixtures::kWvcThresholdSelectionFixtures[0];
    ::n4m::core::Context ctx;
    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::WvcThresholdSelectionResult result;

    CHECK_EQ(::n4m::core::select_by_wvc_threshold(ctx,
                                                      X,
                                                      Y,
                                                      0,
                                                      fixture.normalize,
                                                      fixture.score_threshold,
                                                      fixture.threshold_factor,
                                                      fixture.min_selected,
                                                      result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_wvc_threshold(ctx,
                                                      X,
                                                      Y,
                                                      fixture.n_components,
                                                      fixture.normalize,
                                                      -0.1,
                                                      fixture.threshold_factor,
                                                      fixture.min_selected,
                                                      result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_wvc_threshold(ctx,
                                                      X,
                                                      Y,
                                                      fixture.n_components,
                                                      fixture.normalize,
                                                      fixture.score_threshold,
                                                      -0.1,
                                                      fixture.min_selected,
                                                      result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::select_by_wvc_threshold(ctx,
                                                      X,
                                                      Y,
                                                      fixture.n_components,
                                                      fixture.normalize,
                                                      fixture.score_threshold,
                                                      fixture.threshold_factor,
                                                      0,
                                                      result),
             N4M_ERR_INVALID_ARGUMENT);
}
