// SPDX-License-Identifier: CeCILL-2.1

#include "core/wvc_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/wvc_selection_fixtures.hpp"
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
                   const ::pls4all::test::fixtures::WvcSelectionIndexRef& expected) {
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
                   const ::pls4all::test::fixtures::WvcSelectionFixture& fixture) {
    ::pls4all::core::Context ctx;
    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);

    ::pls4all::core::WvcSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            fixture.n_components,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             P4A_OK);
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

}  // namespace

TEST(wvc_selection_phase5m, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kWvcSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(wvc_selection_phase5m, rejects_invalid_wvc_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kWvcSelectionFixtures[0];
    ::pls4all::core::Context ctx;
    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::WvcSelectionResult result;

    CHECK_EQ(::pls4all::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            0,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            fixture.n_components,
                                            0,
                                            fixture.normalize,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_wvc(ctx,
                                            X,
                                            Y,
                                            static_cast<std::int32_t>(fixture.X.cols),
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t bad_y = Y;
    bad_y.cols = 2;
    CHECK_EQ(::pls4all::core::select_by_wvc(ctx,
                                            X,
                                            bad_y,
                                            fixture.n_components,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_wvc(ctx,
                                            X,
                                            mismatched,
                                            fixture.n_components,
                                            fixture.top_k,
                                            fixture.normalize,
                                            result),
             P4A_ERR_SHAPE_MISMATCH);
}
