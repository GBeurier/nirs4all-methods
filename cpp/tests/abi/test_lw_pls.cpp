// SPDX-License-Identifier: CECILL-2.1

#include "core/lw_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/lw_pls_fixtures.hpp"
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
                   const std::vector<std::int64_t>& actual,
                   const ::n4m::test::fixtures::LwPlsIndexRef& expected) {
    CHECK_EQ(actual.size(), expected.size);
    const std::size_t n = std::min(actual.size(), expected.size);
    for (std::size_t i = 0; i < n; ++i) {
        if (actual[i] != expected.values[i]) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL neighbor[%lu]: actual %lld expected %lld\n",
                         static_cast<unsigned long>(i),
                         static_cast<long long>(actual[i]),
                         static_cast<long long>(expected.values[i]));
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::n4m::test::fixtures::LwPlsFixture& fixture) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_NIPALS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    ::n4m::core::LwPlsResult result;
    CHECK_EQ(::n4m::core::fit_predict_lw_pls(ctx,
                                                 cfg,
                                                 X,
                                                 Y,
                                                 fixture.n_neighbors,
                                                 result),
             N4M_OK);
    CHECK_EQ(result.n_samples, fixture.X.rows);
    CHECK_EQ(result.n_features, fixture.X.cols);
    CHECK_EQ(result.n_targets, fixture.Y.cols);
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.n_neighbors, fixture.n_neighbors);
    check_close_values(failures, "lw_pls_predictions", result.predictions, fixture.predictions);
    check_indices(failures, result.neighbor_indices, fixture.neighbor_indices);
}

}  // namespace

TEST(lw_pls_phase4s, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kLwPlsFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(lw_pls_phase4s, rejects_invalid_neighbor_counts) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = 2;
    double x_values[] = {
        0.0, 0.1,
        0.2, 0.3,
        0.4, 0.5,
        0.6, 0.7,
    };
    double y_values[] = {
        0.0,
        0.2,
        0.4,
        0.6,
    };
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x_values, 4, 2, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y, y_values, 4, 1, N4M_DTYPE_F64), N4M_OK);

    ::n4m::core::LwPlsResult result;
    CHECK_EQ(::n4m::core::fit_predict_lw_pls(ctx, cfg, X, Y, 2, result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::fit_predict_lw_pls(ctx, cfg, X, Y, 5, result),
             N4M_ERR_INVALID_ARGUMENT);

    n4m_matrix_view_t mismatched_y = Y;
    mismatched_y.rows = 3;
    CHECK_EQ(::n4m::core::fit_predict_lw_pls(ctx, cfg, X, mismatched_y, 3, result),
             N4M_ERR_SHAPE_MISMATCH);
}
