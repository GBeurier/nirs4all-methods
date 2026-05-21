// SPDX-License-Identifier: CECILL-2.1

#include "core/variable_importance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "fixtures/variable_importance_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

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

void fit_fixture(int& failures,
                 const ::n4m::test::fixtures::VariableImportanceFixture& fixture,
                 bool store_scores,
                 ::n4m::core::Context& ctx,
                 std::unique_ptr<::n4m::core::Model>& model,
                 n4m_matrix_view_t& X,
                 n4m_matrix_view_t& Y) {
    ::n4m::core::Config cfg;
    cfg.n_components = fixture.n_components;
    cfg.store_scores = store_scores ? 1 : 0;

    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(::n4m::core::fit_model(ctx, cfg, X, Y, model), N4M_OK);
    CHECK_NE(model.get(), nullptr);
}

void check_fixture(int& failures,
                   const ::n4m::test::fixtures::VariableImportanceFixture& fixture) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    fit_fixture(failures, fixture, true, ctx, model, X, Y);

    std::vector<double> vip;
    std::vector<double> selectivity_ratio;
    CHECK_EQ(::n4m::core::compute_vip_scores(ctx, *model, vip), N4M_OK);
    CHECK_EQ(::n4m::core::compute_selectivity_ratio(ctx, *model, X, selectivity_ratio), N4M_OK);
    check_close_values(failures, "vip", vip, fixture.vip);
    check_close_values(failures, "selectivity_ratio", selectivity_ratio, fixture.selectivity_ratio);
}

}  // namespace

TEST(variable_importance_phase3o, generated_fixture_matches_sklearn_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kVariableImportanceFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(variable_importance_phase3o, rejects_missing_scores_and_invalid_x) {
    const auto& fixture = ::n4m::test::fixtures::kVariableImportanceFixtures[0];
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    fit_fixture(failures, fixture, false, ctx, model, X, Y);

    std::vector<double> values;
    CHECK_EQ(::n4m::core::compute_vip_scores(ctx, *model, values), N4M_ERR_UNSUPPORTED);
    CHECK_EQ(::n4m::core::compute_selectivity_ratio(ctx, *model, X, values), N4M_ERR_UNSUPPORTED);

    model.reset();
    fit_fixture(failures, fixture, true, ctx, model, X, Y);

    n4m_matrix_view_t mismatched = X;
    mismatched.cols = X.cols - 1;
    CHECK_EQ(::n4m::core::compute_selectivity_ratio(ctx, *model, mismatched, values),
             N4M_ERR_SHAPE_MISMATCH);

    std::int32_t int_values[] = {1, 2, 3, 4};
    n4m_matrix_view_t int_view{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&int_view, int_values, 2, 2, N4M_DTYPE_I32), N4M_OK);
    CHECK_EQ(::n4m::core::compute_selectivity_ratio(ctx, *model, int_view, values),
             N4M_ERR_DTYPE_MISMATCH);

    std::vector<double> bad_x(fixture.X.values, fixture.X.values + fixture.X.size);
    bad_x[3] = std::numeric_limits<double>::quiet_NaN();
    n4m_matrix_view_t bad_view{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&bad_view,
                                           bad_x.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(::n4m::core::compute_selectivity_ratio(ctx, *model, bad_view, values),
             N4M_ERR_INVALID_ARGUMENT);
}
