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

void fit_fixture(int& failures,
                 const ::pls4all::test::fixtures::VariableImportanceFixture& fixture,
                 bool store_scores,
                 ::pls4all::core::Context& ctx,
                 std::unique_ptr<::pls4all::core::Model>& model,
                 p4a_matrix_view_t& X,
                 p4a_matrix_view_t& Y) {
    ::pls4all::core::Config cfg;
    cfg.n_components = fixture.n_components;
    cfg.store_scores = store_scores ? 1 : 0;

    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(::pls4all::core::fit_model(ctx, cfg, X, Y, model), P4A_OK);
    CHECK_NE(model.get(), nullptr);
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::VariableImportanceFixture& fixture) {
    ::pls4all::core::Context ctx;
    std::unique_ptr<::pls4all::core::Model> model;
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    fit_fixture(failures, fixture, true, ctx, model, X, Y);

    std::vector<double> vip;
    std::vector<double> selectivity_ratio;
    CHECK_EQ(::pls4all::core::compute_vip_scores(ctx, *model, vip), P4A_OK);
    CHECK_EQ(::pls4all::core::compute_selectivity_ratio(ctx, *model, X, selectivity_ratio), P4A_OK);
    check_close_values(failures, "vip", vip, fixture.vip);
    check_close_values(failures, "selectivity_ratio", selectivity_ratio, fixture.selectivity_ratio);
}

}  // namespace

TEST(variable_importance_phase3o, generated_fixture_matches_sklearn_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kVariableImportanceFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(variable_importance_phase3o, rejects_missing_scores_and_invalid_x) {
    const auto& fixture = ::pls4all::test::fixtures::kVariableImportanceFixtures[0];
    ::pls4all::core::Context ctx;
    std::unique_ptr<::pls4all::core::Model> model;
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    fit_fixture(failures, fixture, false, ctx, model, X, Y);

    std::vector<double> values;
    CHECK_EQ(::pls4all::core::compute_vip_scores(ctx, *model, values), P4A_ERR_UNSUPPORTED);
    CHECK_EQ(::pls4all::core::compute_selectivity_ratio(ctx, *model, X, values), P4A_ERR_UNSUPPORTED);

    model.reset();
    fit_fixture(failures, fixture, true, ctx, model, X, Y);

    p4a_matrix_view_t mismatched = X;
    mismatched.cols = X.cols - 1;
    CHECK_EQ(::pls4all::core::compute_selectivity_ratio(ctx, *model, mismatched, values),
             P4A_ERR_SHAPE_MISMATCH);

    std::int32_t int_values[] = {1, 2, 3, 4};
    p4a_matrix_view_t int_view{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&int_view, int_values, 2, 2, P4A_DTYPE_I32), P4A_OK);
    CHECK_EQ(::pls4all::core::compute_selectivity_ratio(ctx, *model, int_view, values),
             P4A_ERR_DTYPE_MISMATCH);

    std::vector<double> bad_x(fixture.X.values, fixture.X.values + fixture.X.size);
    bad_x[3] = std::numeric_limits<double>::quiet_NaN();
    p4a_matrix_view_t bad_view{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&bad_view,
                                           bad_x.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(::pls4all::core::compute_selectivity_ratio(ctx, *model, bad_view, values),
             P4A_ERR_INVALID_ARGUMENT);
}
