// SPDX-License-Identifier: CeCILL-2.1

#include "core/model_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fixtures/component_cv_fixtures.hpp"
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

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::ComponentCvFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    if (std::strcmp(fixture.solver, "simpls") == 0) {
        cfg.solver = P4A_SOLVER_SIMPLS;
    } else {
        CHECK(false);
        return;
    }

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             P4A_OK);

    ::pls4all::core::ComponentCvResult result;
    CHECK_EQ(::pls4all::core::cross_validate_component_prefixes(ctx,
                                                                cfg,
                                                                X,
                                                                Y,
                                                                plan,
                                                                fixture.n_components_max,
                                                                result),
             P4A_OK);
    CHECK_EQ(result.max_components, fixture.n_components_max);
    CHECK_EQ(result.best_n_components, fixture.best_n_components);
    CHECK_EQ(result.metrics_by_component.size(),
             static_cast<std::size_t>(fixture.n_components_max));
    check_close_values(failures,
                       "component_cv_metrics",
                       result.metrics_matrix,
                       fixture.metrics_by_component);
}

}  // namespace

TEST(model_selection_phase4o, generated_component_cv_fixture_matches_numpy_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kComponentCvFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(model_selection_phase4o, rejects_invalid_component_cv_requests) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = 1;
    double x_values[] = {
        0.0, 0.2,
        0.4, 0.7,
        0.9, 1.1,
        1.3, 1.4,
    };
    double y_values[] = {0.1, 0.5, 1.0, 1.2};
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x_values, 4, 2, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y, y_values, 4, 1, P4A_DTYPE_F64), P4A_OK);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx, 4, 2, plan), P4A_OK);

    ::pls4all::core::ComponentCvResult result;
    CHECK_EQ(::pls4all::core::cross_validate_component_prefixes(ctx,
                                                                cfg,
                                                                X,
                                                                Y,
                                                                plan,
                                                                0,
                                                                result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::cross_validate_component_prefixes(ctx,
                                                                cfg,
                                                                X,
                                                                Y,
                                                                plan,
                                                                3,
                                                                result),
             P4A_ERR_INVALID_ARGUMENT);

    ::pls4all::core::ValidationPlan empty;
    empty.n_samples = 4;
    CHECK_EQ(::pls4all::core::cross_validate_component_prefixes(ctx,
                                                                cfg,
                                                                X,
                                                                Y,
                                                                empty,
                                                                1,
                                                                result),
             P4A_ERR_INVALID_ARGUMENT);
}

TEST(model_selection_phase10, one_se_rule_picks_smaller_model_when_within_se) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_SIMPLS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = 1;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;
    cfg.store_scores = 0;

    // 12-sample, 4-feature dataset designed so the marginal benefit of
    // additional components is small — the one-SE rule should pick the
    // smallest competitive component count.
    double x_values[] = {
        0.10, 0.45, 0.95, 0.20,
        0.30, 0.40, 0.80, 0.65,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.55,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
        0.15, 0.55, 0.85, 0.25,
        0.35, 0.45, 0.75, 0.62,
        0.65, 0.15, 0.25, 0.75,
        0.45, 0.35, 0.65, 0.92,
    };
    double y_values[] = {
        1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3,
        1.05, 0.72, 0.45, 1.02,
    };
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x_values, 12, 4, P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y, y_values, 12, 1, P4A_DTYPE_F64),
             P4A_OK);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx, 12, 4, plan),
             P4A_OK);

    ::pls4all::core::ComponentCvResult result;
    CHECK_EQ(::pls4all::core::cross_validate_component_prefixes(ctx, cfg, X, Y,
                                                                plan, 3, result),
             P4A_OK);
    CHECK_EQ(result.fold_rmse_matrix.size(),
             static_cast<std::size_t>(3 * 4));
    CHECK_EQ(result.n_folds, 4);

    // The one-SE rule should default to best_n_components before being
    // applied.
    CHECK_EQ(result.one_se_n_components, result.best_n_components);

    CHECK_EQ(::pls4all::core::select_one_se_components(ctx, result), P4A_OK);
    CHECK(result.one_se_standard_error >= 0.0);
    CHECK(result.one_se_threshold >=
          result.metrics_by_component[
              static_cast<std::size_t>(result.best_n_components - 1)].rmse);
    CHECK(result.one_se_n_components >= 1);
    CHECK(result.one_se_n_components <= result.best_n_components);
}

TEST(model_selection_phase10, one_se_rule_rejects_unpopulated_result) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::ComponentCvResult result;
    CHECK_EQ(::pls4all::core::select_one_se_components(ctx, result),
             P4A_ERR_INVALID_ARGUMENT);

    result.max_components = 3;
    result.n_folds = 1;
    CHECK_EQ(::pls4all::core::select_one_se_components(ctx, result),
             P4A_ERR_INVALID_ARGUMENT);
}
