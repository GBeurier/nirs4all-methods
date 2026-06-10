// SPDX-License-Identifier: CECILL-2.1
//
// Internal-core parity tests for the cross-validation / model-selection
// families that have NO public C ABI surface (they are exercised only through
// the internal core: cross_validate_regression, cross_validate_component_
// prefixes, compute_regression_coefficients_by_component). Like the UVE test,
// these symbols are hidden in libn4m, so this TU is linked into
// n4m_internal_tests (static archive) rather than n4m_tests.
//
// Families covered:
//   - cross-validation        (synthetic_cv_kfold_nipals_pls{1,2})
//   - component-cv            (synthetic_component_cv_simpls_pls2)
//   - component-coefficients  (synthetic_component_coefficients_pls2)
//
// Per parity/tolerances.md these gate at abs/rel 1e-8
// (`sklearn/PLSRegression/kfold-cv`, `pls4all-numpy-simpls-component-cv`,
//  `sklearn/PLSRegression/component-coefficients`).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "core/common/context.hpp"
#include "core/component_coefficients.hpp"
#include "core/config.hpp"
#include "core/cross_validation.hpp"
#include "core/model.hpp"
#include "core/model_selection.hpp"
#include "core/validation.hpp"

#include "fixtures/component_coefficients_fixtures.hpp"
#include "fixtures/component_cv_fixtures.hpp"
#include "fixtures/cross_validation_fixtures.hpp"

namespace {

using ::n4m::test::fixtures::MatrixRef;

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

int g_cv_failures = 0;

#define CV_CHECK(cond)                                                       \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
            ++g_cv_failures;                                                 \
        }                                                                    \
    } while (0)

n4m_matrix_view_t make_view(const MatrixRef& ref) {
    n4m_matrix_view_t v{};
    v.data = const_cast<double*>(ref.values);
    v.rows = ref.rows;
    v.cols = ref.cols;
    v.row_stride = ref.cols > 0 ? ref.cols : 1;
    v.col_stride = 1;
    v.dtype = N4M_DTYPE_F64;
    return v;
}

bool close(const std::vector<double>& actual, const MatrixRef& expected, const char* tag) {
    if (actual.size() != expected.size) {
        std::printf("  FAIL %s size: actual %zu expected %zu\n",
                    tag, actual.size(), expected.size);
        return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double diff  = std::fabs(actual[i] - expected.values[i]);
        const double scale = std::max(std::max(std::fabs(actual[i]),
                                               std::fabs(expected.values[i])), 1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            std::printf("  FAIL %s[%zu]: actual %.17g expected %.17g diff %.3g\n",
                        tag, i, actual[i], expected.values[i], diff);
            return false;
        }
    }
    return true;
}

template <typename IndexRef>
bool close_index(const std::vector<std::int64_t>& actual, const IndexRef& expected,
                 const char* tag) {
    if (actual.size() != expected.size) {
        std::printf("  FAIL %s size: actual %zu expected %zu\n",
                    tag, actual.size(), expected.size);
        return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected.values[i]) {
            std::printf("  FAIL %s[%zu]: actual %lld expected %lld\n",
                        tag, i, static_cast<long long>(actual[i]),
                        static_cast<long long>(expected.values[i]));
            return false;
        }
    }
    return true;
}

void test_cross_validation() {
    for (const auto& fx : ::n4m::test::fixtures::kCrossValidationFixtures) {
        ::n4m::core::Context ctx;
        ::n4m::core::Config cfg;
        cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
        cfg.solver = N4M_SOLVER_NIPALS;
        cfg.deflation = N4M_DEFLATION_REGRESSION;
        cfg.n_components = fx.n_components;

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_matrix_view_t Y = make_view(fx.Y);

        ::n4m::core::ValidationPlan plan;
        CV_CHECK(::n4m::core::make_kfold_validation_plan(ctx, fx.X.rows, fx.n_splits, plan) == N4M_OK);

        ::n4m::core::CrossValidationResult result;
        CV_CHECK(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, plan, result) == N4M_OK);
        CV_CHECK(result.n_samples == fx.X.rows);
        CV_CHECK(result.n_targets == fx.Y.cols);
        CV_CHECK(result.n_folds == static_cast<std::int64_t>(fx.n_splits));

        CV_CHECK(close(result.predictions, fx.predictions, "cv_predictions"));

        const double metrics[] = {result.metrics.rmse, result.metrics.mae, result.metrics.bias,
                                  result.metrics.r2, result.metrics.q2, result.metrics.slope,
                                  result.metrics.intercept, result.metrics.rpd, result.metrics.rpiq};
        const std::vector<double> metrics_vec(metrics, metrics + 9);
        CV_CHECK(fx.metrics.size == 9U);
        CV_CHECK(close(metrics_vec, fx.metrics, "cv_metrics"));

        CV_CHECK(close_index(result.test_offsets, fx.test_offsets, "cv_test_offsets"));
        CV_CHECK(close_index(result.test_indices, fx.test_indices, "cv_test_indices"));
    }
}

void test_component_cv() {
    for (const auto& fx : ::n4m::test::fixtures::kComponentCvFixtures) {
        ::n4m::core::Context ctx;
        ::n4m::core::Config cfg;
        cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
        cfg.deflation = N4M_DEFLATION_REGRESSION;
        if (std::strcmp(fx.solver, "simpls") == 0) {
            cfg.solver = N4M_SOLVER_SIMPLS;
        } else {
            CV_CHECK(false);  // fixture declares an unexpected solver
            continue;
        }

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_matrix_view_t Y = make_view(fx.Y);
        ::n4m::core::ValidationPlan plan;
        CV_CHECK(::n4m::core::make_kfold_validation_plan(ctx, fx.X.rows, fx.n_splits, plan) == N4M_OK);

        ::n4m::core::ComponentCvResult result;
        CV_CHECK(::n4m::core::cross_validate_component_prefixes(
                     ctx, cfg, X, Y, plan, fx.n_components_max, result) == N4M_OK);
        CV_CHECK(result.max_components == fx.n_components_max);
        CV_CHECK(result.best_n_components == fx.best_n_components);
        CV_CHECK(result.metrics_by_component.size() ==
                 static_cast<std::size_t>(fx.n_components_max));
        CV_CHECK(close(result.metrics_matrix, fx.metrics_by_component, "component_cv_metrics"));
    }
}

void test_component_coefficients() {
    for (const auto& fx : ::n4m::test::fixtures::kComponentCoefficientsFixtures) {
        ::n4m::core::Context ctx;
        ::n4m::core::Config cfg;
        cfg.n_components = fx.n_components;

        n4m_matrix_view_t X = make_view(fx.X);
        n4m_matrix_view_t Y = make_view(fx.Y);

        std::unique_ptr<::n4m::core::Model> model;
        CV_CHECK(::n4m::core::fit_model(ctx, cfg, X, Y, model) == N4M_OK);
        CV_CHECK(model.get() != nullptr);

        std::vector<double> coefficients;
        CV_CHECK(::n4m::core::compute_regression_coefficients_by_component(
                     ctx, *model, coefficients) == N4M_OK);
        CV_CHECK(fx.coefficients_by_component.rows == fx.n_components);
        CV_CHECK(fx.coefficients_by_component.cols == fx.X.cols * fx.Y.cols);
        CV_CHECK(close(coefficients, fx.coefficients_by_component, "coefficients_by_component"));
    }
}

}  // namespace

// Entry point invoked from the n4m_internal_tests main (test_internal_rng_choice.cpp).
int run_internal_cv_tests() {
    test_cross_validation();
    test_component_cv();
    test_component_coefficients();
    if (g_cv_failures == 0) {
        std::printf("=== internal_cv: CV / component-CV / component-coefficients parity OK ===\n");
    } else {
        std::printf("=== internal_cv: %d failure(s) ===\n", g_cv_failures);
    }
    return g_cv_failures;
}
