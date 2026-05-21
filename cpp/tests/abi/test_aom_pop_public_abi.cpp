// SPDX-License-Identifier: CECILL-2.1
//
// Public C ABI parity tests for per-component (POP) AOM selection.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "pls4all/p4a.h"

#include "fixtures/aom_pop_selection_fixtures.hpp"
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

void check_close_buffer(int& failures,
                        const char* label,
                        const double* actual,
                        std::int32_t actual_size,
                        const ::n4m::test::fixtures::MatrixRef& expected) {
    const auto expected_size = static_cast<std::int32_t>(expected.size);
    if (actual_size != expected_size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s size: actual %d expected %d\n",
                     label,
                     static_cast<int>(actual_size),
                     static_cast<int>(expected_size));
        return;
    }
    if (actual_size == 0) return;
    for (std::int32_t i = 0; i < actual_size; ++i) {
        const double diff = std::fabs(actual[i] - expected.values[i]);
        const double scale = std::max(std::max(std::fabs(actual[i]),
                                               std::fabs(expected.values[i])),
                                      1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%d]: actual %.17g expected %.17g diff %.3g\n",
                         label,
                         static_cast<int>(i),
                         actual[i],
                         expected.values[i],
                         diff);
            return;
        }
    }
}

void check_operator_kinds(int& failures,
                          const n4m_operator_kind_t* actual,
                          std::int32_t actual_size,
                          const ::n4m::test::fixtures::AomPopSelectionIndexRef& expected) {
    const auto expected_size = static_cast<std::int32_t>(expected.size);
    if (actual_size != expected_size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL operator_kinds size: actual %d expected %d\n",
                     static_cast<int>(actual_size),
                     static_cast<int>(expected_size));
        return;
    }
    for (std::int32_t i = 0; i < actual_size; ++i) {
        const auto act = static_cast<std::int64_t>(actual[i]);
        const auto exp = expected.values[i];
        if (act != exp) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL operator_kinds[%d]: actual %lld expected %lld\n",
                         static_cast<int>(i),
                         static_cast<long long>(act),
                         static_cast<long long>(exp));
            return;
        }
    }
}

void check_selected_indices(int& failures,
                            const std::int32_t* actual,
                            std::int32_t actual_size,
                            const ::n4m::test::fixtures::AomPopSelectionIndexRef& expected) {
    const auto expected_size = static_cast<std::int32_t>(expected.size);
    if (actual_size != expected_size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL selected_operator_indices size: "
                     "actual %d expected %d\n",
                     static_cast<int>(actual_size),
                     static_cast<int>(expected_size));
        return;
    }
    for (std::int32_t i = 0; i < actual_size; ++i) {
        const auto act = static_cast<std::int64_t>(actual[i]);
        const auto exp = expected.values[i];
        if (act != exp) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL selected_operator_indices[%d]: "
                         "actual %lld expected %lld\n",
                         static_cast<int>(i),
                         static_cast<long long>(act),
                         static_cast<long long>(exp));
            return;
        }
    }
}

n4m_operator_bank_t* build_bank(
    const ::n4m::test::fixtures::AomPopSelectionFixture& fixture) {
    n4m_operator_bank_t* bank = nullptr;
    if (n4m_operator_bank_create(&bank) != N4M_OK) return nullptr;
    for (std::size_t op = 0; op < fixture.operator_kinds.size; ++op) {
        const auto start = static_cast<std::size_t>(
            fixture.operator_param_offsets.values[op]);
        const auto stop = static_cast<std::size_t>(
            fixture.operator_param_offsets.values[op + 1U]);
        const double* params = stop > start
            ? fixture.operator_params.values + start : nullptr;
        const auto n = static_cast<std::int32_t>(stop - start);
        if (n4m_operator_bank_add(
                bank,
                static_cast<n4m_operator_kind_t>(fixture.operator_kinds.values[op]),
                params, n) != N4M_OK) {
            n4m_operator_bank_destroy(bank);
            return nullptr;
        }
    }
    return bank;
}

n4m_validation_plan_t* build_plan(
    const ::n4m::test::fixtures::AomPopSelectionFixture& fixture) {
    n4m_validation_plan_t* plan = nullptr;
    if (n4m_validation_plan_create(&plan) != N4M_OK) return nullptr;
    if (n4m_validation_plan_set_n_samples(plan, fixture.X.rows) != N4M_OK) {
        n4m_validation_plan_destroy(plan);
        return nullptr;
    }
    for (std::int32_t fold = 0; fold < fixture.n_folds; ++fold) {
        const auto train_start = static_cast<std::size_t>(
            fixture.fold_train_offsets.values[fold]);
        const auto train_stop = static_cast<std::size_t>(
            fixture.fold_train_offsets.values[fold + 1]);
        const auto test_start = static_cast<std::size_t>(
            fixture.fold_test_offsets.values[fold]);
        const auto test_stop = static_cast<std::size_t>(
            fixture.fold_test_offsets.values[fold + 1]);
        const auto n_train = static_cast<std::int64_t>(train_stop - train_start);
        const auto n_test = static_cast<std::int64_t>(test_stop - test_start);
        if (n4m_validation_plan_add_fold(
                plan,
                fixture.fold_train_indices.values + train_start, n_train,
                fixture.fold_test_indices.values + test_start, n_test) != N4M_OK) {
            n4m_validation_plan_destroy(plan);
            return nullptr;
        }
    }
    return plan;
}

n4m_config_t* simpls_config() {
    n4m_config_t* cfg = nullptr;
    if (n4m_config_create(&cfg) != N4M_OK) return nullptr;
    n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
    n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
    n4m_config_set_center_x(cfg, 1);
    n4m_config_set_scale_x(cfg, 0);
    n4m_config_set_center_y(cfg, 1);
    n4m_config_set_scale_y(cfg, 0);
    n4m_config_set_store_scores(cfg, 0);
    return cfg;
}

void check_fixture(int& failures,
                   const ::n4m::test::fixtures::AomPopSelectionFixture& fixture) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    n4m_config_t* cfg = simpls_config();
    CHECK(cfg != nullptr);
    n4m_operator_bank_t* bank = build_bank(fixture);
    CHECK(bank != nullptr);
    n4m_validation_plan_t* plan = build_plan(fixture);
    CHECK(plan != nullptr);
    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);

    n4m_aom_per_component_result_t* result = nullptr;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, bank, &X, &Y, plan, fixture.max_components, &result),
             N4M_OK);
    CHECK(result != nullptr);

    int32_t n_operators = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_n_operators(result, &n_operators),
             N4M_OK);
    CHECK_EQ(n_operators, fixture.n_operators);

    int32_t max_components = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_max_components(result, &max_components),
             N4M_OK);
    CHECK_EQ(max_components, fixture.max_components);

    int32_t selected_n_components = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_selected_n_components(
                 result, &selected_n_components),
             N4M_OK);
    CHECK_EQ(selected_n_components, fixture.selected_n_components);

    double best_score = 0.0;
    CHECK_EQ(n4m_aom_per_component_result_get_best_score(result, &best_score), N4M_OK);
    const double best_diff = std::fabs(best_score - fixture.best_score);
    if (best_diff > kAbsTol &&
        best_diff > kRelTol * std::max(std::fabs(fixture.best_score), 1.0)) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL best_score: actual %.17g expected %.17g diff %.3g\n",
                     best_score,
                     fixture.best_score,
                     best_diff);
    }

    const n4m_operator_kind_t* kinds = nullptr;
    int32_t kinds_size = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_operator_kinds(
                 result, &kinds, &kinds_size),
             N4M_OK);
    check_operator_kinds(failures, kinds, kinds_size, fixture.operator_kinds);

    const std::int32_t* sel = nullptr;
    int32_t sel_size = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_selected_operator_indices(
                 result, &sel, &sel_size),
             N4M_OK);
    check_selected_indices(failures, sel, sel_size, fixture.selected_operator_indices);

    const double* comp_scores = nullptr;
    int32_t comp_rows = 0;
    int32_t comp_cols = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_component_scores(
                 result, &comp_scores, &comp_rows, &comp_cols),
             N4M_OK);
    CHECK_EQ(comp_rows, fixture.max_components);
    CHECK_EQ(comp_cols, fixture.n_operators);
    check_close_buffer(failures, "component_scores", comp_scores,
                       comp_rows * comp_cols, fixture.component_scores);

    const double* prefix = nullptr;
    int32_t prefix_size = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_prefix_scores(
                 result, &prefix, &prefix_size),
             N4M_OK);
    check_close_buffer(failures, "prefix_scores", prefix,
                       prefix_size, fixture.prefix_scores);

    const double* preds = nullptr;
    int64_t pred_rows = 0;
    int64_t pred_cols = 0;
    CHECK_EQ(n4m_aom_per_component_result_get_predictions(
                 result, &preds, &pred_rows, &pred_cols),
             N4M_OK);
    CHECK_EQ(pred_rows, fixture.X.rows);
    CHECK_EQ(pred_cols, fixture.Y.cols);
    check_close_buffer(failures, "predictions", preds,
                       static_cast<int32_t>(pred_rows * pred_cols),
                       fixture.predictions);

    n4m_aom_per_component_result_destroy(result);
    n4m_validation_plan_destroy(plan);
    n4m_operator_bank_destroy(bank);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

TEST(aom_per_component_public_abi_phase6f,
     fixture_matches_internal_kernel_through_public_surface) {
    for (const auto& fixture : ::n4m::test::fixtures::kAomPopSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(aom_per_component_public_abi_phase6f, rejects_invalid_inputs) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    n4m_config_t* cfg = simpls_config();
    CHECK(cfg != nullptr);
    const auto& fixture = ::n4m::test::fixtures::kAomPopSelectionFixtures[0];
    n4m_operator_bank_t* bank = build_bank(fixture);
    n4m_validation_plan_t* plan = build_plan(fixture);
    n4m_matrix_view_t X = matrix_view(fixture.X);
    n4m_matrix_view_t Y = matrix_view(fixture.Y);
    auto* sentinel = reinterpret_cast<n4m_aom_per_component_result_t*>(0x1);

    n4m_aom_per_component_result_t* result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 nullptr, cfg, bank, &X, &Y, plan, fixture.max_components, &result),
             N4M_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, nullptr, bank, &X, &Y, plan, fixture.max_components, &result),
             N4M_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, nullptr, &X, &Y, plan, fixture.max_components, &result),
             N4M_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, bank, nullptr, &Y, plan, fixture.max_components, &result),
             N4M_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, bank, &X, nullptr, plan, fixture.max_components, &result),
             N4M_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, bank, &X, &Y, nullptr, fixture.max_components, &result),
             N4M_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, bank, &X, &Y, plan, -1, &result),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK(result == nullptr);

    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, bank, &X, &Y, plan, fixture.max_components, nullptr),
             N4M_ERR_NULL_POINTER);

    n4m_operator_bank_t* non_strict = nullptr;
    CHECK_EQ(n4m_operator_bank_create(&non_strict), N4M_OK);
    CHECK_EQ(n4m_operator_bank_add(non_strict, N4M_OP_SNV, nullptr, 0), N4M_OK);
    result = sentinel;
    CHECK_EQ(n4m_aom_per_component_select(
                 ctx, cfg, non_strict, &X, &Y, plan,
                 fixture.max_components, &result),
             N4M_ERR_UNSUPPORTED);
    CHECK(result == nullptr);
    n4m_operator_bank_destroy(non_strict);

    n4m_validation_plan_destroy(plan);
    n4m_operator_bank_destroy(bank);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

TEST(aom_per_component_public_abi_phase6f, getter_null_paths) {
    int32_t i = 0;
    double d = 0.0;
    const double* dp = nullptr;
    const n4m_operator_kind_t* kp = nullptr;
    const int32_t* ip = nullptr;
    int32_t r = 0;
    int32_t c = 0;
    int64_t r64 = 0;
    int64_t c64 = 0;

    CHECK_EQ(n4m_aom_per_component_result_get_n_operators(nullptr, &i),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_max_components(nullptr, &i),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_selected_n_components(nullptr, &i),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_best_score(nullptr, &d),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_operator_kinds(nullptr, &kp, &i),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_selected_operator_indices(
                 nullptr, &ip, &i),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_component_scores(
                 nullptr, &dp, &r, &c),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_prefix_scores(nullptr, &dp, &i),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_aom_per_component_result_get_predictions(nullptr, &dp, &r64, &c64),
             N4M_ERR_NULL_POINTER);

    n4m_aom_per_component_result_destroy(nullptr);
}
