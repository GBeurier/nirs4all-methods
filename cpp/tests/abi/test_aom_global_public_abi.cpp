// SPDX-License-Identifier: CeCILL-2.1
//
// Public C ABI parity tests for global AOM selection. These exercise the
// p4a_aom_global_select / *_result_get_* / *_result_destroy entry points and
// compare them against the same `kAomSelectionFixtures` constants the
// internal-test suite uses.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "pls4all/p4a.h"

#include "fixtures/aom_selection_fixtures.hpp"
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

void check_close_buffer(int& failures,
                        const char* label,
                        const double* actual,
                        std::int32_t actual_size,
                        const ::pls4all::test::fixtures::MatrixRef& expected) {
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

void check_indices_buffer(int& failures,
                          const char* label,
                          const p4a_operator_kind_t* actual,
                          std::int32_t actual_size,
                          const ::pls4all::test::fixtures::AomSelectionIndexRef& expected) {
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
    for (std::int32_t i = 0; i < actual_size; ++i) {
        const auto act = static_cast<std::int64_t>(actual[i]);
        const auto exp = static_cast<std::int64_t>(expected.values[i]);
        if (act != exp) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%d]: actual %lld expected %lld\n",
                         label,
                         static_cast<int>(i),
                         static_cast<long long>(act),
                         static_cast<long long>(exp));
            return;
        }
    }
}

p4a_operator_bank_t* build_bank(
    const ::pls4all::test::fixtures::AomSelectionFixture& fixture) {
    p4a_operator_bank_t* bank = nullptr;
    if (p4a_operator_bank_create(&bank) != P4A_OK) {
        return nullptr;
    }
    for (std::size_t op = 0; op < fixture.operator_kinds.size; ++op) {
        const auto start = static_cast<std::size_t>(
            fixture.operator_param_offsets.values[op]);
        const auto stop = static_cast<std::size_t>(
            fixture.operator_param_offsets.values[op + 1U]);
        const double* params = stop > start
            ? fixture.operator_params.values + start : nullptr;
        const auto n = static_cast<std::int32_t>(stop - start);
        if (p4a_operator_bank_add(
                bank,
                static_cast<p4a_operator_kind_t>(fixture.operator_kinds.values[op]),
                params, n) != P4A_OK) {
            p4a_operator_bank_destroy(bank);
            return nullptr;
        }
    }
    return bank;
}

p4a_validation_plan_t* build_plan(
    const ::pls4all::test::fixtures::AomSelectionFixture& fixture) {
    p4a_validation_plan_t* plan = nullptr;
    if (p4a_validation_plan_create(&plan) != P4A_OK) {
        return nullptr;
    }
    if (p4a_validation_plan_set_n_samples(plan, fixture.X.rows) != P4A_OK) {
        p4a_validation_plan_destroy(plan);
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
        if (p4a_validation_plan_add_fold(
                plan,
                fixture.fold_train_indices.values + train_start, n_train,
                fixture.fold_test_indices.values + test_start, n_test) != P4A_OK) {
            p4a_validation_plan_destroy(plan);
            return nullptr;
        }
    }
    return plan;
}

p4a_config_t* simpls_config() {
    p4a_config_t* cfg = nullptr;
    if (p4a_config_create(&cfg) != P4A_OK) return nullptr;
    p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_REGRESSION);
    p4a_config_set_solver(cfg, P4A_SOLVER_SIMPLS);
    p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION);
    p4a_config_set_center_x(cfg, 1);
    p4a_config_set_scale_x(cfg, 0);
    p4a_config_set_center_y(cfg, 1);
    p4a_config_set_scale_y(cfg, 0);
    p4a_config_set_store_scores(cfg, 0);
    return cfg;
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::AomSelectionFixture& fixture) {
    p4a_context_t* ctx = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    p4a_config_t* cfg = simpls_config();
    CHECK(cfg != nullptr);
    p4a_operator_bank_t* bank = build_bank(fixture);
    CHECK(bank != nullptr);
    p4a_validation_plan_t* plan = build_plan(fixture);
    CHECK(plan != nullptr);
    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);

    p4a_aom_global_result_t* result = nullptr;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, plan, fixture.max_components, &result),
             P4A_OK);
    CHECK(result != nullptr);

    int32_t n_operators = 0;
    CHECK_EQ(p4a_aom_global_result_get_n_operators(result, &n_operators), P4A_OK);
    CHECK_EQ(n_operators, fixture.n_operators);

    int32_t max_components = 0;
    CHECK_EQ(p4a_aom_global_result_get_max_components(result, &max_components), P4A_OK);
    CHECK_EQ(max_components, fixture.max_components);

    int32_t selected_operator_index = -1;
    CHECK_EQ(p4a_aom_global_result_get_selected_operator_index(
                 result, &selected_operator_index),
             P4A_OK);
    CHECK_EQ(selected_operator_index, fixture.selected_operator_index);

    int32_t selected_n_components = 0;
    CHECK_EQ(p4a_aom_global_result_get_selected_n_components(
                 result, &selected_n_components),
             P4A_OK);
    CHECK_EQ(selected_n_components, fixture.selected_n_components);

    double best_score = 0.0;
    CHECK_EQ(p4a_aom_global_result_get_best_score(result, &best_score), P4A_OK);
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

    const p4a_operator_kind_t* kinds = nullptr;
    int32_t kinds_size = 0;
    CHECK_EQ(p4a_aom_global_result_get_operator_kinds(result, &kinds, &kinds_size),
             P4A_OK);
    check_indices_buffer(failures, "operator_kinds", kinds, kinds_size,
                         fixture.operator_kinds);

    const double* op_scores = nullptr;
    int32_t op_scores_size = 0;
    CHECK_EQ(p4a_aom_global_result_get_operator_scores(
                 result, &op_scores, &op_scores_size),
             P4A_OK);
    check_close_buffer(failures, "operator_scores", op_scores, op_scores_size,
                       fixture.operator_scores);

    const double* curves = nullptr;
    int32_t curve_rows = 0;
    int32_t curve_cols = 0;
    CHECK_EQ(p4a_aom_global_result_get_rmse_curves(
                 result, &curves, &curve_rows, &curve_cols),
             P4A_OK);
    CHECK_EQ(curve_rows, fixture.n_operators);
    CHECK_EQ(curve_cols, fixture.max_components);
    check_close_buffer(failures, "rmse_curves", curves,
                       curve_rows * curve_cols, fixture.rmse_curves);

    const double* preds = nullptr;
    int64_t pred_rows = 0;
    int64_t pred_cols = 0;
    CHECK_EQ(p4a_aom_global_result_get_predictions(
                 result, &preds, &pred_rows, &pred_cols),
             P4A_OK);
    CHECK_EQ(pred_rows, fixture.X.rows);
    CHECK_EQ(pred_cols, fixture.Y.cols);
    check_close_buffer(failures, "predictions", preds,
                       static_cast<int32_t>(pred_rows * pred_cols),
                       fixture.predictions);

    p4a_aom_global_result_destroy(result);
    p4a_validation_plan_destroy(plan);
    p4a_operator_bank_destroy(bank);
    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
}

}  // namespace

TEST(aom_global_public_abi_phase6f,
     fixture_matches_internal_kernel_through_public_surface) {
    for (const auto& fixture : ::pls4all::test::fixtures::kAomSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(aom_global_public_abi_phase6f, rejects_invalid_inputs) {
    p4a_context_t* ctx = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    p4a_config_t* cfg = simpls_config();
    CHECK(cfg != nullptr);
    const auto& fixture = ::pls4all::test::fixtures::kAomSelectionFixtures[0];
    p4a_operator_bank_t* bank = build_bank(fixture);
    p4a_validation_plan_t* plan = build_plan(fixture);
    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    auto* sentinel = reinterpret_cast<p4a_aom_global_result_t*>(0x1);

    p4a_aom_global_result_t* result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 nullptr, cfg, bank, &X, &Y, plan, fixture.max_components, &result),
             P4A_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, nullptr, bank, &X, &Y, plan, fixture.max_components, &result),
             P4A_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, nullptr, &X, &Y, plan, fixture.max_components, &result),
             P4A_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, nullptr, &Y, plan, fixture.max_components, &result),
             P4A_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, nullptr, plan, fixture.max_components, &result),
             P4A_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, nullptr, fixture.max_components, &result),
             P4A_ERR_NULL_POINTER);
    CHECK(result == nullptr);

    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, plan, 0, &result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK(result == nullptr);

    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, plan, fixture.max_components, nullptr),
             P4A_ERR_NULL_POINTER);

    // Empty bank -> P4A_ERR_INVALID_ARGUMENT.
    p4a_operator_bank_t* empty_bank = nullptr;
    CHECK_EQ(p4a_operator_bank_create(&empty_bank), P4A_OK);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, empty_bank, &X, &Y, plan,
                 fixture.max_components, &result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK(result == nullptr);
    p4a_operator_bank_destroy(empty_bank);

    // Empty plan -> P4A_ERR_INVALID_ARGUMENT.
    p4a_validation_plan_t* empty_plan = nullptr;
    CHECK_EQ(p4a_validation_plan_create(&empty_plan), P4A_OK);
    CHECK_EQ(p4a_validation_plan_set_n_samples(empty_plan, fixture.X.rows), P4A_OK);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, empty_plan,
                 fixture.max_components, &result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK(result == nullptr);
    p4a_validation_plan_destroy(empty_plan);

    // Plan sample-count mismatch -> P4A_ERR_SHAPE_MISMATCH.
    p4a_validation_plan_t* mismatched = nullptr;
    CHECK_EQ(p4a_validation_plan_create(&mismatched), P4A_OK);
    CHECK_EQ(p4a_validation_plan_set_n_samples(mismatched, fixture.X.rows + 1), P4A_OK);
    const int64_t train_idx[] = {0, 1};
    const int64_t test_idx[] = {2};
    CHECK_EQ(p4a_validation_plan_add_fold(
                 mismatched, train_idx, 2, test_idx, 1),
             P4A_OK);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, mismatched,
                 fixture.max_components, &result),
             P4A_ERR_SHAPE_MISMATCH);
    CHECK(result == nullptr);
    p4a_validation_plan_destroy(mismatched);

    // Plan with train/test overlap -> P4A_ERR_INVALID_ARGUMENT.
    p4a_validation_plan_t* overlap = nullptr;
    CHECK_EQ(p4a_validation_plan_create(&overlap), P4A_OK);
    CHECK_EQ(p4a_validation_plan_set_n_samples(overlap, fixture.X.rows), P4A_OK);
    const int64_t overlap_train[] = {0, 1, 2};
    const int64_t overlap_test[] = {2, 3};
    CHECK_EQ(p4a_validation_plan_add_fold(
                 overlap, overlap_train, 3, overlap_test, 2),
             P4A_OK);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, overlap,
                 fixture.max_components, &result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK(result == nullptr);
    p4a_validation_plan_destroy(overlap);

    // Out-of-range index -> P4A_ERR_INVALID_ARGUMENT.
    p4a_validation_plan_t* bad_index = nullptr;
    CHECK_EQ(p4a_validation_plan_create(&bad_index), P4A_OK);
    CHECK_EQ(p4a_validation_plan_set_n_samples(bad_index, fixture.X.rows), P4A_OK);
    const int64_t bad_train[] = {0, 1};
    const int64_t bad_test[] = {static_cast<int64_t>(fixture.X.rows)};
    CHECK_EQ(p4a_validation_plan_add_fold(
                 bad_index, bad_train, 2, bad_test, 1),
             P4A_OK);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, bank, &X, &Y, bad_index,
                 fixture.max_components, &result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK(result == nullptr);
    p4a_validation_plan_destroy(bad_index);

    // Unsupported solver -> P4A_ERR_UNSUPPORTED.
    p4a_config_t* bad_cfg = simpls_config();
    p4a_config_set_solver(bad_cfg, P4A_SOLVER_NIPALS);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, bad_cfg, bank, &X, &Y, plan,
                 fixture.max_components, &result),
             P4A_ERR_UNSUPPORTED);
    CHECK(result == nullptr);
    p4a_config_destroy(bad_cfg);

    // Non-strict operator -> P4A_ERR_UNSUPPORTED from kernel.
    p4a_operator_bank_t* non_strict = nullptr;
    CHECK_EQ(p4a_operator_bank_create(&non_strict), P4A_OK);
    CHECK_EQ(p4a_operator_bank_add(non_strict, P4A_OP_SNV, nullptr, 0), P4A_OK);
    result = sentinel;
    CHECK_EQ(p4a_aom_global_select(
                 ctx, cfg, non_strict, &X, &Y, plan,
                 fixture.max_components, &result),
             P4A_ERR_UNSUPPORTED);
    CHECK(result == nullptr);
    p4a_operator_bank_destroy(non_strict);

    p4a_validation_plan_destroy(plan);
    p4a_operator_bank_destroy(bank);
    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
}

TEST(aom_global_public_abi_phase6f, getter_null_paths) {
    int32_t i = 0;
    double d = 0.0;
    const double* dp = nullptr;
    const p4a_operator_kind_t* kp = nullptr;
    int32_t r = 0;
    int32_t c = 0;
    int64_t r64 = 0;
    int64_t c64 = 0;

    CHECK_EQ(p4a_aom_global_result_get_n_operators(nullptr, &i),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_max_components(nullptr, &i),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_selected_operator_index(nullptr, &i),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_selected_n_components(nullptr, &i),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_best_score(nullptr, &d),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_operator_kinds(nullptr, &kp, &i),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_operator_scores(nullptr, &dp, &i),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_rmse_curves(nullptr, &dp, &r, &c),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_aom_global_result_get_predictions(nullptr, &dp, &r64, &c64),
             P4A_ERR_NULL_POINTER);

    p4a_aom_global_result_destroy(nullptr);
}
