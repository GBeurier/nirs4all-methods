// SPDX-License-Identifier: CECILL-2.1
//
// Internal validation split generators used by future cross-validation engines.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"

namespace n4m::core {

struct ValidationFold {
    std::vector<std::int64_t> train_indices;
    std::vector<std::int64_t> test_indices;
};

struct ValidationPlan {
    std::int64_t n_samples{0};
    std::vector<ValidationFold> folds;
};

[[nodiscard]] n4m_status_t make_kfold_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    std::int32_t n_splits,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_leave_one_out_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_holdout_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    std::int64_t test_start,
    std::int64_t test_count,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_external_folds_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    const std::int64_t* fold_ids,
    std::int32_t n_folds,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_repeated_kfold_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    std::int32_t n_splits,
    std::int32_t n_repeats,
    std::uint64_t seed,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_monte_carlo_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    std::int64_t test_count,
    std::int32_t n_repeats,
    std::uint64_t seed,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_kennard_stone_validation_plan(
    Context& ctx,
    const n4m_matrix_view_t& X,
    std::int64_t train_count,
    ValidationPlan& out);

[[nodiscard]] n4m_status_t make_spxy_validation_plan(
    Context& ctx,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int64_t train_count,
    ValidationPlan& out);

}  // namespace n4m::core

struct n4m_validation_plan_s : public ::n4m::core::ValidationPlan {};
