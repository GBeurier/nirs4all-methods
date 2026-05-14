// SPDX-License-Identifier: CeCILL-2.1
//
// Internal validation split generators used by future cross-validation engines.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"

namespace pls4all::core {

struct ValidationFold {
    std::vector<std::int64_t> train_indices;
    std::vector<std::int64_t> test_indices;
};

struct ValidationPlan {
    std::int64_t n_samples{0};
    std::vector<ValidationFold> folds;
};

[[nodiscard]] p4a_status_t make_kfold_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    std::int32_t n_splits,
    ValidationPlan& out);

[[nodiscard]] p4a_status_t make_leave_one_out_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    ValidationPlan& out);

[[nodiscard]] p4a_status_t make_holdout_validation_plan(
    Context& ctx,
    std::int64_t n_samples,
    std::int64_t test_start,
    std::int64_t test_count,
    ValidationPlan& out);

}  // namespace pls4all::core
