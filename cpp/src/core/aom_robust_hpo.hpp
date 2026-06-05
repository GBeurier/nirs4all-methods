// SPDX-License-Identifier: CECILL-2.1
//
// Native AOM robust-HPO screening over strict-linear preprocessing chains.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

enum class AomRobustHpoProfile : std::int32_t {
    kCompact = 0,
    kWide = 1,
};

enum class AomRobustHpoHead : std::int32_t {
    kRidge = 0,
    kPls = 1,
};

constexpr std::int32_t kAomRobustHpoHeadMaskRidge = 1;
constexpr std::int32_t kAomRobustHpoHeadMaskPls = 2;

struct AomRobustHpoResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int32_t n_features_transformed{0};
    std::int32_t profile{0};
    std::int32_t cv{0};
    std::int32_t heads_mask{0};

    std::int32_t selected_chain_id{-1};
    std::int32_t selected_head_id{-1};
    double selected_param{0.0};
    double selected_cv_rmse{0.0};

    std::int32_t n_chains{0};
    std::int32_t n_candidates{0};

    std::vector<double> predictions;                // n x 1
    std::vector<double> coefficients_transformed;   // p_transformed x 1
    std::vector<double> input_coefficients;          // original input p x 1
    std::vector<double> intercept;                  // 1 x 1
    std::vector<double> candidate_scores;           // n_candidates x 4:
                                                     // chain_id, head_id,
                                                     // param, mean CV RMSE.
};

[[nodiscard]] n4m_status_t fit_aom_robust_hpo(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    std::int32_t heads_mask,
    AomRobustHpoResult& out);

}  // namespace n4m::core
