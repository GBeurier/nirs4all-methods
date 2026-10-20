// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic Hotelling T2-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct T2SelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_latent_directions{0};  // components with a nonzero weight column
    std::int32_t n_alphas{0};
    std::int32_t min_selected{0};
    std::int32_t best_error_index{-1};
    std::int32_t min_set_index{-1};
    double best_rmse{0.0};
    double min_set_rmse{0.0};

    std::vector<double> t2_scores;
    std::vector<double> ucl_by_alpha;
    std::vector<double> rmse_by_alpha;
    std::vector<std::int64_t> selected_counts;
    std::vector<std::int64_t> selected_mask; // row-major n_alphas x n_features
    std::vector<std::int64_t> selected_indices_best_error;
    std::vector<std::int64_t> selected_indices_min_set;
};

// Upper (1 - alpha) quantile of Beta(a, b), a, b >= 1/2, alpha in (0, 1):
// the x with P(X > x) = alpha. The small tail (upper for alpha <= 1/2, lower
// otherwise) is inverted in logarithms on the odds x / (1 - x), by bisection
// on the normal doubles, so neither tail is lost to a complement close to 1;
// relative accuracy of x about 1e-14 (continued fraction and Lanczos ln B).
// A quantile below DBL_MIN / (1 + DBL_MIN) is returned as about DBL_MIN.
// False when the continued fraction did not converge.
[[nodiscard]] bool t2_beta_upper_quantile(double alpha, double a, double b, double& out) noexcept;

// Weight-T2 variable selection on the components with a nonzero weight
// column (null PLS components excluded; m of them): T2 over the m weight
// coordinates, limit (p - 1)^2 / p Beta_{1-alpha}(m / 2, (p - m - 1) / 2),
// both those of the model truncated to its resolved components. With m = 0
// every T2 and limit is 0.
[[nodiscard]] n4m_status_t select_by_t2(Context& ctx,
                                        const Config& cfg,
                                        const n4m_matrix_view_t& X,
                                        const n4m_matrix_view_t& Y,
                                        const ValidationPlan& plan,
                                        const std::vector<double>& alpha,
                                        std::int32_t min_selected,
                                        T2SelectionResult& out);

}  // namespace n4m::core
