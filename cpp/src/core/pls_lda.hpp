// SPDX-License-Identifier: CECILL-2.1
//
// Internal PLS-LDA classifier kernel.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

struct PlsLdaResult {
    std::int64_t n_samples{0};
    std::int32_t n_classes{0};
    std::int32_t n_components{0};

    std::vector<std::int32_t> predictions;
    std::vector<double> decision_scores; // row-major n_samples x n_classes
};

// Linear discriminant head fitted on PLS scores: the decision for a score
// row s and class c is constants[c] + s . inv_means[c].
struct PlsLdaHead {
    std::int32_t n_classes{0};
    std::int32_t n_components{0};
    std::vector<double> inv_means;  // row-major n_classes x n_components
    std::vector<double> constants;  // n_classes
};

// Fits the head on row-major scores (n x k) and 0-based labels.
[[nodiscard]] n4m_status_t fit_pls_lda_head(Context& ctx,
                                            const std::vector<double>& scores,
                                            const std::vector<std::int32_t>& labels,
                                            std::int64_t n_samples,
                                            std::int32_t n_components,
                                            std::int32_t n_classes,
                                            PlsLdaHead& out);

// Decision scores (row-major n x n_classes) for row-major scores (n x k).
void pls_lda_decision(const PlsLdaHead& head, const double* scores, std::int64_t n_samples,
                      double* decision);

[[nodiscard]] n4m_status_t fit_predict_pls_lda(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& labels,
    std::int32_t n_classes,
    PlsLdaResult& out);

}  // namespace n4m::core
