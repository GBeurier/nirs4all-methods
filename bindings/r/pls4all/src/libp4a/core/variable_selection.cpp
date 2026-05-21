// SPDX-License-Identifier: CECILL-2.1

#include "core/variable_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <numeric>
#include <utility>

#include "core/variable_importance.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] n4m_status_t validate_top_k(::n4m::core::Context& ctx,
                                          const ::n4m::core::Model& model,
                                          std::int32_t top_k) {
    if (model.n_features <= 0 || model.n_targets <= 0) {
        ctx.set_error("fitted model dimensions are invalid for variable selection");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (top_k <= 0 || top_k > model.n_features) {
        ctx.set_errorf("top_k must be in [1, %d]", static_cast<int>(model.n_features));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t compute_coefficient_magnitude(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Model& model,
    std::vector<double>& out) {
    const auto p = static_cast<std::size_t>(model.n_features);
    const auto q = static_cast<std::size_t>(model.n_targets);
    if (model.coefficients.size() != p * q) {
        ctx.set_error("fitted model coefficients are inconsistent for variable selection");
        return N4M_ERR_INTERNAL;
    }

    out.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double max_abs = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            max_abs = std::max(max_abs, std::fabs(model.coefficients[idx(feature, q, target)]));
        }
        out[feature] = max_abs;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t compute_scores(::n4m::core::Context& ctx,
                                          const ::n4m::core::Model& model,
                                          const n4m_matrix_view_t& X,
                                          ::n4m::core::VariableSelectionMethod method,
                                          std::vector<double>& scores) {
    switch (method) {
    case ::n4m::core::VariableSelectionMethod::Vip:
        return ::n4m::core::compute_vip_scores(ctx, model, scores);
    case ::n4m::core::VariableSelectionMethod::CoefficientMagnitude:
        return compute_coefficient_magnitude(ctx, model, scores);
    case ::n4m::core::VariableSelectionMethod::SelectivityRatio:
        return ::n4m::core::compute_selectivity_ratio(ctx, model, X, scores);
    }
    ctx.set_error("unknown variable-selection method");
    return N4M_ERR_INVALID_ARGUMENT;
}

[[nodiscard]] std::vector<std::int64_t> rank_descending(const std::vector<double>& scores) {
    std::vector<std::int64_t> order(scores.size(), 0);
    std::iota(order.begin(), order.end(), static_cast<std::int64_t>(0));
    std::stable_sort(order.begin(), order.end(), [&scores](std::int64_t lhs, std::int64_t rhs) {
        const double left = scores[static_cast<std::size_t>(lhs)];
        const double right = scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });
    return order;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_variables(Context& ctx,
                              const Model& model,
                              const n4m_matrix_view_t& X,
                              VariableSelectionMethod method,
                              std::int32_t top_k,
                              VariableSelectionResult& out) {
    try {
        out = VariableSelectionResult{};
        n4m_status_t status = validate_top_k(ctx, model, top_k);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> scores;
        status = compute_scores(ctx, model, X, method, scores);
        if (status != N4M_OK) {
            return status;
        }
        if (scores.size() != static_cast<std::size_t>(model.n_features)) {
            ctx.set_error("variable-selection score count does not match fitted feature count");
            return N4M_ERR_INTERNAL;
        }

        std::vector<std::int64_t> order = rank_descending(scores);
        order.resize(static_cast<std::size_t>(top_k));

        out.n_features = model.n_features;
        out.top_k = top_k;
        out.scores = std::move(scores);
        out.selected_indices = std::move(order);
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while selecting variables");
        out = VariableSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while selecting variables");
        out = VariableSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
