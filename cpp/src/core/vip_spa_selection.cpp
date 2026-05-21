// SPDX-License-Identifier: CECILL-2.1

#include "core/vip_spa_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/variable_importance.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] bool checked_matrix_size(std::int64_t rows,
                                       std::int64_t cols,
                                       std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const auto urows = static_cast<std::size_t>(rows);
    const auto ucols = static_cast<std::size_t>(cols);
    if (ucols != 0U &&
        urows > std::numeric_limits<std::size_t>::max() / ucols) {
        return false;
    }
    out = urows * ucols;
    return true;
}

[[nodiscard]] double read_value(const n4m_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == N4M_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
}

[[nodiscard]] n4m_status_t validate_float_view(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& view,
                                               const char* name) noexcept {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(view);
    if (status != N4M_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::n4m::core::status_to_string(status));
        return status;
    }
    if (view.dtype != N4M_DTYPE_F64 && view.dtype != N4M_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_request(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& cfg,
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            double vip_threshold,
                                            std::int32_t top_k) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("VIP_SPA matrices must be non-empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (top_k <= 0 || top_k > X.cols) {
        ctx.set_errorf("top_k must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(top_k));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(vip_threshold) || vip_threshold < 0.0) {
        ctx.set_errorf("vip_threshold must be a finite non-negative scalar; got %g",
                       vip_threshold);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("VIP_SPA matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_raw_x(::n4m::core::Context& ctx,
                                       const n4m_matrix_view_t& X,
                                       std::vector<double>& out) {
    // auswahl.VIP_SPA / SPA run successive projections on RAW masked X
    // (auswahl._spa._fit_spa only L2-normalizes the *current* direction
    // before projecting; it never centers/scales columns). Copying raw
    // values preserves that contract so the parity tolerance reflects
    // RNG-only differences instead of algorithmic divergence.
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, X.cols, n_values)) {
        ctx.set_error("VIP_SPA matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(X, row, col);
            if (!std::isfinite(value)) {
                ctx.set_error("VIP_SPA X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] double dot(const std::vector<double>& lhs,
                         const std::vector<double>& rhs) noexcept {
    double out = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        out += lhs[i] * rhs[i];
    }
    return out;
}

[[nodiscard]] double norm2(const std::vector<double>& values) noexcept {
    return std::sqrt(dot(values, values));
}

[[nodiscard]] std::vector<double> column_vector(const std::vector<double>& matrix,
                                                std::size_t rows,
                                                std::size_t cols,
                                                std::size_t col) {
    std::vector<double> out(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        out[row] = matrix[idx(row, cols, col)];
    }
    return out;
}

void subtract_projection(std::vector<double>& vector,
                         const std::vector<double>& basis_vector) noexcept {
    const double coefficient = dot(vector, basis_vector);
    for (std::size_t i = 0; i < vector.size(); ++i) {
        vector[i] -= coefficient * basis_vector[i];
    }
}

[[nodiscard]] std::vector<std::vector<double>> orthonormal_basis(
    const std::vector<double>& standardized_x,
    std::size_t rows,
    std::size_t cols,
    const std::vector<std::int64_t>& selected) {
    std::vector<std::vector<double>> basis;
    basis.reserve(selected.size());
    for (const std::int64_t feature : selected) {
        std::vector<double> vector =
            column_vector(standardized_x, rows, cols, static_cast<std::size_t>(feature));
        for (const auto& basis_vector : basis) {
            subtract_projection(vector, basis_vector);
        }
        const double nrm = norm2(vector);
        if (nrm > std::numeric_limits<double>::epsilon()) {
            for (double& value : vector) {
                value /= nrm;
            }
            basis.push_back(std::move(vector));
        }
    }
    return basis;
}

[[nodiscard]] double residual_norm(const std::vector<double>& standardized_x,
                                   std::size_t rows,
                                   std::size_t cols,
                                   std::size_t feature,
                                   const std::vector<std::vector<double>>& basis) {
    std::vector<double> residual = column_vector(standardized_x, rows, cols, feature);
    for (const auto& basis_vector : basis) {
        subtract_projection(residual, basis_vector);
    }
    return norm2(residual);
}

[[nodiscard]] std::int64_t argmax_break_ties_low(
    const std::vector<double>& scores,
    const std::vector<unsigned char>& eligible) noexcept {
    std::int64_t best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (eligible[i] == 0U) {
            continue;
        }
        if (scores[i] > best_score) {
            best = static_cast<std::int64_t>(i);
            best_score = scores[i];
        }
    }
    return best;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_vip_spa(Context& ctx,
                               const Config& cfg,
                               const n4m_matrix_view_t& X,
                               const n4m_matrix_view_t& Y,
                               double vip_threshold,
                               std::int32_t top_k,
                               VipSpaSelectionResult& out) {
    try {
        out = VipSpaSelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, X, Y, vip_threshold, top_k);
        if (status != N4M_OK) {
            return status;
        }

        // VIP needs the model to keep its T scores. Take a local copy of
        // the config and force store_scores=1 so the caller doesn't have
        // to remember it. Mirrors how compute_vip_scores asserts on a
        // model that was fitted without scores.
        Config local_cfg = cfg;
        local_cfg.store_scores = 1;

        std::unique_ptr<Model> model;
        status = fit_model(ctx, local_cfg, X, Y, model);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> vip_scores;
        status = compute_vip_scores(ctx, *model, vip_scores);
        if (status != N4M_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(X.cols);
        if (vip_scores.size() != p) {
            ctx.set_error("VIP_SPA: VIP score length does not match feature count");
            return N4M_ERR_INTERNAL;
        }

        std::vector<unsigned char> mask(p, 0U);
        std::size_t candidates = 0;
        std::size_t argmax_vip = 0;
        double argmax_score = -std::numeric_limits<double>::infinity();
        for (std::size_t feature = 0; feature < p; ++feature) {
            if (vip_scores[feature] > vip_threshold) {
                mask[feature] = 1U;
                ++candidates;
            }
            if (vip_scores[feature] > argmax_score) {
                argmax_score = vip_scores[feature];
                argmax_vip = feature;
            }
        }
        // Matches auswahl VIP.get_support_for_threshold fallback: if no
        // feature passes the threshold, select the best-scoring one alone.
        if (candidates == 0U) {
            mask[argmax_vip] = 1U;
            candidates = 1U;
        }

        const auto k = static_cast<std::size_t>(top_k);
        const std::size_t reachable = std::min(k, candidates);

        std::vector<double> raw_x;
        status = copy_raw_x(ctx, X, raw_x);
        if (status != N4M_OK) {
            return status;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        std::vector<unsigned char> is_selected(p, 0U);
        std::vector<std::int64_t> selected;
        std::vector<double> selection_scores;
        selected.reserve(reachable);
        selection_scores.reserve(reachable);

        std::vector<unsigned char> eligible = mask;
        const std::int64_t start = argmax_break_ties_low(vip_scores, eligible);
        if (start < 0) {
            ctx.set_error("VIP_SPA: no candidate feature survived the VIP mask");
            return N4M_ERR_INTERNAL;
        }
        const auto start_idx = static_cast<std::size_t>(start);
        selected.push_back(start);
        is_selected[start_idx] = 1U;
        eligible[start_idx] = 0U;
        selection_scores.push_back(vip_scores[start_idx]);

        while (selected.size() < reachable) {
            const auto basis = orthonormal_basis(raw_x, rows, p, selected);
            std::int64_t best_feature = -1;
            double best_score = -std::numeric_limits<double>::infinity();
            for (std::size_t feature = 0; feature < p; ++feature) {
                if (eligible[feature] == 0U) {
                    continue;
                }
                const double score = residual_norm(raw_x, rows, p, feature, basis);
                if (score > best_score ||
                    (score == best_score && static_cast<std::int64_t>(feature) < best_feature)) {
                    best_feature = static_cast<std::int64_t>(feature);
                    best_score = score;
                }
            }
            if (best_feature < 0 || !std::isfinite(best_score)) {
                break;
            }
            const auto pick = static_cast<std::size_t>(best_feature);
            selected.push_back(best_feature);
            is_selected[pick] = 1U;
            eligible[pick] = 0U;
            selection_scores.push_back(best_score);
        }

        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.top_k = top_k;
        out.n_selected = static_cast<std::int32_t>(selected.size());
        out.n_candidates = static_cast<std::int32_t>(candidates);
        out.vip_threshold = vip_threshold;
        out.vip_scores = std::move(vip_scores);
        out.vip_mask = std::move(mask);
        out.selection_scores = std::move(selection_scores);
        out.selected_indices = std::move(selected);

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running VIP_SPA selection");
        out = VipSpaSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running VIP_SPA selection");
        out = VipSpaSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
