// SPDX-License-Identifier: CECILL-2.1

#include "core/spa_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

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
        ctx.set_error("SPA matrices must be non-empty");
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
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("SPA matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_standardized_x(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& X,
                                               std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, X.cols, n_values)) {
        ctx.set_error("SPA matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(X, row, col);
            if (!std::isfinite(value)) {
                ctx.set_error("SPA X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }

    for (std::size_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += out[idx(row, cols, col)];
        }
        const double mean = sum / static_cast<double>(rows);

        double ss = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double centered = out[idx(row, cols, col)] - mean;
            out[idx(row, cols, col)] = centered;
            ss += centered * centered;
        }

        double scale = 1.0;
        if (rows > 1U) {
            scale = std::sqrt(ss / static_cast<double>(rows - 1U));
            if (!std::isfinite(scale) || scale <= std::numeric_limits<double>::epsilon()) {
                scale = 1.0;
            }
        }
        for (std::size_t row = 0; row < rows; ++row) {
            out[idx(row, cols, col)] /= scale;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t compute_coefficient_scores(::n4m::core::Context& ctx,
                                                      const ::n4m::core::Config& cfg,
                                                      const n4m_matrix_view_t& X,
                                                      const n4m_matrix_view_t& Y,
                                                      std::vector<double>& out) {
    std::unique_ptr<::n4m::core::Model> model;
    n4m_status_t status = ::n4m::core::fit_model(ctx, cfg, X, Y, model);
    if (status != N4M_OK) {
        return status;
    }
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("SPA fitted model returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }

    out.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double best = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            best = std::max(best, std::fabs(model->coefficients[idx(feature, q, target)]));
        }
        out[feature] = best;
    }
    return N4M_OK;
}

[[nodiscard]] std::int64_t argmax_desc_index_asc(const std::vector<double>& scores) noexcept {
    std::int64_t best = 0;
    double best_score = scores.empty() ? 0.0 : scores[0];
    for (std::size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > best_score) {
            best = static_cast<std::int64_t>(i);
            best_score = scores[i];
        }
    }
    return best;
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

}  // namespace

namespace n4m::core {

n4m_status_t select_by_spa(Context& ctx,
                           const Config& cfg,
                           const n4m_matrix_view_t& X,
                           const n4m_matrix_view_t& Y,
                           std::int32_t top_k,
                           SpaSelectionResult& out) {
    try {
        out = SpaSelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, X, Y, top_k);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> standardized_x;
        status = copy_standardized_x(ctx, X, standardized_x);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> coefficient_scores;
        status = compute_coefficient_scores(ctx, cfg, X, Y, coefficient_scores);
        if (status != N4M_OK) {
            out = SpaSelectionResult{};
            return status;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const auto wanted = static_cast<std::size_t>(top_k);
        std::vector<unsigned char> is_selected(cols, 0U);
        std::vector<std::int64_t> selected;
        std::vector<double> selection_scores;
        selected.reserve(wanted);
        selection_scores.reserve(wanted);

        const std::int64_t start = argmax_desc_index_asc(coefficient_scores);
        selected.push_back(start);
        is_selected[static_cast<std::size_t>(start)] = 1U;
        selection_scores.push_back(coefficient_scores[static_cast<std::size_t>(start)]);

        while (selected.size() < wanted) {
            const auto basis = orthonormal_basis(standardized_x, rows, cols, selected);
            std::int64_t best_feature = -1;
            double best_score = -std::numeric_limits<double>::infinity();
            for (std::size_t feature = 0; feature < cols; ++feature) {
                if (is_selected[feature] != 0U) {
                    continue;
                }
                const double score = residual_norm(standardized_x, rows, cols, feature, basis);
                if (score > best_score ||
                    (score == best_score && static_cast<std::int64_t>(feature) < best_feature)) {
                    best_feature = static_cast<std::int64_t>(feature);
                    best_score = score;
                }
            }
            if (best_feature < 0 || !std::isfinite(best_score)) {
                ctx.set_error("SPA selection failed to find a finite candidate");
                out = SpaSelectionResult{};
                return N4M_ERR_INTERNAL;
            }
            selected.push_back(best_feature);
            is_selected[static_cast<std::size_t>(best_feature)] = 1U;
            selection_scores.push_back(best_score);
        }

        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.top_k = top_k;
        out.coefficient_scores = std::move(coefficient_scores);
        out.selection_scores = std::move(selection_scores);
        out.selected_indices = std::move(selected);
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running SPA selection");
        out = SpaSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running SPA selection");
        out = SpaSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
