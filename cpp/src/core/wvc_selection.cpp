// SPDX-License-Identifier: CeCILL-2.1

#include "core/wvc_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
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

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& view,
                                               const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(view);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (view.dtype != P4A_DTYPE_F64 && view.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_request(::pls4all::core::Context& ctx,
                                            const p4a_matrix_view_t& X,
                                            const p4a_matrix_view_t& Y,
                                            std::int32_t n_components,
                                            std::int32_t top_k) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows < 2 || X.cols < 2 || Y.cols != 1) {
        ctx.set_error("WVC-PLS requires at least 2 rows, 2 X columns and one numeric Y column");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (n_components < 1 || static_cast<std::int64_t>(n_components) >= X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld); got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(n_components));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (top_k < 1 || static_cast<std::int64_t>(top_k) > X.cols) {
        ctx.set_errorf("top_k must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(top_k));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("WVC-PLS dimensions exceed int32 result storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_matrix(::pls4all::core::Context& ctx,
                                       const p4a_matrix_view_t& view,
                                       const char* name,
                                       std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_matrix_size(view.rows, view.cols, n_values)) {
        ctx.set_errorf("%s matrix shape is too large", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(view, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return P4A_OK;
}

void center_scale_columns(std::vector<double>& values,
                          std::size_t rows,
                          std::size_t cols) {
    for (std::size_t col = 0; col < cols; ++col) {
        double mean = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            mean += values[idx(row, cols, col)];
        }
        mean /= static_cast<double>(rows);
        double sumsq = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double centered = values[idx(row, cols, col)] - mean;
            values[idx(row, cols, col)] = centered;
            sumsq += centered * centered;
        }
        double scale = rows > 1U ? std::sqrt(sumsq / static_cast<double>(rows - 1U)) : 1.0;
        if (!std::isfinite(scale) || scale == 0.0) {
            scale = 1.0;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            values[idx(row, cols, col)] /= scale;
        }
    }
}

[[nodiscard]] double dot(const std::vector<double>& lhs,
                         const std::vector<double>& rhs) noexcept {
    double out = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        out += lhs[i] * rhs[i];
    }
    return out;
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_wvc(Context& ctx,
                           const p4a_matrix_view_t& X,
                           const p4a_matrix_view_t& Y,
                           std::int32_t n_components,
                           std::int32_t top_k,
                           bool normalize,
                           WvcSelectionResult& out) {
    try {
        out = WvcSelectionResult{};
        p4a_status_t status = validate_request(ctx, X, Y, n_components, top_k);
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> xwork;
        std::vector<double> ywork;
        status = copy_matrix(ctx, X, "X", xwork);
        if (status != P4A_OK) {
            return status;
        }
        status = copy_matrix(ctx, Y, "Y", ywork);
        if (status != P4A_OK) {
            return status;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const auto comps = static_cast<std::size_t>(n_components);
        center_scale_columns(xwork, rows, cols);

        out.weights_w.assign(cols * comps, 0.0);
        out.loadings_p.assign(cols * comps, 0.0);
        out.y_loadings_q.assign(comps, 0.0);
        out.wvc_scores.assign(cols * comps, 0.0);
        std::vector<double> ee(comps, 0.0);
        std::vector<double> ss(comps, 0.0);

        for (std::size_t comp = 0; comp < comps; ++comp) {
            std::vector<double> w(cols, 0.0);
            for (std::size_t col = 0; col < cols; ++col) {
                for (std::size_t row = 0; row < rows; ++row) {
                    w[col] += xwork[idx(row, cols, col)] * ywork[row];
                }
            }
            double w_norm = std::sqrt(dot(w, w));
            if (!(w_norm > std::numeric_limits<double>::epsilon())) {
                ctx.set_error("WVC-PLS covariance vector collapsed");
                out = WvcSelectionResult{};
                return P4A_ERR_INVALID_ARGUMENT;
            }
            for (double& value : w) {
                value /= w_norm;
            }
            std::size_t sign_idx = 0;
            double sign_abs = std::fabs(w[0]);
            for (std::size_t col = 1; col < cols; ++col) {
                const double candidate = std::fabs(w[col]);
                if (candidate > sign_abs) {
                    sign_abs = candidate;
                    sign_idx = col;
                }
            }
            if (w[sign_idx] < 0.0) {
                for (double& value : w) {
                    value = -value;
                }
            }

            std::vector<double> score(rows, 0.0);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    score[row] += xwork[idx(row, cols, col)] * w[col];
                }
            }
            const double score_ss = dot(score, score);
            if (!(score_ss > std::numeric_limits<double>::epsilon())) {
                ctx.set_error("WVC-PLS score vector collapsed");
                out = WvcSelectionResult{};
                return P4A_ERR_INVALID_ARGUMENT;
            }

            std::vector<double> pvec(cols, 0.0);
            for (std::size_t col = 0; col < cols; ++col) {
                for (std::size_t row = 0; row < rows; ++row) {
                    pvec[col] += xwork[idx(row, cols, col)] * score[row];
                }
                pvec[col] /= score_ss;
            }
            double q = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                q += ywork[row] * score[row];
            }
            q /= score_ss;
            double sy = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                sy += score[row] * ywork[row];
            }
            ss[comp] = sy * q;
            ee[comp] = ss[comp] / score_ss;
            for (std::size_t col = 0; col < cols; ++col) {
                out.weights_w[idx(col, comps, comp)] = w[col];
                out.loadings_p[idx(col, comps, comp)] = pvec[col];
            }
            out.y_loadings_q[comp] = q;

            double denominator = 0.0;
            for (std::size_t prev = 0; prev <= comp; ++prev) {
                denominator += ee[prev] * ss[prev];
            }
            if (!(std::fabs(denominator) > std::numeric_limits<double>::epsilon())) {
                ctx.set_error("WVC-PLS denominator collapsed");
                out = WvcSelectionResult{};
                return P4A_ERR_INVALID_ARGUMENT;
            }
            double max_wvc = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                double numerator = 0.0;
                for (std::size_t prev = 0; prev <= comp; ++prev) {
                    const double weight = out.weights_w[idx(col, comps, prev)];
                    numerator += ee[prev] * weight * weight * ss[prev];
                }
                const double value =
                    std::sqrt(std::max(static_cast<double>(cols) * numerator / denominator, 0.0));
                out.wvc_scores[idx(col, comps, comp)] = value;
                max_wvc = std::max(max_wvc, value);
            }
            if (normalize && max_wvc > std::numeric_limits<double>::epsilon()) {
                for (std::size_t col = 0; col < cols; ++col) {
                    out.wvc_scores[idx(col, comps, comp)] /= max_wvc;
                }
            }

            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    xwork[idx(row, cols, col)] -= score[row] * pvec[col];
                }
                ywork[row] -= score[row] * q;
            }
        }

        out.final_scores.assign(cols, 0.0);
        for (std::size_t col = 0; col < cols; ++col) {
            out.final_scores[col] = out.wvc_scores[idx(col, comps, comps - 1U)];
        }
        std::vector<std::int64_t> order(cols, 0);
        std::iota(order.begin(), order.end(), static_cast<std::int64_t>(0));
        std::stable_sort(order.begin(), order.end(), [&out](std::int64_t lhs, std::int64_t rhs) {
            const double left = out.final_scores[static_cast<std::size_t>(lhs)];
            const double right = out.final_scores[static_cast<std::size_t>(rhs)];
            if (left == right) {
                return lhs < rhs;
            }
            return left > right;
        });
        out.selected_indices.assign(order.begin(), order.begin() + top_k);

        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = n_components;
        out.top_k = top_k;
        out.normalize = normalize ? 1 : 0;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running WVC-PLS selection");
        out = WvcSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running WVC-PLS selection");
        out = WvcSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t select_by_wvc_threshold(Context& ctx,
                                     const p4a_matrix_view_t& X,
                                     const p4a_matrix_view_t& Y,
                                     std::int32_t n_components,
                                     bool normalize,
                                     double score_threshold,
                                     double threshold_factor,
                                     std::int32_t min_selected,
                                     WvcThresholdSelectionResult& out) {
    try {
        out = WvcThresholdSelectionResult{};
        p4a_status_t status = validate_request(ctx, X, Y, n_components, 1);
        if (status != P4A_OK) {
            return status;
        }
        if (!std::isfinite(score_threshold) || score_threshold < 0.0) {
            ctx.set_error("WVC threshold score_threshold must be finite and non-negative");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (!std::isfinite(threshold_factor) || threshold_factor < 0.0) {
            ctx.set_error("WVC threshold_factor must be finite and non-negative");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (min_selected < 1 || static_cast<std::int64_t>(min_selected) > X.cols) {
            ctx.set_errorf("min_selected must be in [1, %lld]; got %d",
                           static_cast<long long>(X.cols),
                           static_cast<int>(min_selected));
            return P4A_ERR_INVALID_ARGUMENT;
        }

        WvcSelectionResult base;
        status = select_by_wvc(ctx,
                               X,
                               Y,
                               n_components,
                               static_cast<std::int32_t>(X.cols),
                               normalize,
                               base);
        if (status != P4A_OK) {
            out = WvcThresholdSelectionResult{};
            return status;
        }

        out.final_scores = base.final_scores;
        out.ranked_indices = base.selected_indices;
        double sum = 0.0;
        for (const double value : out.final_scores) {
            sum += value;
        }
        out.mean_score = sum / static_cast<double>(out.final_scores.size());
        out.score_threshold = score_threshold;
        out.threshold_factor = threshold_factor;
        out.effective_threshold = std::max(score_threshold, threshold_factor * out.mean_score);

        for (const std::int64_t feature : out.ranked_indices) {
            const double score = out.final_scores[static_cast<std::size_t>(feature)];
            if (score >= out.effective_threshold) {
                out.selected_indices.push_back(feature);
            }
        }
        if (out.selected_indices.size() < static_cast<std::size_t>(min_selected)) {
            out.selected_indices.assign(out.ranked_indices.begin(),
                                        out.ranked_indices.begin() + min_selected);
        }

        out.n_features = base.n_features;
        out.n_targets = base.n_targets;
        out.n_components = n_components;
        out.min_selected = min_selected;
        out.normalize = normalize ? 1 : 0;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running thresholded WVC-PLS selection");
        out = WvcThresholdSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running thresholded WVC-PLS selection");
        out = WvcThresholdSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
