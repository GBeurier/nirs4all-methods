// SPDX-License-Identifier: CECILL-2.1

#include "core/lw_pls.hpp"

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

[[nodiscard]] bool checked_mul_size(std::size_t a,
                                    std::size_t b,
                                    std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

[[nodiscard]] double read_float(const n4m_matrix_view_t& view,
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

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
}

[[nodiscard]] n4m_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
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

[[nodiscard]] n4m_status_t copy_float_matrix(::n4m::core::Context& ctx,
                                             const n4m_matrix_view_t& view,
                                             const char* name,
                                             std::vector<double>& out) {
    const auto rows = static_cast<std::size_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_errorf("%s matrix size overflows size_t", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    resize_fill(out, total, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_float(view, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t standardize_for_distance(::n4m::core::Context& ctx,
                                                    const std::vector<double>& X,
                                                    std::size_t rows,
                                                    std::size_t cols,
                                                    std::vector<double>& Xs) {
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_error("LW-PLS distance matrix size overflows size_t");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    resize_fill(Xs, total, 0.0);
    std::vector<double> mean(cols, 0.0);
    std::vector<double> scale(cols, 1.0);
    for (std::size_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += X[idx(row, cols, col)];
        }
        mean[col] = sum / static_cast<double>(rows);
        double sumsq = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double delta = X[idx(row, cols, col)] - mean[col];
            sumsq += delta * delta;
        }
        double stddev = std::sqrt(sumsq / static_cast<double>(rows - 1U));
        if (stddev == 0.0 || !std::isfinite(stddev)) {
            stddev = 1.0;
        }
        scale[col] = stddev;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            Xs[idx(row, cols, col)] = (X[idx(row, cols, col)] - mean[col]) / scale[col];
        }
    }
    return N4M_OK;
}

// ---------------------------------------------------------------------------
// Gaussian-weighted local PLS (default mode).
//
// This mirrors the nirs4all reference implementation
// (operators/models/sklearn/lwpls.py::_lwpls_predict) exactly:
//   - Euclidean distances are computed on the *raw* training rows (the caller
//     is expected to standardize X / y upstream, just like sklearn's
//     StandardScaler does in the reference). pls4all does not apply an
//     internal autoscaling here because the reference uses scale=False.
//   - The kernel bandwidth `lambda` is derived from the caller-supplied
//     `n_neighbors` parameter the same way the registry adapter does it:
//     `lambda = max(1.0, 0.5 * n_neighbors)`.
//   - Per query row: weights = exp(-d_i / std(d, ddof=1) / lambda), and a
//     weighted PLS regression is fit using all training points. The
//     prediction is then the accumulated `t_q * q_a` contribution after
//     `n_components` components.
// ---------------------------------------------------------------------------
[[nodiscard]] n4m_status_t fit_predict_lw_pls_weighted(
    ::n4m::core::Context& /*ctx*/,
    const ::n4m::core::Config& cfg,
    const std::vector<double>& x_values,
    const std::vector<double>& y_values,
    std::size_t rows,
    std::size_t cols,
    std::size_t targets,
    std::int32_t n_neighbors,
    ::n4m::core::LwPlsResult& out) {
    const auto n_components = static_cast<std::size_t>(cfg.n_components);
    const double lambda = std::max(
        1.0, 0.5 * static_cast<double>(n_neighbors));

    // Precompute pairwise Euclidean distance row blocks lazily (cheap given
    // we already touch every row in the outer loop).
    std::vector<double> distances(rows, 0.0);
    std::vector<double> weights(rows, 0.0);
    std::vector<double> centered_y(rows, 0.0);
    std::vector<double> centered_x(rows * cols, 0.0);
    std::vector<double> centered_query(cols, 0.0);
    std::vector<double> x_w(cols, 0.0);
    std::vector<double> w_a(cols, 0.0);
    std::vector<double> t_a(rows, 0.0);
    std::vector<double> p_a(cols, 0.0);

    for (std::size_t row = 0; row < rows; ++row) {
        // 1. Distances and Gaussian weights based on the query x_values[row].
        for (std::size_t other = 0; other < rows; ++other) {
            double d = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                const double delta = x_values[idx(other, cols, col)] -
                                     x_values[idx(row, cols, col)];
                d += delta * delta;
            }
            distances[other] = std::sqrt(d);
        }

        double dmean = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            dmean += distances[i];
        }
        dmean /= static_cast<double>(rows);
        double dvar = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            const double delta = distances[i] - dmean;
            dvar += delta * delta;
        }
        double dstd = (rows > 1U)
                          ? std::sqrt(dvar / static_cast<double>(rows - 1U))
                          : 0.0;
        if (!(dstd > 0.0) || !std::isfinite(dstd)) {
            dstd = 1.0;
        }

        double w_sum = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            weights[i] = std::exp(-distances[i] / dstd / lambda);
            w_sum += weights[i];
        }
        if (!(w_sum > 1e-10) || !std::isfinite(w_sum)) {
            // Reference uses uniform weights summing to 1 in this degenerate
            // case. Match it exactly.
            const double uniform = 1.0 / static_cast<double>(rows);
            for (std::size_t i = 0; i < rows; ++i) {
                weights[i] = uniform;
            }
            w_sum = 1.0;
        }

        // 2. Per-target weighted PLS regression.
        for (std::size_t tcol = 0; tcol < targets; ++tcol) {
            // Weighted mean of y for this target.
            double y_w = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                y_w += weights[i] * y_values[idx(i, targets, tcol)];
            }
            y_w /= w_sum;

            // Weighted mean of x (vector of size cols).
            for (std::size_t col = 0; col < cols; ++col) {
                x_w[col] = 0.0;
            }
            for (std::size_t i = 0; i < rows; ++i) {
                const double wi = weights[i];
                for (std::size_t col = 0; col < cols; ++col) {
                    x_w[col] += wi * x_values[idx(i, cols, col)];
                }
            }
            for (std::size_t col = 0; col < cols; ++col) {
                x_w[col] /= w_sum;
            }

            // Centered training x and y, centered query.
            for (std::size_t i = 0; i < rows; ++i) {
                centered_y[i] =
                    y_values[idx(i, targets, tcol)] - y_w;
                for (std::size_t col = 0; col < cols; ++col) {
                    centered_x[idx(i, cols, col)] =
                        x_values[idx(i, cols, col)] - x_w[col];
                }
            }
            for (std::size_t col = 0; col < cols; ++col) {
                centered_query[col] =
                    x_values[idx(row, cols, col)] - x_w[col];
            }

            // Prediction starts at the weighted mean.
            double prediction = y_w;

            for (std::size_t comp = 0; comp < n_components; ++comp) {
                // numerator = X.T @ (w * y)
                double norm_sq = 0.0;
                for (std::size_t col = 0; col < cols; ++col) {
                    double acc = 0.0;
                    for (std::size_t i = 0; i < rows; ++i) {
                        acc += centered_x[idx(i, cols, col)] *
                               (weights[i] * centered_y[i]);
                    }
                    w_a[col] = acc;
                    norm_sq += acc * acc;
                }
                const double norm_val = std::sqrt(norm_sq);
                if (!(norm_val > 1e-10) || !std::isfinite(norm_val)) {
                    break;
                }
                for (std::size_t col = 0; col < cols; ++col) {
                    w_a[col] /= norm_val;
                }

                // t_a = X @ w_a
                for (std::size_t i = 0; i < rows; ++i) {
                    double acc = 0.0;
                    for (std::size_t col = 0; col < cols; ++col) {
                        acc += centered_x[idx(i, cols, col)] * w_a[col];
                    }
                    t_a[i] = acc;
                }

                // denom = sum(w * t^2)
                double denom = 0.0;
                for (std::size_t i = 0; i < rows; ++i) {
                    denom += weights[i] * t_a[i] * t_a[i];
                }
                if (!(denom > 1e-10) || !std::isfinite(denom)) {
                    break;
                }

                // p_a = (X.T @ (w * t)) / denom
                for (std::size_t col = 0; col < cols; ++col) {
                    double acc = 0.0;
                    for (std::size_t i = 0; i < rows; ++i) {
                        acc += centered_x[idx(i, cols, col)] *
                               (weights[i] * t_a[i]);
                    }
                    p_a[col] = acc / denom;
                }

                // q_a = (y.T @ (w * t)) / denom
                double q_a = 0.0;
                for (std::size_t i = 0; i < rows; ++i) {
                    q_a += centered_y[i] * weights[i] * t_a[i];
                }
                q_a /= denom;

                // Query score t_q = <centered_query, w_a>
                double t_q = 0.0;
                for (std::size_t col = 0; col < cols; ++col) {
                    t_q += centered_query[col] * w_a[col];
                }

                prediction += t_q * q_a;

                // Deflate for next component, skipping after the last one.
                if (comp + 1U < n_components) {
                    for (std::size_t i = 0; i < rows; ++i) {
                        const double ti = t_a[i];
                        for (std::size_t col = 0; col < cols; ++col) {
                            centered_x[idx(i, cols, col)] -= ti * p_a[col];
                        }
                        centered_y[i] -= ti * q_a;
                    }
                    for (std::size_t col = 0; col < cols; ++col) {
                        centered_query[col] -= t_q * p_a[col];
                    }
                }
            }

            out.predictions[idx(row, targets, tcol)] = prediction;
        }
    }

    return N4M_OK;
}

// ---------------------------------------------------------------------------
// Legacy k-NN cutoff variant (opt-in via cfg.solver == N4M_SOLVER_NIPALS-with
// flag set elsewhere — we keep the implementation guard-free here and let the
// caller request it explicitly through the new dispatch logic).
// ---------------------------------------------------------------------------
[[nodiscard]] n4m_status_t fit_predict_lw_pls_knn(::n4m::core::Context& ctx,
                                                  const ::n4m::core::Config& cfg,
                                                  const std::vector<double>& x_values,
                                                  const std::vector<double>& y_values,
                                                  std::size_t rows,
                                                  std::size_t cols,
                                                  std::size_t targets,
                                                  const n4m_matrix_view_t& X_view,
                                                  const n4m_matrix_view_t& Y_view,
                                                  std::int32_t n_neighbors,
                                                  ::n4m::core::LwPlsResult& out) {
    const auto neighbors = static_cast<std::size_t>(n_neighbors);
    std::vector<double> standardized;
    const n4m_status_t status = standardize_for_distance(ctx, x_values, rows, cols, standardized);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<std::pair<double, std::int64_t>> order(rows);
    std::vector<double> local_x(neighbors * cols, 0.0);
    std::vector<double> local_y(neighbors * targets, 0.0);
    std::vector<double> query(cols, 0.0);
    std::vector<double> query_pred(targets, 0.0);

    ::n4m::core::Config local_cfg = cfg;
    local_cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    local_cfg.deflation = N4M_DEFLATION_REGRESSION;
    local_cfg.center_x = 1;
    local_cfg.scale_x = 1;
    local_cfg.center_y = 1;
    local_cfg.scale_y = 1;

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t other = 0; other < rows; ++other) {
            double distance = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                const double delta =
                    standardized[idx(other, cols, col)] -
                    standardized[idx(row, cols, col)];
                distance += delta * delta;
            }
            order[other] = {distance, static_cast<std::int64_t>(other)};
        }
        std::sort(order.begin(), order.end());
        for (std::size_t nidx = 0; nidx < neighbors; ++nidx) {
            const auto selected = static_cast<std::size_t>(order[nidx].second);
            out.neighbor_indices[idx(row, neighbors, nidx)] = order[nidx].second;
            for (std::size_t col = 0; col < cols; ++col) {
                local_x[idx(nidx, cols, col)] = x_values[idx(selected, cols, col)];
            }
            for (std::size_t target = 0; target < targets; ++target) {
                local_y[idx(nidx, targets, target)] =
                    y_values[idx(selected, targets, target)];
            }
        }
        for (std::size_t col = 0; col < cols; ++col) {
            query[col] = x_values[idx(row, cols, col)];
        }

        n4m_matrix_view_t local_x_view =
            rowmajor_f64_view(local_x, static_cast<std::int64_t>(neighbors), X_view.cols);
        n4m_matrix_view_t local_y_view =
            rowmajor_f64_view(local_y, static_cast<std::int64_t>(neighbors), Y_view.cols);
        std::unique_ptr<::n4m::core::Model> model;
        n4m_status_t fit_status = ::n4m::core::fit_model(ctx, local_cfg, local_x_view, local_y_view, model);
        if (fit_status != N4M_OK) {
            return fit_status;
        }
        n4m_matrix_view_t query_view = rowmajor_f64_view(query, 1, X_view.cols);
        n4m_matrix_view_t pred_view = rowmajor_f64_view(query_pred, 1, Y_view.cols);
        fit_status = ::n4m::core::predict_into(ctx, *model, query_view, pred_view);
        if (fit_status != N4M_OK) {
            return fit_status;
        }
        for (std::size_t target = 0; target < targets; ++target) {
            out.predictions[idx(row, targets, target)] = query_pred[target];
        }
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t fit_predict_lw_pls(Context& ctx,
                               const Config& cfg,
                               const n4m_matrix_view_t& X,
                               const n4m_matrix_view_t& Y,
                               std::int32_t n_neighbors,
                               LwPlsResult& out) {
    try {
        out = LwPlsResult{};
        n4m_status_t status = validate_float_view(ctx, X, "X");
        if (status != N4M_OK) {
            return status;
        }
        status = validate_float_view(ctx, Y, "Y");
        if (status != N4M_OK) {
            return status;
        }
        if (X.rows != Y.rows) {
            ctx.set_error("X and Y row counts must match");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        if (X.rows < 2 || X.cols < 1 || Y.cols < 1) {
            ctx.set_error("LW-PLS requires at least 2 rows, 1 X column and 1 Y column");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("LW-PLS requires at least one PLS component");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (n_neighbors < cfg.n_components + 1 || n_neighbors > X.rows) {
            ctx.set_error("LW-PLS n_neighbors must be in [n_components + 1, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("LW-PLS dimensions exceed ABI limits");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const auto targets = static_cast<std::size_t>(Y.cols);
        const auto neighbors = static_cast<std::size_t>(n_neighbors);
        std::size_t pred_size = 0;
        std::size_t neighbor_size = 0;
        if (!checked_mul_size(rows, targets, pred_size) ||
            !checked_mul_size(rows, neighbors, neighbor_size)) {
            ctx.set_error("LW-PLS output size overflows size_t");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> x_values;
        std::vector<double> y_values;
        status = copy_float_matrix(ctx, X, "X", x_values);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_float_matrix(ctx, Y, "Y", y_values);
        if (status != N4M_OK) {
            return status;
        }

        out.predictions.assign(pred_size, 0.0);
        out.neighbor_indices.assign(neighbor_size, 0);

        // Default to the Gaussian-weighted local PLS that matches the nirs4all
        // reference. The legacy k-NN cutoff variant remains available as an
        // opt-in via cfg.solver == N4M_SOLVER_SIMPLS (the reference uses a
        // NIPALS-style weighted recurrence, so SIMPLS is the only solver flag
        // we can repurpose without growing the C ABI surface).
        const bool use_knn = (cfg.solver == N4M_SOLVER_SIMPLS);
        if (use_knn) {
            status = fit_predict_lw_pls_knn(
                ctx, cfg, x_values, y_values, rows, cols, targets,
                X, Y, n_neighbors, out);
        } else {
            // In the weighted mode we still expose the requested neighbor
            // count via the result so downstream code (and the C ABI handle)
            // keeps a stable shape; the indices are simply the rows sorted by
            // raw Euclidean distance to the query, which preserves the
            // documented contract.
            std::vector<std::pair<double, std::int64_t>> order(rows);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t other = 0; other < rows; ++other) {
                    double d = 0.0;
                    for (std::size_t col = 0; col < cols; ++col) {
                        const double delta =
                            x_values[idx(other, cols, col)] -
                            x_values[idx(row, cols, col)];
                        d += delta * delta;
                    }
                    order[other] = {d, static_cast<std::int64_t>(other)};
                }
                std::sort(order.begin(), order.end());
                for (std::size_t nidx = 0; nidx < neighbors; ++nidx) {
                    out.neighbor_indices[idx(row, neighbors, nidx)] =
                        order[nidx].second;
                }
            }
            status = fit_predict_lw_pls_weighted(
                ctx, cfg, x_values, y_values, rows, cols, targets,
                n_neighbors, out);
        }
        if (status != N4M_OK) {
            out = LwPlsResult{};
            return status;
        }

        out.n_samples = X.rows;
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_neighbors = n_neighbors;
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting LW-PLS");
        out = LwPlsResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting LW-PLS");
        out = LwPlsResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
