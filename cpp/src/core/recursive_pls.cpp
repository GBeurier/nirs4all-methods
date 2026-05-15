// SPDX-License-Identifier: CeCILL-2.1

#include "core/recursive_pls.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/model.hpp"

namespace pls4all::core {

namespace {

inline std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

p4a_matrix_view_t make_rowmajor_view(double* data,
                                      std::int64_t rows,
                                      std::int64_t cols) {
    p4a_matrix_view_t view{};
    view.data = data;
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

}  // namespace

p4a_status_t fit_predict_recursive_pls(Context& ctx,
                                        const Config& cfg,
                                        const p4a_matrix_view_t& X,
                                        const p4a_matrix_view_t& Y,
                                        std::int32_t window_size,
                                        RecursivePlsResult& out) {
    out = RecursivePlsResult{};
    if (validate_nonnull_view(X) != P4A_OK ||
        validate_nonnull_view(Y) != P4A_OK) {
        ctx.set_error("invalid matrix view");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must have the same row count");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (X.dtype != P4A_DTYPE_F64 || Y.dtype != P4A_DTYPE_F64) {
        ctx.set_error("recursive PLS requires F64 inputs");
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (window_size < 2) {
        ctx.set_errorf("window_size must be >= 2; got %d",
                       static_cast<int>(window_size));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1) {
        ctx.set_error("n_components must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (static_cast<std::int64_t>(window_size) > X.rows) {
        ctx.set_errorf("window_size (%d) must be <= X.rows (%lld)",
                       static_cast<int>(window_size),
                       static_cast<long long>(X.rows));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components >= window_size) {
        ctx.set_errorf("n_components (%d) must be < window_size (%d)",
                       static_cast<int>(cfg.n_components),
                       static_cast<int>(window_size));
        return P4A_ERR_INVALID_ARGUMENT;
    }

    try {
        const std::size_t n = static_cast<std::size_t>(X.rows);
        const std::size_t p = static_cast<std::size_t>(X.cols);
        const std::size_t q = static_cast<std::size_t>(Y.cols);
        const std::size_t w = static_cast<std::size_t>(window_size);
        out.n_samples = X.rows;
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.window_size = window_size;
        out.predictions.assign(n * q, 0.0);
        out.in_window.assign(n, 0);

        std::vector<double> x_window(w * p, 0.0);
        std::vector<double> y_window(w * q, 0.0);
        std::vector<double> query(p, 0.0);
        std::vector<double> query_pred(q, 0.0);

        const auto* x_ptr = static_cast<const double*>(X.data);
        const auto* y_ptr = static_cast<const double*>(Y.data);
        const std::size_t x_row_stride = static_cast<std::size_t>(X.row_stride);
        const std::size_t x_col_stride = static_cast<std::size_t>(X.col_stride);
        const std::size_t y_row_stride = static_cast<std::size_t>(Y.row_stride);
        const std::size_t y_col_stride = static_cast<std::size_t>(Y.col_stride);

        for (std::size_t i = w; i < n; ++i) {
            // Copy the previous `w` rows into the window buffers.
            for (std::size_t r = 0; r < w; ++r) {
                const std::size_t src = i - w + r;
                for (std::size_t col = 0; col < p; ++col) {
                    x_window[idx(r, p, col)] = x_ptr[src * x_row_stride +
                                                     col * x_col_stride];
                }
                for (std::size_t col = 0; col < q; ++col) {
                    y_window[idx(r, q, col)] = y_ptr[src * y_row_stride +
                                                     col * y_col_stride];
                }
            }
            for (std::size_t col = 0; col < p; ++col) {
                query[col] = x_ptr[i * x_row_stride + col * x_col_stride];
            }

            Config local = cfg;
            if (local.algorithm != P4A_ALGO_PCR &&
                local.algorithm != P4A_ALGO_PLS_REGRESSION) {
                local.algorithm = P4A_ALGO_PLS_REGRESSION;
            }
            if (local.solver != P4A_SOLVER_NIPALS &&
                local.solver != P4A_SOLVER_SIMPLS &&
                local.solver != P4A_SOLVER_SVD) {
                local.solver = P4A_SOLVER_SIMPLS;
            }
            local.deflation = P4A_DEFLATION_REGRESSION;

            p4a_matrix_view_t x_view = make_rowmajor_view(
                x_window.data(), static_cast<std::int64_t>(w),
                static_cast<std::int64_t>(p));
            p4a_matrix_view_t y_view = make_rowmajor_view(
                y_window.data(), static_cast<std::int64_t>(w),
                static_cast<std::int64_t>(q));
            std::unique_ptr<Model> model;
            const p4a_status_t status =
                fit_model(ctx, local, x_view, y_view, model);
            if (status != P4A_OK) {
                out = RecursivePlsResult{};
                return status;
            }
            p4a_matrix_view_t query_view = make_rowmajor_view(
                query.data(), 1, static_cast<std::int64_t>(p));
            p4a_matrix_view_t pred_view = make_rowmajor_view(
                query_pred.data(), 1, static_cast<std::int64_t>(q));
            const p4a_status_t predict_status =
                predict_into(ctx, *model, query_view, pred_view);
            if (predict_status != P4A_OK) {
                out = RecursivePlsResult{};
                return predict_status;
            }
            for (std::size_t col = 0; col < q; ++col) {
                out.predictions[idx(i, q, col)] = query_pred[col];
            }
            out.in_window[i] = 1;
        }

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory in recursive PLS");
        out = RecursivePlsResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception in recursive PLS");
        out = RecursivePlsResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
