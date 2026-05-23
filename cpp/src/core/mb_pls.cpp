// SPDX-License-Identifier: CECILL-2.1

#include "core/mb_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/common/svd.h"
#include "core/model.hpp"
#include "core/common/status.hpp"

// MB-PLS — Multi-block PLS, Westerhuis (1998) NIPALS variant.
//
// Default algorithm (cfg.scale_x == 0): per-block centering only, NIPALS
// super-score multi-block iteration matching
// `nirs4all.operators.models.sklearn.mbpls.MBPLS(standardize=False)`. This
// is the parity-gate sanctioned external reference.
//
// Legacy algorithm (cfg.scale_x != 0): per-block standardize + balance by
// 1/sqrt(block_width), then run single-block PLS on the concatenated stack.
// Kept as opt-in for callers who relied on the historical behaviour.

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

// ----------------------------------------------------------------------------
//  Legacy block-balanced SIMPLS preprocessing (kept behind cfg.scale_x != 0).
// ----------------------------------------------------------------------------

[[nodiscard]] n4m_status_t legacy_transform_blocks(
    ::n4m::core::Context& ctx,
    const n4m_matrix_view_t& X,
    const std::int64_t* block_sizes,
    std::size_t n_blocks,
    std::vector<double>& transformed,
    std::vector<double>& x_mean,
    std::vector<double>& x_scale,
    std::vector<double>& feature_weights,
    std::vector<double>& block_weights) {
    std::int64_t total_cols = 0;
    for (std::size_t block = 0; block < n_blocks; ++block) {
        if (block_sizes[block] <= 0) {
            ctx.set_error("MB-PLS block sizes must be positive");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (total_cols > std::numeric_limits<std::int64_t>::max() - block_sizes[block]) {
            ctx.set_error("MB-PLS block sizes overflow int64");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        total_cols += block_sizes[block];
    }
    if (total_cols != X.cols) {
        ctx.set_error("MB-PLS block sizes must sum to X columns");
        return N4M_ERR_SHAPE_MISMATCH;
    }

    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_error("MB-PLS transformed matrix size overflows size_t");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    resize_fill(transformed, total, 0.0);
    resize_fill(x_mean, cols, 0.0);
    resize_fill(x_scale, cols, 1.0);
    resize_fill(feature_weights, cols, 1.0);
    resize_fill(block_weights, n_blocks, 1.0);

    std::size_t start = 0;
    for (std::size_t block = 0; block < n_blocks; ++block) {
        const auto width = static_cast<std::size_t>(block_sizes[block]);
        const std::size_t stop = start + width;
        const double block_weight = 1.0 / std::sqrt(static_cast<double>(width));
        block_weights[block] = block_weight;
        for (std::size_t col = start; col < stop; ++col) {
            double sum = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                const double value = read_float(X, row, col);
                if (!std::isfinite(value)) {
                    ctx.set_error("X contains NaN or Inf");
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                sum += value;
            }
            const double mean = sum / static_cast<double>(rows);
            x_mean[col] = mean;
            double sumsq = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                const double delta = read_float(X, row, col) - mean;
                sumsq += delta * delta;
            }
            double scale = std::sqrt(sumsq / static_cast<double>(rows - 1U));
            if (scale == 0.0 || !std::isfinite(scale)) {
                scale = 1.0;
            }
            x_scale[col] = scale;
            feature_weights[col] = block_weight;
            for (std::size_t row = 0; row < rows; ++row) {
                transformed[idx(row, cols, col)] =
                    (read_float(X, row, col) - mean) / scale * block_weight;
            }
        }
        start = stop;
    }
    return N4M_OK;
}

// ----------------------------------------------------------------------------
//  Linear-algebra helpers for the nirs4all NIPALS multi-block path.
// ----------------------------------------------------------------------------

// In-place matrix multiply C = A @ B for row-major dense matrices.
void matmul(const double* A, std::size_t a_rows, std::size_t a_cols,
            const double* B, std::size_t b_cols,
            double* C) {
    for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t j = 0; j < b_cols; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < a_cols; ++k) {
                sum += A[i * a_cols + k] * B[k * b_cols + j];
            }
            C[i * b_cols + j] = sum;
        }
    }
}

// Compute Moore-Penrose pseudoinverse of a square k x k matrix via SVD.
// Mirrors numpy.linalg.pinv default rcond (~ max(M,N) * eps * largest sv).
[[nodiscard]] n4m_status_t square_pinv(const double* A, std::size_t k,
                                       std::vector<double>& Apinv) {
    if (k == 0) {
        Apinv.clear();
        return N4M_OK;
    }
    std::vector<double> work(A, A + k * k);
    std::vector<double> U(k * k, 0.0);
    std::vector<double> S(k, 0.0);
    std::vector<double> Vt(k * k, 0.0);
    const n4m_status_t st = n4m_svd_compact(work.data(),
                                            static_cast<std::int64_t>(k),
                                            static_cast<std::int64_t>(k),
                                            U.data(), S.data(), Vt.data());
    if (st != N4M_OK) {
        return st;
    }
    // numpy pinv default: rcond = max(M, N) * eps_largest_sv.
    double s_max = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        if (S[i] > s_max) {
            s_max = S[i];
        }
    }
    const double tol = static_cast<double>(k)
                       * std::numeric_limits<double>::epsilon()
                       * s_max;
    // Build V @ diag(1/S_clipped) @ U.T  (k x k).
    Apinv.assign(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            double sum = 0.0;
            for (std::size_t r = 0; r < k; ++r) {
                if (S[r] > tol) {
                    // V[i, r] = Vt[r, i]; Ut[r, j] = U[j, r].
                    sum += Vt[r * k + i] * U[j * k + r] / S[r];
                }
            }
            Apinv[i * k + j] = sum;
        }
    }
    return N4M_OK;
}

// ----------------------------------------------------------------------------
//  Default nirs4all-NIPALS multi-block algorithm (standardize=False).
// ----------------------------------------------------------------------------

[[nodiscard]] n4m_status_t fit_predict_nirs4all_nipals(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::int64_t* block_sizes,
    std::size_t n_blocks,
    ::n4m::core::MbPlsResult& out) {
    // ---- Geometry checks ------------------------------------------------
    std::int64_t total_cols = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        if (block_sizes[b] <= 0) {
            ctx.set_error("MB-PLS block sizes must be positive");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (total_cols > std::numeric_limits<std::int64_t>::max() - block_sizes[b]) {
            ctx.set_error("MB-PLS block sizes overflow int64");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        total_cols += block_sizes[b];
    }
    if (total_cols != X.cols) {
        ctx.set_error("MB-PLS block sizes must sum to X columns");
        return N4M_ERR_SHAPE_MISMATCH;
    }

    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);
    const auto targets = static_cast<std::size_t>(Y.cols);
    const auto K = static_cast<std::size_t>(cfg.n_components);

    if (K < 1) {
        ctx.set_error("MB-PLS requires at least one PLS component");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    // ---- Materialise X (row-major, contiguous) --------------------------
    std::vector<double> Xbuf;
    n4m_status_t status = copy_float_matrix(ctx, X, "X", Xbuf);
    if (status != N4M_OK) {
        return status;
    }
    std::vector<double> Ybuf;
    status = copy_float_matrix(ctx, Y, "Y", Ybuf);
    if (status != N4M_OK) {
        return status;
    }

    // ---- Compute per-block column means (no scaling) --------------------
    out.x_mean.assign(cols, 0.0);
    out.x_scale.assign(cols, 1.0);  // standardize=False parity
    out.block_weights.assign(n_blocks, 1.0);

    std::size_t start = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const auto width = static_cast<std::size_t>(block_sizes[b]);
        const std::size_t stop = start + width;
        for (std::size_t c = start; c < stop; ++c) {
            double sum = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                sum += Xbuf[r * cols + c];
            }
            out.x_mean[c] = sum / static_cast<double>(rows);
        }
        start = stop;
    }

    // ---- Center X in place (per-block but mean is column-wise, same op) -
    std::vector<double> X_res(rows * cols, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            X_res[r * cols + c] = Xbuf[r * cols + c] - out.x_mean[c];
        }
    }

    // ---- Compute y_mean (column-wise), center Y -------------------------
    std::vector<double> y_mean(targets, 0.0);
    for (std::size_t t = 0; t < targets; ++t) {
        double sum = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            sum += Ybuf[r * targets + t];
        }
        y_mean[t] = sum / static_cast<double>(rows);
    }
    std::vector<double> y_res(rows * targets, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < targets; ++t) {
            y_res[r * targets + t] = Ybuf[r * targets + t] - y_mean[t];
        }
    }

    // ---- NIPALS multi-block iteration -----------------------------------
    // W, P : (cols x K), Q : (targets x K).
    std::vector<double> W(cols * K, 0.0);
    std::vector<double> P(cols * K, 0.0);
    std::vector<double> Q(targets * K, 0.0);

    // Scratch buffers reused per component.
    std::vector<double> wb(cols, 0.0);          // concatenated block weight w
    std::vector<double> pb(cols, 0.0);          // concatenated block loading p
    std::vector<double> q(targets, 0.0);
    std::vector<double> t_super(rows, 0.0);
    std::vector<double> t_block(rows, 0.0);
    std::vector<double> t_acc(rows, 0.0);

    const double tiny = 1e-10;

    for (std::size_t comp = 0; comp < K; ++comp) {
        // Block-wise: compute w_b = X_res_b^T @ y_res[:, 0]; normalise;
        // t_b = X_res_b @ w_b. Then super score t = mean(t_b).
        std::fill(t_acc.begin(), t_acc.end(), 0.0);

        std::size_t blk_start = 0;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const auto width = static_cast<std::size_t>(block_sizes[b]);
            const std::size_t blk_stop = blk_start + width;

            // w_b[j] = sum_r X_res[r, j] * y_res[r, 0]   (first target only)
            // Length: width.
            double w_norm_sq = 0.0;
            for (std::size_t c = blk_start; c < blk_stop; ++c) {
                double acc = 0.0;
                for (std::size_t r = 0; r < rows; ++r) {
                    acc += X_res[r * cols + c] * y_res[r * targets + 0];
                }
                wb[c] = acc;
                w_norm_sq += acc * acc;
            }
            const double w_norm = std::sqrt(w_norm_sq);
            if (w_norm > tiny) {
                for (std::size_t c = blk_start; c < blk_stop; ++c) {
                    wb[c] /= w_norm;
                }
            }

            // t_b = X_res[:, blk_start:blk_stop] @ wb[blk_start:blk_stop]
            for (std::size_t r = 0; r < rows; ++r) {
                double acc = 0.0;
                for (std::size_t c = blk_start; c < blk_stop; ++c) {
                    acc += X_res[r * cols + c] * wb[c];
                }
                t_block[r] = acc;
                t_acc[r] += acc;
            }
            blk_start = blk_stop;
        }

        // Super score = mean(block_scores); then normalise.
        const double inv_nb = 1.0 / static_cast<double>(n_blocks);
        double t_norm_sq = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            t_super[r] = t_acc[r] * inv_nb;
            t_norm_sq += t_super[r] * t_super[r];
        }
        const double t_norm = std::sqrt(t_norm_sq);
        if (t_norm > tiny) {
            for (std::size_t r = 0; r < rows; ++r) {
                t_super[r] /= t_norm;
            }
        }
        // t_dot = t.T @ t + tiny (after normalisation, ~ 1 + tiny when t_norm > tiny).
        double t_dot = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            t_dot += t_super[r] * t_super[r];
        }
        t_dot += tiny;

        // Loadings per block: p_b = X_res_b^T @ t / t_dot.
        blk_start = 0;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const auto width = static_cast<std::size_t>(block_sizes[b]);
            const std::size_t blk_stop = blk_start + width;
            for (std::size_t c = blk_start; c < blk_stop; ++c) {
                double acc = 0.0;
                for (std::size_t r = 0; r < rows; ++r) {
                    acc += X_res[r * cols + c] * t_super[r];
                }
                pb[c] = acc / t_dot;
            }
            blk_start = blk_stop;
        }

        // y loading q = y_res^T @ t / t_dot  (shape: targets).
        for (std::size_t k_t = 0; k_t < targets; ++k_t) {
            double acc = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                acc += y_res[r * targets + k_t] * t_super[r];
            }
            q[k_t] = acc / t_dot;
        }

        // Store columns of W, P, Q.
        for (std::size_t c = 0; c < cols; ++c) {
            W[c * K + comp] = wb[c];
            P[c * K + comp] = pb[c];
        }
        for (std::size_t k_t = 0; k_t < targets; ++k_t) {
            Q[k_t * K + comp] = q[k_t];
        }

        // Deflate each block: X_res_b -= t @ p_b^T; deflate Y: y_res -= t @ q^T.
        blk_start = 0;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const auto width = static_cast<std::size_t>(block_sizes[b]);
            const std::size_t blk_stop = blk_start + width;
            for (std::size_t r = 0; r < rows; ++r) {
                const double tr = t_super[r];
                for (std::size_t c = blk_start; c < blk_stop; ++c) {
                    X_res[r * cols + c] -= tr * pb[c];
                }
            }
            blk_start = blk_stop;
        }
        for (std::size_t r = 0; r < rows; ++r) {
            const double tr = t_super[r];
            for (std::size_t k_t = 0; k_t < targets; ++k_t) {
                y_res[r * targets + k_t] -= tr * q[k_t];
            }
        }
    }

    // ---- B = W @ pinv(P^T @ W + tiny * I) @ Q^T -------------------------
    std::vector<double> PtW(K * K, 0.0);
    for (std::size_t i = 0; i < K; ++i) {
        for (std::size_t j = 0; j < K; ++j) {
            double acc = 0.0;
            for (std::size_t r = 0; r < cols; ++r) {
                acc += P[r * K + i] * W[r * K + j];
            }
            if (i == j) {
                acc += tiny;
            }
            PtW[i * K + j] = acc;
        }
    }
    std::vector<double> PtW_pinv;
    n4m_status_t pinv_st = square_pinv(PtW.data(), K, PtW_pinv);
    if (pinv_st != N4M_OK) {
        ctx.set_error("MB-PLS pinv(P^T W) failed");
        return pinv_st;
    }

    // tmp1 = W @ PtW_pinv  : (cols x K).
    std::vector<double> tmp1(cols * K, 0.0);
    matmul(W.data(), cols, K, PtW_pinv.data(), K, tmp1.data());

    // B = tmp1 @ Q^T  : (cols x targets).
    std::vector<double> B(cols * targets, 0.0);
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t t = 0; t < targets; ++t) {
            double acc = 0.0;
            for (std::size_t k_c = 0; k_c < K; ++k_c) {
                acc += tmp1[c * K + k_c] * Q[t * K + k_c];
            }
            B[c * targets + t] = acc;
        }
    }

    // ---- Predictions: Y_hat = (X - x_mean) @ B + y_mean -----------------
    out.predictions.assign(rows * targets, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < targets; ++t) {
            double acc = y_mean[t];
            for (std::size_t c = 0; c < cols; ++c) {
                acc += (Xbuf[r * cols + c] - out.x_mean[c]) * B[c * targets + t];
            }
            out.predictions[r * targets + t] = acc;
        }
    }

    // ---- Coefficients on the original X scale (no scaling, factor = 1) --
    out.coefficients = std::move(B);
    out.intercept.assign(targets, 0.0);
    for (std::size_t t = 0; t < targets; ++t) {
        double value = y_mean[t];
        for (std::size_t c = 0; c < cols; ++c) {
            value -= out.x_mean[c] * out.coefficients[c * targets + t];
        }
        out.intercept[t] = value;
    }

    out.n_samples = X.rows;
    out.n_features = static_cast<std::int32_t>(X.cols);
    out.n_targets = static_cast<std::int32_t>(Y.cols);
    out.n_components = cfg.n_components;
    out.n_blocks = static_cast<std::int32_t>(n_blocks);
    (void)cfg;  // tolerance / max_iter unused (closed-form NIPALS).
    return N4M_OK;
}

// ----------------------------------------------------------------------------
//  Legacy block-balanced standardised PLS (cfg.scale_x != 0).
// ----------------------------------------------------------------------------

[[nodiscard]] n4m_status_t fit_predict_legacy(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::int64_t* block_sizes,
    std::size_t n_blocks,
    ::n4m::core::MbPlsResult& out) {
    std::vector<double> transformed;
    std::vector<double> feature_weights;
    n4m_status_t status = legacy_transform_blocks(
        ctx, X, block_sizes, n_blocks,
        transformed, out.x_mean, out.x_scale,
        feature_weights, out.block_weights);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> y_values;
    status = copy_float_matrix(ctx, Y, "Y", y_values);
    if (status != N4M_OK) {
        return status;
    }

    ::n4m::core::Config pls_cfg = cfg;
    pls_cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    pls_cfg.deflation = N4M_DEFLATION_REGRESSION;
    pls_cfg.center_x = 1;
    pls_cfg.scale_x = 0;
    pls_cfg.center_y = 1;
    pls_cfg.scale_y = 0;
    n4m_matrix_view_t Xt = rowmajor_f64_view(transformed, X.rows, X.cols);
    n4m_matrix_view_t Yt = rowmajor_f64_view(y_values, Y.rows, Y.cols);
    std::unique_ptr<::n4m::core::Model> model;
    status = ::n4m::core::fit_model(ctx, pls_cfg, Xt, Yt, model);
    if (status != N4M_OK) {
        return status;
    }

    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);
    const auto targets = static_cast<std::size_t>(Y.cols);
    std::size_t pred_size = 0;
    std::size_t coef_size = 0;
    if (!checked_mul_size(rows, targets, pred_size) ||
        !checked_mul_size(cols, targets, coef_size)) {
        ctx.set_error("MB-PLS output size overflows size_t");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out.predictions.assign(pred_size, 0.0);
    n4m_matrix_view_t pred_view = rowmajor_f64_view(out.predictions, X.rows, Y.cols);
    status = ::n4m::core::predict_into(ctx, *model, Xt, pred_view);
    if (status != N4M_OK) {
        return status;
    }

    out.coefficients.assign(coef_size, 0.0);
    out.intercept.assign(targets, 0.0);
    for (std::size_t feature = 0; feature < cols; ++feature) {
        const double factor = feature_weights[feature] / out.x_scale[feature];
        for (std::size_t target = 0; target < targets; ++target) {
            out.coefficients[idx(feature, targets, target)] =
                model->coefficients[idx(feature, targets, target)] * factor;
        }
    }
    for (std::size_t target = 0; target < targets; ++target) {
        double value = model->y_mean[target];
        for (std::size_t feature = 0; feature < cols; ++feature) {
            value -= model->x_mean[feature] *
                     model->coefficients[idx(feature, targets, target)];
            value -= out.x_mean[feature] *
                     out.coefficients[idx(feature, targets, target)];
        }
        out.intercept[target] = value;
    }

    out.n_samples = X.rows;
    out.n_features = static_cast<std::int32_t>(X.cols);
    out.n_targets = static_cast<std::int32_t>(Y.cols);
    out.n_components = cfg.n_components;
    out.n_blocks = static_cast<std::int32_t>(n_blocks);
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t fit_predict_mb_pls(Context& ctx,
                                const Config& cfg,
                                const n4m_matrix_view_t& X,
                                const n4m_matrix_view_t& Y,
                                const std::int64_t* block_sizes,
                                std::size_t n_blocks,
                                MbPlsResult& out) {
    try {
        out = MbPlsResult{};
        if (block_sizes == nullptr || n_blocks == 0) {
            ctx.set_error("MB-PLS requires at least one block size");
            return N4M_ERR_INVALID_ARGUMENT;
        }
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
        if (X.rows < 2 || X.cols < 1) {
            ctx.set_error("MB-PLS requires at least 2 rows and 1 X column");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (Y.cols < 1) {
            ctx.set_error("Y must have at least one target column");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            n_blocks > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("MB-PLS dimensions exceed ABI limits");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("MB-PLS requires at least one PLS component");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        // Algorithm dispatch. Default: nirs4all NIPALS multi-block
        // (standardize=False). Opt-in legacy: cfg.scale_x != 0 selects the
        // historical block-balanced standardised path.
        if (cfg.scale_x != 0) {
            status = fit_predict_legacy(ctx, cfg, X, Y,
                                        block_sizes, n_blocks, out);
        } else {
            status = fit_predict_nirs4all_nipals(ctx, cfg, X, Y,
                                                 block_sizes, n_blocks, out);
        }
        if (status != N4M_OK) {
            out = MbPlsResult{};
            return status;
        }
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting MB-PLS");
        out = MbPlsResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting MB-PLS");
        out = MbPlsResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
