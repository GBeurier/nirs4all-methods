// SPDX-License-Identifier: CECILL-2.1

#include "core/pls_lda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] bool checked_mul_i64(std::int64_t a,
                                   std::int64_t b,
                                   std::int64_t& out) noexcept {
    if (a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
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

[[nodiscard]] double read_numeric(const p4a_matrix_view_t& view,
                                  std::size_t row,
                                  std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    switch (view.dtype) {
        case P4A_DTYPE_F64: {
            const auto* ptr = static_cast<const double*>(view.data);
            return ptr[uoff];
        }
        case P4A_DTYPE_F32: {
            const auto* ptr = static_cast<const float*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case P4A_DTYPE_I32: {
            const auto* ptr = static_cast<const std::int32_t*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case P4A_DTYPE_I64: {
            const auto* ptr = static_cast<const std::int64_t*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case P4A_DTYPE_UNKNOWN:
            return 0.0;
    }
    return 0.0;
}

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
}

[[nodiscard]] bool invert_square(std::vector<double> a,
                                 std::size_t n,
                                 std::vector<double>& inverse) {
    resize_fill(inverse, n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        inverse[idx(i, n, i)] = 1.0;
    }
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(a[idx(col, n, col)]);
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double candidate = std::fabs(a[idx(row, n, col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= 0.0) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[idx(col, n, j)], a[idx(pivot, n, j)]);
                std::swap(inverse[idx(col, n, j)], inverse[idx(pivot, n, j)]);
            }
        }
        const double diag = a[idx(col, n, col)];
        for (std::size_t j = 0; j < n; ++j) {
            a[idx(col, n, j)] /= diag;
            inverse[idx(col, n, j)] /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[idx(row, n, col)];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                a[idx(row, n, j)] -= factor * a[idx(col, n, j)];
                inverse[idx(row, n, j)] -= factor * inverse[idx(col, n, j)];
            }
        }
    }
    return true;
}

[[nodiscard]] p4a_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    p4a_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

[[nodiscard]] p4a_status_t validate_labels(::pls4all::core::Context& ctx,
                                           const p4a_matrix_view_t& labels,
                                           std::int64_t n_samples,
                                           std::int32_t n_classes,
                                           std::vector<std::int32_t>& out) {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(labels);
    if (status != P4A_OK) {
        ctx.set_errorf("labels matrix view is invalid: %s",
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    std::int64_t label_count = 0;
    if (!checked_mul_i64(labels.rows, labels.cols, label_count)) {
        ctx.set_error("label count overflows int64");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (label_count != n_samples) {
        ctx.set_error("label count must match X rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    out.clear();
    out.reserve(static_cast<std::size_t>(label_count));
    for (std::int64_t row = 0; row < labels.rows; ++row) {
        for (std::int64_t col = 0; col < labels.cols; ++col) {
            const double value = read_numeric(labels,
                                              static_cast<std::size_t>(row),
                                              static_cast<std::size_t>(col));
            if (!std::isfinite(value) || std::floor(value) != value ||
                value < 0.0 || value >= static_cast<double>(n_classes)) {
                ctx.set_errorf("labels must be integer class ids in [0, %d)",
                               static_cast<int>(n_classes));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out.push_back(static_cast<std::int32_t>(value));
        }
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t fit_predict_pls_lda(Context& ctx,
                                const Config& cfg,
                                const p4a_matrix_view_t& X,
                                const p4a_matrix_view_t& labels,
                                std::int32_t n_classes,
                                PlsLdaResult& out) {
    try {
        out = PlsLdaResult{};
        const p4a_status_t x_status = validate_nonnull_view(X);
        if (x_status != P4A_OK) {
            ctx.set_errorf("X matrix view is invalid: %s", status_to_string(x_status));
            return x_status;
        }
        if (X.dtype != P4A_DTYPE_F64 && X.dtype != P4A_DTYPE_F32) {
            ctx.set_error("X dtype must be f64 or f32");
            return P4A_ERR_DTYPE_MISMATCH;
        }
        if (X.rows <= 0 || X.cols <= 0) {
            ctx.set_error("X matrix must be non-empty");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (n_classes < 2) {
            ctx.set_errorf("n_classes must be >= 2; got %d", static_cast<int>(n_classes));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("PLS-LDA requires at least one PLS component");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        std::vector<std::int32_t> y_labels;
        p4a_status_t status = validate_labels(ctx, labels, X.rows, n_classes, y_labels);
        if (status != P4A_OK) {
            return status;
        }
        const auto n = static_cast<std::size_t>(X.rows);
        const auto k = static_cast<std::size_t>(cfg.n_components);
        const auto c = static_cast<std::size_t>(n_classes);
        std::size_t n_times_c = 0;
        std::size_t n_times_k = 0;
        std::size_t c_times_k = 0;
        std::size_t k_times_k = 0;
        if (!checked_mul_size(n, c, n_times_c) ||
            !checked_mul_size(n, k, n_times_k) ||
            !checked_mul_size(c, k, c_times_k) ||
            !checked_mul_size(k, k, k_times_k)) {
            ctx.set_error("PLS-LDA working size overflows size_t");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        std::vector<std::int64_t> class_counts(c, 0);
        std::vector<double> dummy_y(n_times_c, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            const auto cls = static_cast<std::size_t>(y_labels[row]);
            class_counts[cls] += 1;
            dummy_y[idx(row, c, cls)] = 1.0;
        }
        for (std::size_t cls = 0; cls < c; ++cls) {
            if (class_counts[cls] == 0) {
                ctx.set_errorf("class %llu has no labels",
                               static_cast<unsigned long long>(cls));
                return P4A_ERR_INVALID_ARGUMENT;
            }
        }
        if (X.rows <= static_cast<std::int64_t>(n_classes)) {
            ctx.set_error("PLS-LDA requires more samples than classes");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        Config pls_cfg = cfg;
        pls_cfg.algorithm = P4A_ALGO_PLS_DA;
        pls_cfg.deflation = P4A_DEFLATION_REGRESSION;
        p4a_matrix_view_t Y = rowmajor_f64_view(dummy_y, X.rows, n_classes);
        std::unique_ptr<Model> pls_model;
        status = fit_model(ctx, pls_cfg, X, Y, pls_model);
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> scores(n_times_k, 0.0);
        p4a_matrix_view_t score_view = rowmajor_f64_view(scores,
                                                         X.rows,
                                                         cfg.n_components);
        status = transform_into(ctx, *pls_model, X, score_view);
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> means(c_times_k, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            const auto cls = static_cast<std::size_t>(y_labels[row]);
            for (std::size_t comp = 0; comp < k; ++comp) {
                means[idx(cls, k, comp)] += scores[idx(row, k, comp)];
            }
        }
        for (std::size_t cls = 0; cls < c; ++cls) {
            const double denom = static_cast<double>(class_counts[cls]);
            for (std::size_t comp = 0; comp < k; ++comp) {
                means[idx(cls, k, comp)] /= denom;
            }
        }

        std::vector<double> covariance(k_times_k, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            const auto cls = static_cast<std::size_t>(y_labels[row]);
            for (std::size_t a = 0; a < k; ++a) {
                const double da = scores[idx(row, k, a)] - means[idx(cls, k, a)];
                for (std::size_t b = 0; b < k; ++b) {
                    const double db = scores[idx(row, k, b)] - means[idx(cls, k, b)];
                    covariance[idx(a, k, b)] += da * db;
                }
            }
        }
        const double cov_denom = static_cast<double>(X.rows - static_cast<std::int64_t>(n_classes));
        for (double& value : covariance) {
            value /= cov_denom;
        }

        std::vector<double> covariance_inv;
        if (!invert_square(covariance, k, covariance_inv)) {
            ctx.set_error("failed to invert PLS-LDA pooled covariance");
            return P4A_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> inv_means(c_times_k, 0.0);
        std::vector<double> constants(c, 0.0);
        for (std::size_t cls = 0; cls < c; ++cls) {
            for (std::size_t a = 0; a < k; ++a) {
                double sum = 0.0;
                for (std::size_t b = 0; b < k; ++b) {
                    sum += covariance_inv[idx(a, k, b)] * means[idx(cls, k, b)];
                }
                inv_means[idx(cls, k, a)] = sum;
            }
            double quad = 0.0;
            for (std::size_t a = 0; a < k; ++a) {
                quad += means[idx(cls, k, a)] * inv_means[idx(cls, k, a)];
            }
            constants[cls] = -0.5 * quad +
                std::log(static_cast<double>(class_counts[cls]) / static_cast<double>(n));
        }

        out.n_samples = X.rows;
        out.n_classes = n_classes;
        out.n_components = cfg.n_components;
        out.predictions.assign(n, 0);
        out.decision_scores.assign(n_times_c, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            std::size_t best_cls = 0;
            double best_score = -std::numeric_limits<double>::infinity();
            for (std::size_t cls = 0; cls < c; ++cls) {
                double score = constants[cls];
                for (std::size_t comp = 0; comp < k; ++comp) {
                    score += scores[idx(row, k, comp)] * inv_means[idx(cls, k, comp)];
                }
                out.decision_scores[idx(row, c, cls)] = score;
                if (score > best_score) {
                    best_score = score;
                    best_cls = cls;
                }
            }
            out.predictions[row] = static_cast<std::int32_t>(best_cls);
        }

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting PLS-LDA");
        out = PlsLdaResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting PLS-LDA");
        out = PlsLdaResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
