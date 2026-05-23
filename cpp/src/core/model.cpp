// SPDX-License-Identifier: CECILL-2.1

#include "core/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/common/linalg.hpp"
#include "core/common/matrix_view.hpp"
#include "core/common/parallel.hpp"
#include "core/common/status.hpp"

namespace {

constexpr double kZeroTolerance = 10.0 * std::numeric_limits<double>::epsilon();
constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] double dot(const std::vector<double>& a,
                         const std::vector<double>& b) noexcept {
    double out = 0.0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        out += a[i] * b[i];
    }
    return out;
}

[[nodiscard]] double squared_norm(const std::vector<double>& a) noexcept {
    return dot(a, a);
}

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
}

[[nodiscard]] double read_value(const n4m_matrix_view_t& v,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == N4M_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(v.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(v.data);
    return static_cast<double>(ptr[uoff]);
}

void write_value(n4m_matrix_view_t& v,
                 std::size_t row,
                 std::size_t col,
                 double value) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == N4M_DTYPE_F64) {
        auto* ptr = static_cast<double*>(v.data);
        ptr[uoff] = value;
        return;
    }
    auto* ptr = static_cast<float*>(v.data);
    ptr[uoff] = static_cast<float>(value);
}

[[nodiscard]] n4m_status_t validate_float_view(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& v,
                                               const char* name) noexcept {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(v);
    if (status != N4M_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::n4m::core::status_to_string(status));
        return status;
    }
    if (v.dtype != N4M_DTYPE_F64 && v.dtype != N4M_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_matrix_checked(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& src,
                                               const char* name,
                                               std::vector<double>& out) {
    const std::size_t rows = static_cast<std::size_t>(src.rows);
    const std::size_t cols = static_cast<std::size_t>(src.cols);
    resize_fill(out, rows * cols, 0.0);
    if ((src.dtype == N4M_DTYPE_F64 || src.dtype == N4M_DTYPE_F32) &&
        src.row_stride == static_cast<std::int64_t>(cols) &&
        src.col_stride == 1) {
        if (src.dtype == N4M_DTYPE_F64) {
            const auto* ptr = static_cast<const double*>(src.data);
            const std::size_t total = rows * cols;
            for (std::size_t offset = 0; offset < total; ++offset) {
                const double value = ptr[offset];
                if (!std::isfinite(value)) {
                    const std::size_t i = offset / cols;
                    const std::size_t j = offset - i * cols;
                    ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                                   name,
                                   ull(i),
                                   ull(j));
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                out[offset] = value;
            }
            return N4M_OK;
        }
        const auto* ptr = static_cast<const float*>(src.data);
        const std::size_t total = rows * cols;
        for (std::size_t offset = 0; offset < total; ++offset) {
            const double value = static_cast<double>(ptr[offset]);
            if (!std::isfinite(value)) {
                const std::size_t i = offset / cols;
                const std::size_t j = offset - i * cols;
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               ull(i),
                               ull(j));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[offset] = value;
        }
        return N4M_OK;
    }
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double value = read_value(src, i, j);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               ull(i),
                               ull(j));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(i, cols, j)] = value;
        }
    }
    return N4M_OK;
}

void center_scale_in_place(const std::vector<double>& original,
                           std::size_t rows,
                           std::size_t cols,
                           bool center_enabled,
                           bool scale_enabled,
                           std::vector<double>& mean,
                           std::vector<double>& scale,
                           std::vector<double>& standardized) {
    resize_fill(mean, cols, 0.0);
    resize_fill(scale, cols, 1.0);
    standardized = original;

    if (center_enabled) {
        for (std::size_t i = 0; i < rows; ++i) {
            const double* row = original.data() + i * cols;
            for (std::size_t j = 0; j < cols; ++j) {
                mean[j] += row[j];
            }
        }
        const double inv_rows = 1.0 / static_cast<double>(rows);
        for (std::size_t j = 0; j < cols; ++j) {
            mean[j] *= inv_rows;
        }
    }

    for (std::size_t i = 0; i < rows; ++i) {
        double* row = standardized.data() + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            row[j] -= mean[j];
        }
    }

    if (scale_enabled) {
        std::vector<double> sumsq(cols, 0.0);
        for (std::size_t i = 0; i < rows; ++i) {
            const double* row = standardized.data() + i * cols;
            for (std::size_t j = 0; j < cols; ++j) {
                const double value = row[j];
                sumsq[j] += value * value;
            }
        }
        const double denom = static_cast<double>(rows - 1U);
        for (std::size_t j = 0; j < cols; ++j) {
            const double stddev = std::sqrt(sumsq[j] / denom);
            scale[j] = (stddev == 0.0 || !std::isfinite(stddev)) ? 1.0 : stddev;
        }
        for (std::size_t i = 0; i < rows; ++i) {
            double* row = standardized.data() + i * cols;
            for (std::size_t j = 0; j < cols; ++j) {
                row[j] /= scale[j];
            }
        }
    }
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
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
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

void matrix_vector_product(const std::vector<double>& matrix,
                           std::size_t rows,
                           std::size_t cols,
                           const std::vector<double>& vector,
                           std::vector<double>& out) {
    n4m::linalg::matvec(matrix, rows, cols, vector, out);
}

void canonicalize_direction_sign(std::vector<double>& direction) noexcept {
    if (direction.empty()) {
        return;
    }
    std::size_t sign_idx = 0;
    double sign_abs = std::fabs(direction[0]);
    for (std::size_t i = 1; i < direction.size(); ++i) {
        const double candidate = std::fabs(direction[i]);
        if (candidate > sign_abs) {
            sign_idx = i;
            sign_abs = candidate;
        }
    }
    if (direction[sign_idx] < 0.0) {
        for (double& value : direction) {
            value = -value;
        }
    }
}

void canonicalize_svd_pair_sign(std::vector<double>& left,
                                std::vector<double>& right) noexcept {
    if (left.empty()) {
        return;
    }
    std::size_t sign_idx = 0;
    double sign_abs = std::fabs(left[0]);
    for (std::size_t i = 1; i < left.size(); ++i) {
        const double candidate = std::fabs(left[i]);
        if (candidate > sign_abs) {
            sign_idx = i;
            sign_abs = candidate;
        }
    }
    if (left[sign_idx] < 0.0) {
        for (double& value : left) {
            value = -value;
        }
        for (double& value : right) {
            value = -value;
        }
    }
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31U);
}

[[nodiscard]] double uniform_signed_from_state(std::uint64_t& state) noexcept {
    const std::uint64_t bits = splitmix64_next(state) >> 11U;
    const double unit =
        static_cast<double>(bits) * (1.0 / 9007199254740992.0);
    return 2.0 * unit - 1.0;
}

[[nodiscard]] n4m_status_t largest_symmetric_eigenvector(
    ::n4m::core::Context& ctx,
    const std::vector<double>& symmetric,
    std::size_t n,
    std::int32_t max_iter,
    double tol,
    const char* label,
    std::size_t component,
    std::vector<double>& eigenvector,
    double& eigenvalue) {
    resize_fill(eigenvector, n, 0.0);
    eigenvalue = 0.0;
    if (n == 0U || symmetric.size() != n * n) {
        ctx.set_errorf("%s eigenproblem shape is invalid at component %llu",
                       label,
                       ull(component));
        return N4M_ERR_INTERNAL;
    }
    if (n == 1U) {
        eigenvalue = symmetric[0];
        eigenvector[0] = 1.0;
        if (!(eigenvalue > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("%s covariance vanished at component %llu", label, ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        return N4M_OK;
    }

    std::vector<double> a = symmetric;
    std::vector<double> vectors;
    resize_fill(vectors, n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        vectors[idx(i, n, i)] = 1.0;
    }

    const std::size_t requested_limit =
        static_cast<std::size_t>(std::max<std::int32_t>(max_iter, 1));
    const std::size_t floor_limit = 32U * n * n;
    const std::size_t rotation_limit = std::max(requested_limit, floor_limit);
    const double stop_tol = std::min(std::max(tol, 1e-14), 1e-12);
    bool converged = false;

    for (std::size_t iter = 0; iter < rotation_limit; ++iter) {
        std::size_t pivot_p = 0;
        std::size_t pivot_q = 1;
        double max_off = 0.0;
        double diag_scale = 1.0;
        for (std::size_t row = 0; row < n; ++row) {
            diag_scale = std::max(diag_scale, std::fabs(a[idx(row, n, row)]));
            for (std::size_t col = row + 1U; col < n; ++col) {
                const double off = std::fabs(a[idx(row, n, col)]);
                if (off > max_off) {
                    max_off = off;
                    pivot_p = row;
                    pivot_q = col;
                }
            }
        }
        if (max_off <= stop_tol * diag_scale) {
            converged = true;
            break;
        }

        const double app = a[idx(pivot_p, n, pivot_p)];
        const double aqq = a[idx(pivot_q, n, pivot_q)];
        const double apq = a[idx(pivot_p, n, pivot_q)];
        if (apq == 0.0) {
            converged = true;
            break;
        }
        const double tau = (aqq - app) / (2.0 * apq);
        const double tau_sign = tau >= 0.0 ? 1.0 : -1.0;
        const double t = tau_sign / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        a[idx(pivot_p, n, pivot_p)] = app - t * apq;
        a[idx(pivot_q, n, pivot_q)] = aqq + t * apq;
        a[idx(pivot_p, n, pivot_q)] = 0.0;
        a[idx(pivot_q, n, pivot_p)] = 0.0;

        for (std::size_t k = 0; k < n; ++k) {
            if (k == pivot_p || k == pivot_q) {
                continue;
            }
            const double akp = a[idx(k, n, pivot_p)];
            const double akq = a[idx(k, n, pivot_q)];
            const double next_p = c * akp - s * akq;
            const double next_q = s * akp + c * akq;
            a[idx(k, n, pivot_p)] = next_p;
            a[idx(pivot_p, n, k)] = next_p;
            a[idx(k, n, pivot_q)] = next_q;
            a[idx(pivot_q, n, k)] = next_q;
        }

        for (std::size_t row = 0; row < n; ++row) {
            const double vip = vectors[idx(row, n, pivot_p)];
            const double viq = vectors[idx(row, n, pivot_q)];
            vectors[idx(row, n, pivot_p)] = c * vip - s * viq;
            vectors[idx(row, n, pivot_q)] = s * vip + c * viq;
        }
    }

    if (!converged) {
        ctx.set_errorf("%s Jacobi SVD failed to converge at component %llu",
                       label,
                       ull(component));
        return N4M_ERR_CONVERGENCE_FAILED;
    }

    std::size_t best = 0;
    double best_value = a[idx(0, n, 0)];
    for (std::size_t i = 1; i < n; ++i) {
        const double candidate = a[idx(i, n, i)];
        if (candidate > best_value) {
            best_value = candidate;
            best = i;
        }
    }
    if (!(best_value > std::numeric_limits<double>::epsilon())) {
        ctx.set_errorf("%s covariance vanished at component %llu", label, ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    for (std::size_t i = 0; i < n; ++i) {
        eigenvector[i] = vectors[idx(i, n, best)];
    }
    const double norm = std::sqrt(squared_norm(eigenvector));
    if (norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("%s eigenvector vanished at component %llu", label, ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : eigenvector) {
        value /= norm;
    }
    eigenvalue = best_value;
    return N4M_OK;
}

[[nodiscard]] n4m_status_t dominant_svd_pair(
    ::n4m::core::Context& ctx,
    const std::vector<double>& covariance,
    std::size_t features,
    std::size_t targets,
    std::int32_t max_iter,
    double tol,
    std::size_t component,
    std::vector<double>& left,
    std::vector<double>& right) {
    resize_fill(left, features, 0.0);
    resize_fill(right, targets, 0.0);
    if (targets == 1U) {
        double norm_sq = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double value = covariance[idx(feature, targets, 0)];
            left[feature] = value;
            norm_sq += value * value;
        }
        if (norm_sq <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SVD covariance vanished at component %llu", ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double norm = std::sqrt(norm_sq);
        for (double& value : left) {
            value /= norm;
        }
        right[0] = 1.0;
        canonicalize_svd_pair_sign(left, right);
        return N4M_OK;
    }

    double singular_value = 0.0;
    n4m_status_t status = N4M_OK;
    if (targets <= features) {
        std::vector<double> gram;
        resize_fill(gram, targets * targets, 0.0);
        for (std::size_t row = 0; row < targets; ++row) {
            for (std::size_t col = row; col < targets; ++col) {
                double sum = 0.0;
                for (std::size_t feature = 0; feature < features; ++feature) {
                    sum += covariance[idx(feature, targets, row)] *
                           covariance[idx(feature, targets, col)];
                }
                gram[idx(row, targets, col)] = sum;
                gram[idx(col, targets, row)] = sum;
            }
        }
        double eigenvalue = 0.0;
        status = largest_symmetric_eigenvector(ctx, gram, targets, max_iter, tol,
                                               "SVD right", component, right, eigenvalue);
        if (status != N4M_OK) {
            return status;
        }
        singular_value = std::sqrt(eigenvalue);
        for (std::size_t feature = 0; feature < features; ++feature) {
            double sum = 0.0;
            for (std::size_t target = 0; target < targets; ++target) {
                sum += covariance[idx(feature, targets, target)] * right[target];
            }
            left[feature] = sum / singular_value;
        }
    } else {
        std::vector<double> gram;
        resize_fill(gram, features * features, 0.0);
        for (std::size_t row = 0; row < features; ++row) {
            for (std::size_t col = row; col < features; ++col) {
                double sum = 0.0;
                for (std::size_t target = 0; target < targets; ++target) {
                    sum += covariance[idx(row, targets, target)] *
                           covariance[idx(col, targets, target)];
                }
                gram[idx(row, features, col)] = sum;
                gram[idx(col, features, row)] = sum;
            }
        }
        double eigenvalue = 0.0;
        status = largest_symmetric_eigenvector(ctx, gram, features, max_iter, tol,
                                               "SVD left", component, left, eigenvalue);
        if (status != N4M_OK) {
            return status;
        }
        singular_value = std::sqrt(eigenvalue);
        for (std::size_t target = 0; target < targets; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum += covariance[idx(feature, targets, target)] * left[feature];
            }
            right[target] = sum / singular_value;
        }
    }

    if (!(singular_value > std::numeric_limits<double>::epsilon())) {
        ctx.set_errorf("SVD singular value vanished at component %llu", ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    double left_norm = std::sqrt(squared_norm(left));
    double right_norm = std::sqrt(squared_norm(right));
    if (left_norm <= std::numeric_limits<double>::epsilon() ||
        right_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("SVD singular vectors vanished at component %llu", ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : left) {
        value /= left_norm;
    }
    for (double& value : right) {
        value /= right_norm;
    }
    canonicalize_svd_pair_sign(left, right);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t dominant_svd_pair_power(
    ::n4m::core::Context& ctx,
    const std::vector<double>& covariance,
    std::size_t features,
    std::size_t targets,
    std::int32_t max_iter,
    double tol,
    std::size_t component,
    std::vector<double>& left,
    std::vector<double>& right) {
    resize_fill(left, features, 0.0);
    resize_fill(right, targets, 0.0);
    if (targets == 1U) {
        double norm_sq = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double value = covariance[idx(feature, targets, 0)];
            left[feature] = value;
            norm_sq += value * value;
        }
        if (norm_sq <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("power covariance vanished at component %llu", ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double norm = std::sqrt(norm_sq);
        for (double& value : left) {
            value /= norm;
        }
        right[0] = 1.0;
        canonicalize_svd_pair_sign(left, right);
        return N4M_OK;
    }

    std::size_t initial_target = 0;
    double best_ss = -1.0;
    for (std::size_t target = 0; target < targets; ++target) {
        double sumsq = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double value = covariance[idx(feature, targets, target)];
            sumsq += value * value;
        }
        if (sumsq > best_ss) {
            best_ss = sumsq;
            initial_target = target;
        }
    }
    if (best_ss <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("power covariance vanished at component %llu", ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    right[initial_target] = 1.0;

    std::vector<double> next_right;
    resize_fill(next_right, targets, 0.0);
    bool converged = false;
    for (std::int32_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t feature = 0; feature < features; ++feature) {
            double sum = 0.0;
            for (std::size_t target = 0; target < targets; ++target) {
                sum += covariance[idx(feature, targets, target)] * right[target];
            }
            left[feature] = sum;
        }
        const double left_norm = std::sqrt(squared_norm(left));
        if (left_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("power left singular vector vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : left) {
            value /= left_norm;
        }

        for (std::size_t target = 0; target < targets; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum += covariance[idx(feature, targets, target)] * left[feature];
            }
            next_right[target] = sum;
        }
        const double right_norm = std::sqrt(squared_norm(next_right));
        if (right_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("power right singular vector vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : next_right) {
            value /= right_norm;
        }

        double diff_same = 0.0;
        double diff_opposite = 0.0;
        for (std::size_t target = 0; target < targets; ++target) {
            const double same = next_right[target] - right[target];
            const double opposite = next_right[target] + right[target];
            diff_same += same * same;
            diff_opposite += opposite * opposite;
        }
        right = next_right;
        if (std::min(diff_same, diff_opposite) < tol) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        ctx.set_errorf("power singular-vector iteration failed to converge at component %llu",
                       ull(component));
        return N4M_ERR_CONVERGENCE_FAILED;
    }

    for (std::size_t feature = 0; feature < features; ++feature) {
        double sum = 0.0;
        for (std::size_t target = 0; target < targets; ++target) {
            sum += covariance[idx(feature, targets, target)] * right[target];
        }
        left[feature] = sum;
    }
    const double left_norm = std::sqrt(squared_norm(left));
    if (left_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("power final left singular vector vanished at component %llu",
                       ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : left) {
        value /= left_norm;
    }

    for (std::size_t target = 0; target < targets; ++target) {
        double sum = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            sum += covariance[idx(feature, targets, target)] * left[feature];
        }
        right[target] = sum;
    }
    const double right_norm = std::sqrt(squared_norm(right));
    if (right_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("power final right singular vector vanished at component %llu",
                       ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : right) {
        value /= right_norm;
    }
    canonicalize_svd_pair_sign(left, right);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t dominant_svd_pair_randomized(
    ::n4m::core::Context& ctx,
    const std::vector<double>& covariance,
    std::size_t features,
    std::size_t targets,
    std::int32_t max_iter,
    double tol,
    std::uint64_t seed,
    std::size_t component,
    std::vector<double>& left,
    std::vector<double>& right) {
    resize_fill(left, features, 0.0);
    resize_fill(right, targets, 0.0);
    if (targets == 1U) {
        double norm_sq = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double value = covariance[idx(feature, targets, 0)];
            left[feature] = value;
            norm_sq += value * value;
        }
        if (norm_sq <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("randomized SVD covariance vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double norm = std::sqrt(norm_sq);
        for (double& value : left) {
            value /= norm;
        }
        right[0] = 1.0;
        canonicalize_svd_pair_sign(left, right);
        return N4M_OK;
    }

    std::uint64_t state =
        seed + kSplitMixGolden * static_cast<std::uint64_t>(component + 1U);
    for (std::size_t target = 0; target < targets; ++target) {
        right[target] = uniform_signed_from_state(state);
    }
    double right_norm = std::sqrt(squared_norm(right));
    if (right_norm <= std::numeric_limits<double>::epsilon()) {
        std::fill(right.begin(), right.end(), 0.0);
        right[0] = 1.0;
    } else {
        for (double& value : right) {
            value /= right_norm;
        }
    }

    std::vector<double> next_right;
    resize_fill(next_right, targets, 0.0);
    bool converged = false;
    for (std::int32_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t feature = 0; feature < features; ++feature) {
            double sum = 0.0;
            for (std::size_t target = 0; target < targets; ++target) {
                sum += covariance[idx(feature, targets, target)] * right[target];
            }
            left[feature] = sum;
        }
        const double left_norm = std::sqrt(squared_norm(left));
        if (left_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("randomized SVD left singular vector vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : left) {
            value /= left_norm;
        }

        for (std::size_t target = 0; target < targets; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum += covariance[idx(feature, targets, target)] * left[feature];
            }
            next_right[target] = sum;
        }
        right_norm = std::sqrt(squared_norm(next_right));
        if (right_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("randomized SVD right singular vector vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : next_right) {
            value /= right_norm;
        }

        double diff_same = 0.0;
        double diff_opposite = 0.0;
        for (std::size_t target = 0; target < targets; ++target) {
            const double same = next_right[target] - right[target];
            const double opposite = next_right[target] + right[target];
            diff_same += same * same;
            diff_opposite += opposite * opposite;
        }
        right = next_right;
        if (std::min(diff_same, diff_opposite) < tol) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        ctx.set_errorf("randomized SVD iteration failed to converge at component %llu",
                       ull(component));
        return N4M_ERR_CONVERGENCE_FAILED;
    }

    for (std::size_t feature = 0; feature < features; ++feature) {
        double sum = 0.0;
        for (std::size_t target = 0; target < targets; ++target) {
            sum += covariance[idx(feature, targets, target)] * right[target];
        }
        left[feature] = sum;
    }
    const double left_norm = std::sqrt(squared_norm(left));
    if (left_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("randomized SVD final left singular vector vanished at component %llu",
                       ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : left) {
        value /= left_norm;
    }

    for (std::size_t target = 0; target < targets; ++target) {
        double sum = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            sum += covariance[idx(feature, targets, target)] * left[feature];
        }
        right[target] = sum;
    }
    right_norm = std::sqrt(squared_norm(right));
    if (right_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("randomized SVD final right singular vector vanished at component %llu",
                       ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : right) {
        value /= right_norm;
    }
    canonicalize_svd_pair_sign(left, right);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t dominant_left_singular_direction(
    ::n4m::core::Context& ctx,
    const std::vector<double>& covariance,
    std::size_t features,
    std::size_t targets,
    std::int32_t max_iter,
    double tol,
    std::size_t component,
    std::vector<double>& out) {
    resize_fill(out, features, 0.0);
    if (targets == 1U) {
        for (std::size_t feature = 0; feature < features; ++feature) {
            out[feature] = covariance[idx(feature, targets, 0)];
        }
        if (squared_norm(out) <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SIMPLS covariance vanished at component %llu", ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        canonicalize_direction_sign(out);
        return N4M_OK;
    }

    std::size_t best_target = 0;
    double best_norm = 0.0;
    for (std::size_t target = 0; target < targets; ++target) {
        double norm = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double value = covariance[idx(feature, targets, target)];
            norm += value * value;
        }
        if (norm > best_norm) {
            best_norm = norm;
            best_target = target;
        }
    }
    if (best_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("SIMPLS covariance vanished at component %llu", ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    const double initial_norm = std::sqrt(best_norm);
    for (std::size_t feature = 0; feature < features; ++feature) {
        out[feature] = covariance[idx(feature, targets, best_target)] / initial_norm;
    }

    std::vector<double> right;
    std::vector<double> next_left;
    resize_fill(right, targets, 0.0);
    resize_fill(next_left, features, 0.0);
    const double stop_tol = std::min(std::max(tol, 1e-14), 1e-12);
    bool converged = false;
    for (std::int32_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t target = 0; target < targets; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum += covariance[idx(feature, targets, target)] * out[feature];
            }
            right[target] = sum;
        }
        const double right_norm = std::sqrt(squared_norm(right));
        if (right_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SIMPLS right singular direction vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : right) {
            value /= right_norm;
        }

        for (std::size_t feature = 0; feature < features; ++feature) {
            double sum = 0.0;
            for (std::size_t target = 0; target < targets; ++target) {
                sum += covariance[idx(feature, targets, target)] * right[target];
            }
            next_left[feature] = sum;
        }
        const double left_norm = std::sqrt(squared_norm(next_left));
        if (left_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SIMPLS left singular direction vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : next_left) {
            value /= left_norm;
        }
        if (dot(next_left, out) < 0.0) {
            for (double& value : next_left) {
                value = -value;
            }
        }

        double diff = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double delta = next_left[feature] - out[feature];
            diff += delta * delta;
        }
        out = next_left;
        if (diff < stop_tol * stop_tol) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        ctx.set_errorf(
            "SIMPLS singular-vector iteration failed to converge at component %llu",
            ull(component));
        return N4M_ERR_CONVERGENCE_FAILED;
    }
    canonicalize_direction_sign(out);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t compute_rotations_and_coefficients(
    ::n4m::core::Context& ctx,
    ::n4m::core::Model& model,
    std::size_t features,
    std::size_t targets,
    std::size_t components) {
    std::vector<double> ptw;
    resize_fill(ptw, components * components, 0.0);
    for (std::size_t row_comp = 0; row_comp < components; ++row_comp) {
        for (std::size_t col_comp = 0; col_comp < components; ++col_comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum += model.loadings_p[idx(feature, components, row_comp)] *
                       model.weights_w[idx(feature, components, col_comp)];
            }
            ptw[idx(row_comp, components, col_comp)] = sum;
        }
    }

    std::vector<double> ptw_inv;
    if (!invert_square(ptw, components, ptw_inv)) {
        ctx.set_error("failed to invert P.T @ W while computing rotations");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    resize_fill(model.rotations_r, features * components, 0.0);
    for (std::size_t feature = 0; feature < features; ++feature) {
        for (std::size_t comp = 0; comp < components; ++comp) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < components; ++inner) {
                sum += model.weights_w[idx(feature, components, inner)] *
                       ptw_inv[idx(inner, components, comp)];
            }
            model.rotations_r[idx(feature, components, comp)] = sum;
        }
    }

    resize_fill(model.coefficients, features * targets, 0.0);
    for (std::size_t feature = 0; feature < features; ++feature) {
        for (std::size_t target = 0; target < targets; ++target) {
            double sum = 0.0;
            for (std::size_t comp = 0; comp < components; ++comp) {
                sum += model.rotations_r[idx(feature, components, comp)] *
                       model.y_loadings_q[idx(target, components, comp)];
            }
            model.coefficients[idx(feature, targets, target)] =
                sum * model.y_scale[target] / model.x_scale[feature];
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_fit_request(::n4m::core::Context& ctx,
                                                const ::n4m::core::Config& cfg,
                                                const n4m_matrix_view_t& X,
                                                const n4m_matrix_view_t& Y) noexcept {
    const bool regression_chassis_algorithm =
        cfg.algorithm == N4M_ALGO_PLS_REGRESSION || cfg.algorithm == N4M_ALGO_PLS_DA;
    const bool supported_pls_regression =
        regression_chassis_algorithm &&
        (cfg.solver == N4M_SOLVER_NIPALS ||
         cfg.solver == N4M_SOLVER_ORTHOGONAL_SCORES ||
         cfg.solver == N4M_SOLVER_SIMPLS ||
         cfg.solver == N4M_SOLVER_KERNEL_ALGORITHM ||
         cfg.solver == N4M_SOLVER_WIDE_KERNEL ||
         cfg.solver == N4M_SOLVER_SVD ||
         cfg.solver == N4M_SOLVER_POWER ||
         cfg.solver == N4M_SOLVER_RANDOMIZED_SVD);
    const bool supported_pls_canonical =
        cfg.algorithm == N4M_ALGO_PLS_CANONICAL &&
        (cfg.solver == N4M_SOLVER_NIPALS || cfg.solver == N4M_SOLVER_SVD);
    const bool supported_pls_svd =
        cfg.algorithm == N4M_ALGO_PLS_SVD && cfg.solver == N4M_SOLVER_SVD;
    const bool opls_chassis_algorithm =
        cfg.algorithm == N4M_ALGO_OPLS || cfg.algorithm == N4M_ALGO_OPLS_DA;
    const bool supported_opls =
        opls_chassis_algorithm && cfg.solver == N4M_SOLVER_NIPALS;
    const bool supported_pcr =
        cfg.algorithm == N4M_ALGO_PCR && cfg.solver == N4M_SOLVER_SVD;
    const bool supported_sparse_pls =
        cfg.algorithm == N4M_ALGO_SPARSE_PLS &&
        cfg.solver == N4M_SOLVER_SIMPLS;
    if (!supported_pls_regression && !supported_pls_canonical && !supported_pls_svd &&
        !supported_opls && !supported_pcr && !supported_sparse_pls) {
        ctx.set_error(
            "this release supports N4M_ALGO_PLS_REGRESSION with N4M_SOLVER_NIPALS, "
            "N4M_SOLVER_ORTHOGONAL_SCORES, N4M_SOLVER_SIMPLS, "
            "N4M_SOLVER_KERNEL_ALGORITHM, N4M_SOLVER_WIDE_KERNEL or "
            "N4M_SOLVER_SVD, N4M_SOLVER_POWER or N4M_SOLVER_RANDOMIZED_SVD, "
            "N4M_ALGO_PLS_CANONICAL with N4M_SOLVER_NIPALS or N4M_SOLVER_SVD, "
            "N4M_ALGO_PLS_SVD with N4M_SOLVER_SVD, "
            "N4M_ALGO_PLS_DA with the PLS regression solver set, "
            "N4M_ALGO_OPLS/N4M_ALGO_OPLS_DA with N4M_SOLVER_NIPALS, "
            "plus N4M_ALGO_PCR with N4M_SOLVER_SVD");
        return N4M_ERR_UNSUPPORTED;
    }
    const bool supported_deflation =
        ((supported_pls_regression || supported_pcr || supported_sparse_pls) &&
         cfg.deflation == N4M_DEFLATION_REGRESSION) ||
        ((supported_pls_canonical || supported_pls_svd) &&
         cfg.deflation == N4M_DEFLATION_CANONICAL) ||
        (supported_opls && cfg.deflation == N4M_DEFLATION_ORTHOGONAL);
    if (!supported_deflation) {
        ctx.set_error(
            "this release supports N4M_DEFLATION_REGRESSION for PLS regression/PCR "
            "and N4M_DEFLATION_CANONICAL for PLS canonical/PLSSVD and "
            "N4M_DEFLATION_ORTHOGONAL for OPLS");
        return N4M_ERR_UNSUPPORTED;
    }
    if (cfg.pipeline != nullptr || cfg.operator_bank != nullptr || cfg.gating_strategy != nullptr) {
        ctx.set_error("pipelines, operator banks and gating strategies are not yet supported by model_fit");
        return N4M_ERR_UNSUPPORTED;
    }

    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.dtype != Y.dtype) {
        ctx.set_error("X and Y must have the same floating dtype");
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (X.rows < 2 || X.cols < 1 || Y.cols < 1) {
        ctx.set_error("fit requires at least 2 rows, 1 X column and 1 Y column");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("matrix column count exceeds the ABI limits");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::int64_t rank_upper_bound = std::min(X.rows, X.cols);
    if (cfg.algorithm == N4M_ALGO_PLS_CANONICAL || cfg.algorithm == N4M_ALGO_PLS_SVD) {
        rank_upper_bound = std::min(rank_upper_bound, Y.cols);
    }
    if (static_cast<std::int64_t>(cfg.n_components) > rank_upper_bound) {
        ctx.set_errorf("n_components upper bound is %lld; got %d",
                       static_cast<long long>(rank_upper_bound),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_apply_input(::n4m::core::Context& ctx,
                                                const ::n4m::core::Model& model,
                                                const n4m_matrix_view_t& X,
                                                const n4m_matrix_view_t& out,
                                                std::int64_t expected_cols,
                                                const char* op) noexcept {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, out, "output");
    if (status != N4M_OK) {
        return status;
    }
    if (X.cols != static_cast<std::int64_t>(model.n_features)) {
        ctx.set_errorf("%s expected %d X features, got %lld",
                       op,
                       static_cast<int>(model.n_features),
                       static_cast<long long>(X.cols));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (out.rows != X.rows || out.cols != expected_cols) {
        ctx.set_errorf("%s output shape must be (%lld, %lld)",
                       op,
                       static_cast<long long>(X.rows),
                       static_cast<long long>(expected_cols));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    return N4M_OK;
}

enum class CanonicalWeightSolver {
    Nipals,
    Svd
};

[[nodiscard]] n4m_status_t canonical_nipals_pair(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Config& cfg,
    const std::vector<double>& xk,
    std::vector<double>& yk,
    std::size_t rows,
    std::size_t features,
    std::size_t targets,
    std::size_t component,
    std::vector<double>& x_weights,
    std::vector<double>& y_weights) {
    for (std::size_t target = 0; target < targets; ++target) {
        bool all_close_zero = true;
        for (std::size_t row = 0; row < rows; ++row) {
            if (std::fabs(yk[idx(row, targets, target)]) >= kZeroTolerance) {
                all_close_zero = false;
                break;
            }
        }
        if (all_close_zero) {
            for (std::size_t row = 0; row < rows; ++row) {
                yk[idx(row, targets, target)] = 0.0;
            }
        }
    }

    std::size_t initial_target = targets;
    for (std::size_t target = 0; target < targets; ++target) {
        for (std::size_t row = 0; row < rows; ++row) {
            if (std::fabs(yk[idx(row, targets, target)]) >
                std::numeric_limits<double>::epsilon()) {
                initial_target = target;
                break;
            }
        }
        if (initial_target != targets) {
            break;
        }
    }
    if (initial_target == targets) {
        ctx.set_errorf("Y residual became constant at canonical component %llu",
                       ull(component));
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    std::vector<double> y_score;
    std::vector<double> x_weights_old;
    std::vector<double> x_score;
    resize_fill(y_score, rows, 0.0);
    resize_fill(x_weights, features, 0.0);
    resize_fill(y_weights, targets, 0.0);
    resize_fill(x_weights_old, features, 100.0);
    resize_fill(x_score, rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        y_score[row] = yk[idx(row, targets, initial_target)];
    }

    bool converged = false;
    for (std::int32_t iter = 0; iter < cfg.max_iter; ++iter) {
        const double y_score_ss = squared_norm(y_score);
        if (y_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("canonical Y score vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        // x_weights = (1 / y_score_ss) * X^T * y_score   (NIPALS inner power step)
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes,
            rows, features,
            1.0 / y_score_ss,
            xk.data(),
            y_score.data(),
            0.0,
            x_weights.data());

        const double x_weight_norm = std::sqrt(squared_norm(x_weights));
        if (x_weight_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("canonical X weights vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double x_weight_denom =
            x_weight_norm + std::numeric_limits<double>::epsilon();
        for (double& value : x_weights) {
            value /= x_weight_denom;
        }

        matrix_vector_product(xk, rows, features, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("canonical X score vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        for (std::size_t target = 0; target < targets; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                sum += yk[idx(row, targets, target)] * x_score[row];
            }
            y_weights[target] = sum / x_score_ss;
        }

        const double y_weight_norm = std::sqrt(squared_norm(y_weights));
        if (y_weight_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("canonical Y weights vanished at component %llu",
                           ull(component));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double y_weight_denom =
            y_weight_norm + std::numeric_limits<double>::epsilon();
        for (double& value : y_weights) {
            value /= y_weight_denom;
        }

        const double y_weight_ss =
            squared_norm(y_weights) + std::numeric_limits<double>::epsilon();
        for (std::size_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (std::size_t target = 0; target < targets; ++target) {
                sum += yk[idx(row, targets, target)] * y_weights[target];
            }
            y_score[row] = sum / y_weight_ss;
        }

        double diff = 0.0;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double delta = x_weights[feature] - x_weights_old[feature];
            diff += delta * delta;
        }
        if (diff < cfg.tol || targets == 1U) {
            converged = true;
            break;
        }
        x_weights_old = x_weights;
    }

    if (!converged) {
        ctx.set_errorf("canonical NIPALS iteration failed to converge at component %llu",
                       ull(component));
        return N4M_ERR_CONVERGENCE_FAILED;
    }

    std::size_t sign_idx = 0;
    double sign_abs = std::fabs(x_weights[0]);
    for (std::size_t feature = 1; feature < features; ++feature) {
        const double candidate = std::fabs(x_weights[feature]);
        if (candidate > sign_abs) {
            sign_idx = feature;
            sign_abs = candidate;
        }
    }
    if (x_weights[sign_idx] < 0.0) {
        for (double& value : x_weights) {
            value = -value;
        }
        for (double& value : y_weights) {
            value = -value;
        }
    }

    return N4M_OK;
}

[[nodiscard]] n4m_status_t canonical_svd_pair(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Config& cfg,
    const std::vector<double>& xk,
    const std::vector<double>& yk,
    std::size_t rows,
    std::size_t features,
    std::size_t targets,
    std::size_t component,
    std::vector<double>& x_weights,
    std::vector<double>& y_weights) {
    std::vector<double> covariance;
    resize_fill(covariance, features * targets, 0.0);
    // Cross-covariance: covariance = X^T * Y  (features x targets).
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        features, targets, rows,
        1.0,
        xk.data(), features,
        yk.data(), targets,
        0.0,
        covariance.data(), targets);
    return dominant_svd_pair(ctx, covariance, features, targets,
                             cfg.max_iter, cfg.tol,
                             component, x_weights, y_weights);
}

[[nodiscard]] n4m_status_t fit_pls_canonical_impl(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<::n4m::core::Model>& out_model,
    CanonicalWeightSolver solver) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<::n4m::core::Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> x_weights;
        std::vector<double> y_weights;
        if (solver == CanonicalWeightSolver::Nipals) {
            status = canonical_nipals_pair(ctx, cfg, xk, yk, n, p, q,
                                           comp, x_weights, y_weights);
        } else {
            status = canonical_svd_pair(ctx, cfg, xk, yk, n, p, q,
                                        comp, x_weights, y_weights);
        }
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("canonical X score vanished after weights at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_scores;
        matrix_vector_product(yk, n, q, y_weights, y_scores);
        const double y_score_ss = squared_norm(y_scores);
        if (y_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("canonical Y score vanished after weights at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
        }

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += y_scores[row] * yk[idx(row, q, target)];
            }
            y_loadings[target] = sum / y_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t target = 0; target < q; ++target) {
                yk[idx(row, q, target)] -= y_scores[row] * y_loadings[target];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t fit_pls_regression_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        for (std::size_t target = 0; target < q; ++target) {
            bool all_close_zero = true;
            for (std::size_t row = 0; row < n; ++row) {
                if (std::fabs(yk[idx(row, q, target)]) >= kZeroTolerance) {
                    all_close_zero = false;
                    break;
                }
            }
            if (all_close_zero) {
                for (std::size_t row = 0; row < n; ++row) {
                    yk[idx(row, q, target)] = 0.0;
                }
            }
        }

        std::size_t initial_target = q;
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t row = 0; row < n; ++row) {
                if (std::fabs(yk[idx(row, q, target)]) >
                    std::numeric_limits<double>::epsilon()) {
                    initial_target = target;
                    break;
                }
            }
            if (initial_target != q) {
                break;
            }
        }
        if (initial_target == q) {
            ctx.set_errorf("Y residual became constant at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_score;
        resize_fill(y_score, n, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            y_score[row] = yk[idx(row, q, initial_target)];
        }

        std::vector<double> x_weights;
        std::vector<double> y_weights;
        std::vector<double> x_weights_old;
        std::vector<double> x_score;
        resize_fill(x_weights, p, 0.0);
        resize_fill(y_weights, q, 0.0);
        resize_fill(x_weights_old, p, 100.0);
        resize_fill(x_score, n, 0.0);
        bool converged = false;

        for (std::int32_t iter = 0; iter < cfg.max_iter; ++iter) {
            const double y_score_ss = squared_norm(y_score);
            if (y_score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("Y score vanished at component %llu", ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            // x_weights = (1 / y_score_ss) * X^T * y_score   (NIPALS inner power step)
            n4m::linalg::gemv(
                n4m::linalg::Trans_Yes,
                n, p,
                1.0 / y_score_ss,
                xk.data(),
                y_score.data(),
                0.0,
                x_weights.data());

            const double x_weight_norm = std::sqrt(squared_norm(x_weights));
            if (x_weight_norm <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("X weights vanished at component %llu", ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            const double x_weight_denom = x_weight_norm + std::numeric_limits<double>::epsilon();
            for (double& value : x_weights) {
                value /= x_weight_denom;
            }

            matrix_vector_product(xk, n, p, x_weights, x_score);
            const double x_score_ss = squared_norm(x_score);
            if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("X score vanished at component %llu", ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            for (std::size_t target = 0; target < q; ++target) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += yk[idx(row, q, target)] * x_score[row];
                }
                y_weights[target] = sum / x_score_ss;
            }

            const double y_weight_ss = squared_norm(y_weights);
            if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("Y weights vanished at component %llu", ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            const double y_score_denom = y_weight_ss + std::numeric_limits<double>::epsilon();
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_weights[target];
                }
                y_score[row] = sum / y_score_denom;
            }

            double diff = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                const double delta = x_weights[feature] - x_weights_old[feature];
                diff += delta * delta;
            }
            if (diff < cfg.tol || q == 1U) {
                converged = true;
                break;
            }
            x_weights_old = x_weights;
        }

        if (!converged) {
            ctx.set_errorf("NIPALS power iteration failed to converge at component %llu",
                           ull(comp));
            return N4M_ERR_CONVERGENCE_FAILED;
        }

        std::size_t sign_idx = 0;
        double sign_abs = std::fabs(x_weights[0]);
        for (std::size_t feature = 1; feature < p; ++feature) {
            const double candidate = std::fabs(x_weights[feature]);
            if (candidate > sign_abs) {
                sign_idx = feature;
                sign_abs = candidate;
            }
        }
        if (x_weights[sign_idx] < 0.0) {
            for (double& value : x_weights) {
                value = -value;
            }
            for (double& value : y_weights) {
                value = -value;
            }
            for (double& value : x_score) {
                value = -value;
            }
        }

        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("X score vanished after sign normalization at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double y_weight_ss = squared_norm(y_weights);
        if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("Y weights vanished after sign normalization at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_score_denom = y_weight_ss + std::numeric_limits<double>::epsilon();
        for (std::size_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                sum += yk[idx(row, q, target)] * y_weights[target];
            }
            y_scores[row] = sum / y_score_denom;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes,
            n, p,
            1.0 / x_score_ss,
            xk.data(),
            x_score.data(),
            0.0,
            x_loadings.data());
        n4m::linalg::ger(
            n, p,
            -1.0,
            x_score.data(),
            x_loadings.data(),
            xk.data(), p);

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes,
            n, q,
            1.0 / x_score_ss,
            yk.data(),
            x_score.data(),
            0.0,
            y_loadings.data());
        n4m::linalg::ger(
            n, q,
            -1.0,
            x_score.data(),
            y_loadings.data(),
            yk.data(), q);

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_canonical_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    return fit_pls_canonical_impl(ctx, cfg, X, Y, out_model,
                                  CanonicalWeightSolver::Nipals);
}

n4m_status_t fit_pls_canonical_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    return fit_pls_canonical_impl(ctx, cfg, X, Y, out_model,
                                  CanonicalWeightSolver::Svd);
}

n4m_status_t fit_pls_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xs;
    std::vector<double> ys;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xs);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, ys);

    resize_fill(model->coefficients, p * q, 0.0);
    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    resize_fill(model->rotations_r, p * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    std::vector<double> covariance;
    resize_fill(covariance, p * q, 0.0);
    // Cross-covariance: covariance = Xs^T * Ys  (p x q).
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        p, q, n,
        1.0,
        xs.data(), p,
        ys.data(), q,
        0.0,
        covariance.data(), q);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> x_weights;
        std::vector<double> y_weights;
        status = dominant_svd_pair(ctx, covariance, p, q, cfg.max_iter, cfg.tol,
                                   comp, x_weights, y_weights);
        if (status != N4M_OK) {
            return status;
        }

        double singular_value = 0.0;
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                singular_value += x_weights[feature] *
                                  covariance[idx(feature, q, target)] *
                                  y_weights[target];
            }
        }
        if (!(singular_value > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("PLSSVD singular value vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> x_score;
        std::vector<double> y_score;
        matrix_vector_product(xs, n, p, x_weights, x_score);
        matrix_vector_product(ys, n, q, y_weights, y_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("PLSSVD X score vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += xs[idx(row, p, feature)] * x_score[row];
            }
            x_loadings[feature] = sum / x_score_ss;
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
            model->rotations_r[idx(feature, a, comp)] = x_weights[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_weights[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_score[row];
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                covariance[idx(feature, q, target)] -=
                    singular_value * x_weights[feature] * y_weights[target];
            }
        }
    }

    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t comp = 0; comp < a; ++comp) {
                sum += model->weights_w[idx(feature, a, comp)] *
                       model->y_loadings_q[idx(target, a, comp)];
            }
            model->coefficients[idx(feature, q, target)] =
                sum * model->y_scale[target] / model->x_scale[feature];
        }
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_opls_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->coefficients, p * q, 0.0);
    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    resize_fill(model->rotations_r, p * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    std::vector<double> projection;
    resize_fill(projection, p * p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        projection[idx(feature, p, feature)] = 1.0;
    }

    auto projection_times_vector =
        [&](const std::vector<double>& weights, std::vector<double>& out) {
            resize_fill(out, p, 0.0);
            for (std::size_t row = 0; row < p; ++row) {
                double sum = 0.0;
                for (std::size_t col = 0; col < p; ++col) {
                    sum += projection[idx(row, p, col)] * weights[col];
                }
                out[row] = sum;
            }
        };

    auto sign_normalize =
        [&](std::vector<double>& weights,
            std::vector<double>* scores,
            std::vector<double>* loadings,
            std::vector<double>* y_loadings) {
            std::size_t sign_idx = 0;
            double sign_abs = std::fabs(weights[0]);
            for (std::size_t feature = 1; feature < p; ++feature) {
                const double candidate = std::fabs(weights[feature]);
                if (candidate > sign_abs) {
                    sign_idx = feature;
                    sign_abs = candidate;
                }
            }
            if (weights[sign_idx] >= 0.0) {
                return;
            }
            for (double& value : weights) {
                value = -value;
            }
            if (scores != nullptr) {
                for (double& value : *scores) {
                    value = -value;
                }
            }
            if (loadings != nullptr) {
                for (double& value : *loadings) {
                    value = -value;
                }
            }
            if (y_loadings != nullptr) {
                for (double& value : *y_loadings) {
                    value = -value;
                }
            }
        };

    auto predictive_component =
        [&](const std::vector<double>& x_work,
            std::size_t component,
            std::vector<double>& weights,
            std::vector<double>& scores,
            std::vector<double>& loadings,
            std::vector<double>& y_loadings,
            std::vector<double>& y_score) -> n4m_status_t {
            resize_fill(weights, p, 0.0);
            std::vector<double> y_weights;
            resize_fill(y_weights, q, 0.0);
            if (q == 1U) {
                for (std::size_t feature = 0; feature < p; ++feature) {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < n; ++row) {
                        sum += x_work[idx(row, p, feature)] * yk[idx(row, q, 0U)];
                    }
                    weights[feature] = sum;
                }
                const double weight_norm = std::sqrt(squared_norm(weights));
                if (weight_norm <= std::numeric_limits<double>::epsilon()) {
                    ctx.set_errorf("OPLS predictive weights vanished at component %llu",
                                   ull(component));
                    return N4M_ERR_NUMERICAL_FAILURE;
                }
                for (double& value : weights) {
                    value /= weight_norm;
                }
                y_weights[0] = 1.0;
            } else {
                std::vector<double> covariance;
                resize_fill(covariance, p * q, 0.0);
                // Cross-covariance: covariance = X_work^T * Yk  (p x q).
                n4m::linalg::gemm(
                    n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
                    p, q, n,
                    1.0,
                    x_work.data(), p,
                    yk.data(), q,
                    0.0,
                    covariance.data(), q);
                const n4m_status_t pair_status =
                    dominant_svd_pair(ctx, covariance, p, q, cfg.max_iter,
                                      cfg.tol, component, weights, y_weights);
                if (pair_status != N4M_OK) {
                    return pair_status;
                }
            }

            matrix_vector_product(x_work, n, p, weights, scores);
            const double score_ss = squared_norm(scores);
            if (score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("OPLS predictive score vanished at component %llu",
                               ull(component));
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            resize_fill(loadings, p, 0.0);
            for (std::size_t feature = 0; feature < p; ++feature) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += x_work[idx(row, p, feature)] * scores[row];
                }
                loadings[feature] = sum / score_ss;
            }

            resize_fill(y_loadings, q, 0.0);
            resize_fill(y_score, n, 0.0);
            const double y_weight_ss = squared_norm(y_weights);
            if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("OPLS predictive Y weights vanished at component %llu",
                               ull(component));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            const double y_score_denom =
                y_weight_ss + std::numeric_limits<double>::epsilon();
            for (std::size_t row = 0; row < n; ++row) {
                double score_sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    y_loadings[target] += yk[idx(row, q, target)] * scores[row];
                    score_sum += yk[idx(row, q, target)] * y_weights[target];
                }
                y_score[row] = score_sum / y_score_denom;
            }
            for (double& value : y_loadings) {
                value /= score_ss;
            }
            sign_normalize(weights, &scores, &loadings, &y_loadings);
            if (q == 1U) {
                const double y_loading_denom =
                    y_loadings[0] * y_loadings[0] +
                    std::numeric_limits<double>::epsilon();
                for (std::size_t row = 0; row < n; ++row) {
                    y_score[row] =
                        yk[idx(row, q, 0U)] * y_loadings[0] / y_loading_denom;
                }
            }
            return N4M_OK;
        };

    for (std::size_t orth = 0; orth + 1U < a; ++orth) {
        std::vector<double> predictive_weights;
        std::vector<double> predictive_scores;
        std::vector<double> predictive_loadings;
        std::vector<double> predictive_y_loadings;
        std::vector<double> predictive_y_score;
        status = predictive_component(xk, orth + 1U, predictive_weights,
                                      predictive_scores, predictive_loadings,
                                      predictive_y_loadings, predictive_y_score);
        if (status != N4M_OK) {
            return status;
        }

        const double predictive_weight_ss = squared_norm(predictive_weights);
        if (predictive_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("OPLS predictive weight norm vanished at orthogonal component %llu",
                           ull(orth));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double alignment = dot(predictive_weights, predictive_loadings) /
                                 predictive_weight_ss;
        std::vector<double> orth_weights;
        resize_fill(orth_weights, p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            orth_weights[feature] =
                predictive_loadings[feature] - predictive_weights[feature] * alignment;
        }
        const double orth_weight_norm = std::sqrt(squared_norm(orth_weights));
        if (orth_weight_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("OPLS orthogonal weights vanished at component %llu",
                           ull(orth));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : orth_weights) {
            value /= orth_weight_norm;
        }
        sign_normalize(orth_weights, nullptr, nullptr, nullptr);

        std::vector<double> orth_scores;
        matrix_vector_product(xk, n, p, orth_weights, orth_scores);
        const double orth_score_ss = squared_norm(orth_scores);
        if (orth_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("OPLS orthogonal score vanished at component %llu",
                           ull(orth));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> orth_loadings;
        resize_fill(orth_loadings, p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += xk[idx(row, p, feature)] * orth_scores[row];
            }
            orth_loadings[feature] = sum / orth_score_ss;
        }

        std::vector<double> raw_rotation;
        projection_times_vector(orth_weights, raw_rotation);

        const std::size_t col = orth + 1U;
        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, col)] = orth_weights[feature];
            model->loadings_p[idx(feature, a, col)] = orth_loadings[feature];
            model->rotations_r[idx(feature, a, col)] = raw_rotation[feature];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, col)] = orth_scores[row];
        }

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= orth_scores[row] * orth_loadings[feature];
            }
        }
        for (std::size_t row = 0; row < p; ++row) {
            for (std::size_t col_update = 0; col_update < p; ++col_update) {
                projection[idx(row, p, col_update)] -=
                    raw_rotation[row] * orth_loadings[col_update];
            }
        }
    }

    std::vector<double> predictive_weights;
    std::vector<double> predictive_scores;
    std::vector<double> predictive_loadings;
    std::vector<double> predictive_y_loadings;
    std::vector<double> predictive_y_score;
    status = predictive_component(xk, 0U, predictive_weights, predictive_scores,
                                  predictive_loadings, predictive_y_loadings,
                                  predictive_y_score);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> raw_predictive_rotation;
    projection_times_vector(predictive_weights, raw_predictive_rotation);
    const double predictive_denom = dot(predictive_loadings, predictive_weights);
    if (std::fabs(predictive_denom) <= std::numeric_limits<double>::epsilon()) {
        ctx.set_error("OPLS predictive denominator vanished");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    for (std::size_t feature = 0; feature < p; ++feature) {
        model->weights_w[idx(feature, a, 0U)] = predictive_weights[feature];
        model->loadings_p[idx(feature, a, 0U)] = predictive_loadings[feature];
        model->rotations_r[idx(feature, a, 0U)] = raw_predictive_rotation[feature];
        for (std::size_t target = 0; target < q; ++target) {
            const double coef_std =
                raw_predictive_rotation[feature] *
                predictive_y_loadings[target] / predictive_denom;
            model->coefficients[idx(feature, q, target)] =
                coef_std * model->y_scale[target] / model->x_scale[feature];
        }
    }
    for (std::size_t target = 0; target < q; ++target) {
        model->y_loadings_q[idx(target, a, 0U)] = predictive_y_loadings[target];
    }
    for (std::size_t row = 0; row < n; ++row) {
        scores_t[idx(row, a, 0U)] = predictive_scores[row];
        y_scores_u[idx(row, a, 0U)] = predictive_y_score[row];
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_regression_orthogonal_scores(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> y_score;
        resize_fill(y_score, n, 0.0);
        if (q == 1U) {
            for (std::size_t row = 0; row < n; ++row) {
                y_score[row] = yk[idx(row, q, 0)];
            }
        } else {
            std::size_t initial_target = 0;
            double best_ss = -1.0;
            for (std::size_t target = 0; target < q; ++target) {
                double sumsq = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    const double value = yk[idx(row, q, target)];
                    sumsq += value * value;
                }
                if (sumsq > best_ss) {
                    best_ss = sumsq;
                    initial_target = target;
                }
            }
            if (best_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("orthogonal-scores Y residual became constant at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            for (std::size_t row = 0; row < n; ++row) {
                y_score[row] = yk[idx(row, q, initial_target)];
            }
        }

        std::vector<double> x_weights;
        std::vector<double> x_score;
        std::vector<double> x_score_old;
        std::vector<double> y_loadings;
        resize_fill(x_weights, p, 0.0);
        resize_fill(x_score, n, 0.0);
        resize_fill(x_score_old, n, 0.0);
        resize_fill(y_loadings, q, 0.0);
        bool converged = false;

        for (std::int32_t iter = 0; iter < cfg.max_iter; ++iter) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += xk[idx(row, p, feature)] * y_score[row];
                }
                x_weights[feature] = sum;
            }

            const double x_weight_norm = std::sqrt(squared_norm(x_weights));
            if (x_weight_norm <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("orthogonal-scores X weights vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            for (double& value : x_weights) {
                value /= x_weight_norm;
            }

            matrix_vector_product(xk, n, p, x_weights, x_score);
            const double x_score_ss = squared_norm(x_score);
            if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("orthogonal-scores X score vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            for (std::size_t target = 0; target < q; ++target) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += yk[idx(row, q, target)] * (x_score[row] / x_score_ss);
                }
                y_loadings[target] = sum;
            }

            if (q == 1U) {
                converged = true;
                break;
            }

            double score_delta = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                const double delta = x_score[row] - x_score_old[row];
                if (x_score[row] == 0.0) {
                    if (delta != 0.0) {
                        score_delta = std::numeric_limits<double>::infinity();
                        break;
                    }
                    continue;
                }
                const double ratio = delta / x_score[row];
                if (!std::isnan(ratio)) {
                    score_delta += std::fabs(ratio);
                }
            }
            if (score_delta < cfg.tol) {
                converged = true;
                break;
            }

            const double y_loading_ss = squared_norm(y_loadings);
            if (y_loading_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("orthogonal-scores Y loadings vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_loadings[target];
                }
                y_score[row] = sum / y_loading_ss;
            }
            x_score_old = x_score;
        }

        if (!converged) {
            ctx.set_errorf("orthogonal-scores iteration failed to converge at component %llu",
                           ull(comp));
            return N4M_ERR_CONVERGENCE_FAILED;
        }

        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("orthogonal-scores X score vanished after convergence at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += xk[idx(row, p, feature)] * (x_score[row] / x_score_ss);
            }
            x_loadings[feature] = sum;
        }

        std::size_t sign_idx = 0;
        double sign_abs = std::fabs(x_weights[0]);
        for (std::size_t feature = 1; feature < p; ++feature) {
            const double candidate = std::fabs(x_weights[feature]);
            if (candidate > sign_abs) {
                sign_idx = feature;
                sign_abs = candidate;
            }
        }
        if (x_weights[sign_idx] < 0.0) {
            for (double& value : x_weights) {
                value = -value;
            }
            for (double& value : x_score) {
                value = -value;
            }
            for (double& value : x_loadings) {
                value = -value;
            }
            for (double& value : y_loadings) {
                value = -value;
            }
            for (double& value : y_score) {
                value = -value;
            }
        }

        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
            for (std::size_t target = 0; target < q; ++target) {
                yk[idx(row, q, target)] -= x_score[row] * y_loadings[target];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_score[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_regression_kernel(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::size_t initial_target = q;
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t row = 0; row < n; ++row) {
                if (std::fabs(yk[idx(row, q, target)]) >
                    std::numeric_limits<double>::epsilon()) {
                    initial_target = target;
                    break;
                }
            }
            if (initial_target != q) {
                break;
            }
        }
        if (initial_target == q) {
            ctx.set_errorf("kernel PLS Y residual became constant at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> kernel;
        resize_fill(kernel, n * n, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = row; col < n; ++col) {
                double sum = 0.0;
                for (std::size_t feature = 0; feature < p; ++feature) {
                    sum += xk[idx(row, p, feature)] * xk[idx(col, p, feature)];
                }
                kernel[idx(row, n, col)] = sum;
                kernel[idx(col, n, row)] = sum;
            }
        }

        std::vector<double> y_score;
        resize_fill(y_score, n, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            y_score[row] = yk[idx(row, q, initial_target)];
        }

        std::vector<double> x_weights;
        std::vector<double> x_score;
        std::vector<double> x_score_old;
        std::vector<double> y_loadings;
        resize_fill(x_weights, p, 0.0);
        resize_fill(x_score, n, 0.0);
        resize_fill(x_score_old, n, 0.0);
        resize_fill(y_loadings, q, 0.0);
        bool converged = false;

        for (std::int32_t iter = 0; iter < cfg.max_iter; ++iter) {
            const double y_score_ss = squared_norm(y_score);
            if (y_score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("kernel PLS Y score vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            std::vector<double> raw_score;
            resize_fill(raw_score, n, 0.0);
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t col = 0; col < n; ++col) {
                    sum += kernel[idx(row, n, col)] * y_score[col];
                }
                raw_score[row] = sum / y_score_ss;
            }
            const double raw_score_norm = std::sqrt(squared_norm(raw_score));
            if (raw_score_norm <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("kernel PLS X score vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            for (std::size_t row = 0; row < n; ++row) {
                x_score[row] = raw_score[row] / raw_score_norm;
            }

            for (std::size_t feature = 0; feature < p; ++feature) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += xk[idx(row, p, feature)] * y_score[row];
                }
                x_weights[feature] = (sum / y_score_ss) / raw_score_norm;
            }

            const double x_score_ss = squared_norm(x_score);
            if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("kernel PLS normalized X score vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            for (std::size_t target = 0; target < q; ++target) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += yk[idx(row, q, target)] * x_score[row];
                }
                y_loadings[target] = sum / x_score_ss;
            }

            const double y_loading_ss = squared_norm(y_loadings);
            if (y_loading_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("kernel PLS Y loadings vanished at component %llu",
                               ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_loadings[target];
                }
                y_score[row] = sum / y_loading_ss;
            }

            if (iter > 0 || q == 1U) {
                double diff_same = 0.0;
                double diff_opposite = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    const double same = x_score[row] - x_score_old[row];
                    const double opposite = x_score[row] + x_score_old[row];
                    diff_same += same * same;
                    diff_opposite += opposite * opposite;
                }
                if (std::min(diff_same, diff_opposite) < cfg.tol || q == 1U) {
                    converged = true;
                    break;
                }
            }
            x_score_old = x_score;
        }

        if (!converged) {
            ctx.set_errorf("kernel PLS iteration failed to converge at component %llu",
                           ull(comp));
            return N4M_ERR_CONVERGENCE_FAILED;
        }

        std::size_t sign_idx = 0;
        double sign_abs = std::fabs(x_weights[0]);
        for (std::size_t feature = 1; feature < p; ++feature) {
            const double candidate = std::fabs(x_weights[feature]);
            if (candidate > sign_abs) {
                sign_idx = feature;
                sign_abs = candidate;
            }
        }
        if (x_weights[sign_idx] < 0.0) {
            for (double& value : x_weights) {
                value = -value;
            }
            for (double& value : x_score) {
                value = -value;
            }
            for (double& value : y_loadings) {
                value = -value;
            }
        }

        const double x_score_ss = squared_norm(x_score);
        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_loading_ss = squared_norm(y_loadings);
        if (y_loading_ss > std::numeric_limits<double>::epsilon()) {
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_loadings[target];
                }
                y_scores[row] = sum / y_loading_ss;
            }
        }

        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
            for (std::size_t target = 0; target < q; ++target) {
                yk[idx(row, q, target)] -= x_score[row] * y_loadings[target];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_regression_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> covariance;
        resize_fill(covariance, p * q, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += xk[idx(row, p, feature)] * yk[idx(row, q, target)];
                }
                covariance[idx(feature, q, target)] = sum;
            }
        }

        std::vector<double> x_weights;
        std::vector<double> y_weights;
        status = dominant_svd_pair(ctx, covariance, p, q, cfg.max_iter, cfg.tol,
                                   comp, x_weights, y_weights);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SVD X score vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_weight_ss = squared_norm(y_weights);
        if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SVD Y weights vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double y_score_denom = y_weight_ss + std::numeric_limits<double>::epsilon();
        for (std::size_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                sum += yk[idx(row, q, target)] * y_weights[target];
            }
            y_scores[row] = sum / y_score_denom;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
        }

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * yk[idx(row, q, target)];
            }
            y_loadings[target] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t target = 0; target < q; ++target) {
                yk[idx(row, q, target)] -= x_score[row] * y_loadings[target];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_regression_power(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> covariance;
        resize_fill(covariance, p * q, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += xk[idx(row, p, feature)] * yk[idx(row, q, target)];
                }
                covariance[idx(feature, q, target)] = sum;
            }
        }

        std::vector<double> x_weights;
        std::vector<double> y_weights;
        status = dominant_svd_pair_power(ctx, covariance, p, q, cfg.max_iter, cfg.tol,
                                         comp, x_weights, y_weights);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("power PLS X score vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_weight_ss = squared_norm(y_weights);
        if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("power PLS Y weights vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double y_score_denom = y_weight_ss + std::numeric_limits<double>::epsilon();
        for (std::size_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                sum += yk[idx(row, q, target)] * y_weights[target];
            }
            y_scores[row] = sum / y_score_denom;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
        }

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * yk[idx(row, q, target)];
            }
            y_loadings[target] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t target = 0; target < q; ++target) {
                yk[idx(row, q, target)] -= x_score[row] * y_loadings[target];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_regression_randomized_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    const std::uint64_t seed = ctx.seed();
    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> covariance;
        resize_fill(covariance, p * q, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += xk[idx(row, p, feature)] * yk[idx(row, q, target)];
                }
                covariance[idx(feature, q, target)] = sum;
            }
        }

        std::vector<double> x_weights;
        std::vector<double> y_weights;
        status = dominant_svd_pair_randomized(ctx, covariance, p, q, cfg.max_iter,
                                              cfg.tol, seed, comp,
                                              x_weights, y_weights);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("randomized SVD PLS X score vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_weight_ss = squared_norm(y_weights);
        if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("randomized SVD PLS Y weights vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double y_score_denom = y_weight_ss + std::numeric_limits<double>::epsilon();
        for (std::size_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                sum += yk[idx(row, q, target)] * y_weights[target];
            }
            y_scores[row] = sum / y_score_denom;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
        }

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * yk[idx(row, q, target)];
            }
            y_loadings[target] = sum / x_score_ss;
        }
        N4M_PARALLEL_FOR_STATIC
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t target = 0; target < q; ++target) {
                yk[idx(row, q, target)] -= x_score[row] * y_loadings[target];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = x_weights[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pcr_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xs;
    std::vector<double> ys;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xs);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, ys);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    // ------------------------------------------------------------------
    // Single-shot PCR via one eigendecomposition of (Xc^T Xc).
    //
    // Mathematically equivalent to the per-component "build Gram + extract
    // top eigenvector + deflate X" loop, but at a fraction of the cost:
    //   * One Gram build  (p x p) via gemm instead of `a` rebuilds.
    //   * One symmetric Jacobi diagonalization that yields ALL eigenpairs
    //     in a single sweep instead of `a` invocations that each do a full
    //     diagonalization just to read off one eigenvector.
    //   * No X deflation: because (Xc^T Xc) eigenvectors are orthonormal,
    //     deflated x_score_i is exactly sigma_i * U[:, i] and x_loadings_i
    //     reduces to V[:, i] = direction_i (closed form, no inner loop).
    // ------------------------------------------------------------------
    if (p == 0U || a == 0U) {
        ctx.set_errorf("PCR requires positive feature and component counts");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (a > p) {
        ctx.set_errorf("PCR n_components (%llu) exceeds feature count (%llu)",
                       ull(a), ull(p));
        return N4M_ERR_INVALID_ARGUMENT;
    }

    // Gram = Xc^T Xc  (symmetric, p x p, row-major contiguous).
    std::vector<double> gram;
    resize_fill(gram, p * p, 0.0);
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        p, p, n,
        1.0,
        xs.data(), p,
        xs.data(), p,
        0.0,
        gram.data(), p);

    // Full symmetric cyclic Jacobi diagonalization. After convergence:
    //   mat[i,i]      = eigenvalues (unsorted)
    //   eigvecs[:, i] = corresponding orthonormal eigenvector
    //
    // Sweeps walk every off-diagonal (row, col > row) in fixed order
    // instead of paying O(p^2) to relocate the max-magnitude entry every
    // single rotation. Cyclic Jacobi converges quadratically and needs
    // only O(log p) sweeps for double-precision eigendecompositions of
    // well-conditioned symmetric matrices; the sweep cap is set to 64
    // which is far above the textbook 10-12 needed in practice.
    std::vector<double> mat = gram;
    std::vector<double> eigvecs;
    resize_fill(eigvecs, p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        eigvecs[idx(i, p, i)] = 1.0;
    }
    const double stop_tol = std::min(std::max(cfg.tol, 1e-14), 1e-12);
    constexpr std::size_t kMaxSweeps = 64U;
    bool converged = (p <= 1U);
    if (p >= 2U) {
        for (std::size_t sweep = 0; sweep < kMaxSweeps && !converged; ++sweep) {
            double max_off = 0.0;
            double diag_scale = 1.0;
            for (std::size_t row = 0; row < p; ++row) {
                diag_scale = std::max(diag_scale, std::fabs(mat[idx(row, p, row)]));
                for (std::size_t col = row + 1U; col < p; ++col) {
                    max_off = std::max(max_off, std::fabs(mat[idx(row, p, col)]));
                }
            }
            const double threshold = stop_tol * diag_scale;
            if (max_off <= threshold) {
                converged = true;
                break;
            }
            for (std::size_t pivot_p = 0; pivot_p < p; ++pivot_p) {
                for (std::size_t pivot_q = pivot_p + 1U; pivot_q < p; ++pivot_q) {
                    const double apq = mat[idx(pivot_p, p, pivot_q)];
                    if (std::fabs(apq) <= threshold) {
                        continue;
                    }
                    const double app = mat[idx(pivot_p, p, pivot_p)];
                    const double aqq = mat[idx(pivot_q, p, pivot_q)];
                    const double tau = (aqq - app) / (2.0 * apq);
                    const double tau_sign = tau >= 0.0 ? 1.0 : -1.0;
                    const double t = tau_sign /
                                     (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                    const double c = 1.0 / std::sqrt(1.0 + t * t);
                    const double s = t * c;

                    mat[idx(pivot_p, p, pivot_p)] = app - t * apq;
                    mat[idx(pivot_q, p, pivot_q)] = aqq + t * apq;
                    mat[idx(pivot_p, p, pivot_q)] = 0.0;
                    mat[idx(pivot_q, p, pivot_p)] = 0.0;

                    for (std::size_t k = 0; k < p; ++k) {
                        if (k == pivot_p || k == pivot_q) {
                            continue;
                        }
                        const double akp = mat[idx(k, p, pivot_p)];
                        const double akq = mat[idx(k, p, pivot_q)];
                        const double next_p = c * akp - s * akq;
                        const double next_q = s * akp + c * akq;
                        mat[idx(k, p, pivot_p)] = next_p;
                        mat[idx(pivot_p, p, k)] = next_p;
                        mat[idx(k, p, pivot_q)] = next_q;
                        mat[idx(pivot_q, p, k)] = next_q;
                    }

                    for (std::size_t row = 0; row < p; ++row) {
                        const double vip = eigvecs[idx(row, p, pivot_p)];
                        const double viq = eigvecs[idx(row, p, pivot_q)];
                        eigvecs[idx(row, p, pivot_p)] = c * vip - s * viq;
                        eigvecs[idx(row, p, pivot_q)] = s * vip + c * viq;
                    }
                }
            }
        }
    }
    if (!converged) {
        ctx.set_error("PCR X covariance Jacobi diagonalization failed to converge");
        return N4M_ERR_CONVERGENCE_FAILED;
    }

    // Collect column indices in descending-eigenvalue order.
    std::vector<std::size_t> order(p);
    for (std::size_t i = 0; i < p; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  return mat[idx(lhs, p, lhs)] > mat[idx(rhs, p, rhs)];
              });

    // For each retained component, collect the eigenvector directions.
    // Then batch the PCR projections through GEMM:
    //   scores_t     = Xc * directions
    //   y_loadings_q = Yc^T * scores_t / eigenvalue
    // This preserves the same math as per-component GEMV while avoiding
    // repeated dispatch overhead and enabling BLAS/CUDA to see a wider op.
    // weights_w == loadings_p == V[:, i] in closed form because each
    // eigenvector is orthonormal and the per-component deflation reduces
    // to the identity in the (Xc^T Xc) eigenbasis.
    std::vector<double> direction(p, 0.0);
    std::vector<double> directions;
    std::vector<double> eigenvalues;
    resize_fill(directions, p * a, 0.0);
    resize_fill(eigenvalues, a, 0.0);
    for (std::size_t comp = 0; comp < a; ++comp) {
        const std::size_t col = order[comp];
        const double eigenvalue = mat[idx(col, p, col)];
        if (!(eigenvalue > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("PCR X covariance vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (std::size_t feature = 0; feature < p; ++feature) {
            direction[feature] = eigvecs[idx(feature, p, col)];
        }
        canonicalize_direction_sign(direction);
        eigenvalues[comp] = eigenvalue;
        for (std::size_t feature = 0; feature < p; ++feature) {
            directions[idx(feature, a, comp)] = direction[feature];
            model->weights_w[idx(feature, a, comp)] = direction[feature];
            model->loadings_p[idx(feature, a, comp)] = direction[feature];
        }
    }

    n4m::linalg::gemm(
        n4m::linalg::Trans_No, n4m::linalg::Trans_No,
        n, a, p,
        1.0,
        xs.data(), p,
        directions.data(), a,
        0.0,
        scores_t.data(), a);
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        q, a, n,
        1.0,
        ys.data(), q,
        scores_t.data(), a,
        0.0,
        model->y_loadings_q.data(), a);
    for (std::size_t comp = 0; comp < a; ++comp) {
        const double inv_score_ss = 1.0 / eigenvalues[comp];
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] *= inv_score_ss;
        }
    }

    if (cfg.store_scores != 0) {
        resize_fill(y_scores_u, n * a, 0.0);
        std::vector<double> scaled_y_loadings;
        resize_fill(scaled_y_loadings, q * a, 0.0);
        for (std::size_t comp = 0; comp < a; ++comp) {
            double y_loading_ss = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                const double value = model->y_loadings_q[idx(target, a, comp)];
                y_loading_ss += value * value;
            }
            if (y_loading_ss > std::numeric_limits<double>::epsilon()) {
                const double inv = 1.0 / y_loading_ss;
                for (std::size_t target = 0; target < q; ++target) {
                    scaled_y_loadings[idx(target, a, comp)] =
                        model->y_loadings_q[idx(target, a, comp)] * inv;
                }
            }
        }
        n4m::linalg::gemm(
            n4m::linalg::Trans_No, n4m::linalg::Trans_No,
            n, a, q,
            1.0,
            ys.data(), q,
            scaled_y_loadings.data(), a,
            0.0,
            y_scores_u.data(), a);
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_regression_simpls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    std::vector<double> covariance;
    resize_fill(covariance, p * q, 0.0);
    // Cross-covariance: covariance = X^T * Y  (p x q).
    // X is (n x p) row-major; Y is (n x q) row-major.
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        p, q, n,
        1.0,
        xk.data(), p,
        yk.data(), q,
        0.0,
        covariance.data(), q);

    std::vector<double> basis_v;
    resize_fill(basis_v, p * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> direction;
        status = dominant_left_singular_direction(ctx, covariance, p, q, cfg.max_iter,
                                                  cfg.tol, comp, direction);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, direction, x_score);
        const double x_score_norm = std::sqrt(squared_norm(x_score));
        if (x_score_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SIMPLS X score vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : x_score) {
            value /= x_score_norm;
        }
        for (double& value : direction) {
            value /= x_score_norm;
        }

        // x_loadings = X^T * x_score  (O(n*p) — was a hand-written
        // strided loop walking xk column-by-column; replaced with a
        // single BLAS gemv via linalg::gemv when libn4m is built with
        // PLS4ALL_WITH_BLAS, which closes the perf gap vs sklearn at
        // mid-to-large sizes. Scalar fallback in linalg::gemv preserves
        // bit-for-bit parity when BLAS is unavailable, and BLAS-vs-
        // scalar diff is the same class already accepted on the cross-
        // covariance gemm earlier in this function.
        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes, n, p, 1.0,
            xk.data(), x_score.data(), 0.0, x_loadings.data());

        // y_loadings = Y^T * x_score  (O(n*q) — same pattern; marginal
        // for PLS1 but kept symmetric so the SIMPLS-family hot path is
        // fully BLAS-routed).
        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes, n, q, 1.0,
            yk.data(), x_score.data(), 0.0, y_loadings.data());

        std::vector<double> v = x_loadings;
        if (comp > 0U) {
            for (std::size_t prev = 0; prev < comp; ++prev) {
                double projection = 0.0;
                for (std::size_t feature = 0; feature < p; ++feature) {
                    projection += basis_v[idx(feature, a, prev)] * v[feature];
                }
                for (std::size_t feature = 0; feature < p; ++feature) {
                    v[feature] -= basis_v[idx(feature, a, prev)] * projection;
                }
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SIMPLS deflation basis vanished at component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : v) {
            value /= v_norm;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_loading_ss = squared_norm(y_loadings);
        if (y_loading_ss > std::numeric_limits<double>::epsilon()) {
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_loadings[target];
                }
                y_scores[row] = sum / y_loading_ss;
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = direction[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
            basis_v[idx(feature, a, comp)] = v[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }

        std::vector<double> v_transpose_s;
        resize_fill(v_transpose_s, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += v[feature] * covariance[idx(feature, q, target)];
            }
            v_transpose_s[target] = sum;
        }
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                covariance[idx(feature, q, target)] -=
                    v[feature] * v_transpose_s[target];
            }
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t predict_into(Context& ctx,
                          const Model& model,
                          const n4m_matrix_view_t& X,
                          n4m_matrix_view_t& out) {
    const n4m_status_t status =
        validate_apply_input(ctx, model, X, out,
                             static_cast<std::int64_t>(model.n_targets),
                             "predict");
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t q = static_cast<std::size_t>(model.n_targets);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            double sum = model.y_mean[t];
            for (std::size_t j = 0; j < p; ++j) {
                const double x_value = read_value(X, i, j);
                if (!std::isfinite(x_value)) {
                    ctx.set_errorf("X contains NaN or Inf at row %llu col %llu",
                                   ull(i),
                                   ull(j));
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                sum += (x_value - model.x_mean[j]) *
                       model.coefficients[idx(j, q, t)];
            }
            write_value(out, i, t, sum);
        }
    }
    ctx.clear_error();
    return N4M_OK;
}

// Pre-0.97.4 di-PLS: SIMPLS direction with a scalar `lambda/(1+lambda)`
// projection along the (mu_source - mu_target) difference vector. Kept
// behind `cfg.di_pls_legacy = 1` for reproducibility of older bundles and
// for the di_pls_phase13 unit tests that exercise this specific geometry.
static n4m_status_t fit_di_pls_legacy(Context& ctx,
                                       const Config& cfg,
                                       const n4m_matrix_view_t& X_source,
                                       const n4m_matrix_view_t& Y_source,
                                       const n4m_matrix_view_t& X_target,
                                       std::unique_ptr<Model>& out_model) {
    Config local = cfg;
    local.algorithm = N4M_ALGO_PLS_REGRESSION;

    n4m_status_t status = validate_fit_request(ctx, local, X_source, Y_source);
    if (status != N4M_OK) return status;

    std::vector<double> x_source_buf;
    std::vector<double> y_source_buf;
    std::vector<double> x_target_buf;
    status = copy_matrix_checked(ctx, X_source, "X_source", x_source_buf);
    if (status != N4M_OK) return status;
    status = copy_matrix_checked(ctx, Y_source, "Y_source", y_source_buf);
    if (status != N4M_OK) return status;
    status = copy_matrix_checked(ctx, X_target, "X_target", x_target_buf);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X_source.rows);
    const std::size_t n_target = static_cast<std::size_t>(X_target.rows);
    const std::size_t p = static_cast<std::size_t>(X_source.cols);
    const std::size_t q = static_cast<std::size_t>(Y_source.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);
    const double lambda = cfg.di_lambda;

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X_source.rows;
    model->n_features = static_cast<std::int32_t>(X_source.cols);
    model->n_targets = static_cast<std::int32_t>(Y_source.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_source_buf, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_source_buf, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    // Compute target mean and the source-vs-target difference vector d
    // (in centered source space).
    std::vector<double> target_mean(p, 0.0);
    for (std::size_t row = 0; row < n_target; ++row) {
        for (std::size_t feature = 0; feature < p; ++feature) {
            target_mean[feature] += x_target_buf[idx(row, p, feature)];
        }
    }
    const double n_target_inv = (n_target > 0)
        ? 1.0 / static_cast<double>(n_target) : 0.0;
    for (double& value : target_mean) value *= n_target_inv;
    std::vector<double> diff(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        diff[feature] = model->x_mean[feature] - target_mean[feature];
    }
    double diff_ss = squared_norm(diff);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    std::vector<double> covariance;
    resize_fill(covariance, p * q, 0.0);
    // Cross-covariance: covariance = X^T * Y  (p x q).
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        p, q, n,
        1.0,
        xk.data(), p,
        yk.data(), q,
        0.0,
        covariance.data(), q);

    std::vector<double> basis_v;
    resize_fill(basis_v, p * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> direction;
        status = dominant_left_singular_direction(ctx, covariance, p, q,
                                                  cfg.max_iter, cfg.tol,
                                                  comp, direction);
        if (status != N4M_OK) return status;

        // DI step: remove a fraction of the direction's component along
        // the source-target difference. The fraction is bounded in
        // [0, 1] and equals lambda / (1 + lambda) so lambda=0 leaves the
        // direction unchanged and lambda → ∞ projects it onto the
        // orthogonal complement of `diff`.
        if (lambda > 0.0 && diff_ss >
            std::numeric_limits<double>::epsilon()) {
            double projection = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                projection += direction[feature] * diff[feature];
            }
            const double frac = lambda / (1.0 + lambda);
            const double scale = frac * (projection / diff_ss);
            for (std::size_t feature = 0; feature < p; ++feature) {
                direction[feature] -= scale * diff[feature];
            }
            const double norm = std::sqrt(squared_norm(direction));
            if (norm <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf(
                    "DI-PLS direction collapsed at component %llu", ull(comp));
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            for (double& value : direction) value /= norm;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, direction, x_score);
        const double x_score_norm = std::sqrt(squared_norm(x_score));
        if (x_score_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("DI-PLS X score vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : x_score) value /= x_score_norm;
        for (double& value : direction) value /= x_score_norm;

        std::vector<double> x_loadings(p, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes, n, p, 1.0,
            xk.data(), x_score.data(), 0.0, x_loadings.data());
        std::vector<double> y_loadings(q, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes, n, q, 1.0,
            yk.data(), x_score.data(), 0.0, y_loadings.data());

        std::vector<double> v = x_loadings;
        if (comp > 0U) {
            for (std::size_t prev = 0; prev < comp; ++prev) {
                double projection = 0.0;
                for (std::size_t feature = 0; feature < p; ++feature) {
                    projection += basis_v[idx(feature, a, prev)] * v[feature];
                }
                for (std::size_t feature = 0; feature < p; ++feature) {
                    v[feature] -= basis_v[idx(feature, a, prev)] * projection;
                }
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("DI-PLS deflation basis vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : v) value /= v_norm;

        std::vector<double> y_scores(n, 0.0);
        const double y_loading_ss = squared_norm(y_loadings);
        if (y_loading_ss > std::numeric_limits<double>::epsilon()) {
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_loadings[target];
                }
                y_scores[row] = sum / y_loading_ss;
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = direction[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
            basis_v[idx(feature, a, comp)] = v[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }

        std::vector<double> v_transpose_s(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += v[feature] * covariance[idx(feature, q, target)];
            }
            v_transpose_s[target] = sum;
        }
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                covariance[idx(feature, q, target)] -=
                    v[feature] * v_transpose_s[target];
            }
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) return status;

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

// Householder tridiagonalization of a symmetric (p x p) matrix in place.
// On entry `A` is row-major symmetric. On exit:
//   * `diag[i]` = tridiagonal diagonal
//   * `subd[i]` = subdiagonal between row i and row i+1 (length p - 1)
//   * `Q` (row-major p x p) accumulates the orthogonal transform such
//     that `A_original = Q * T * Q^T` where `T` is tridiagonal.
//
// Uses the standard Golub-Van Loan algorithm 8.3.1: at each step build a
// Householder reflector that zeros entries below the subdiagonal in column
// `k`, apply it to the trailing submatrix, and accumulate into `Q`.
// Complexity: ~4/3 p^3 flops (vs Jacobi's ~5 p^3 per sweep).
static void householder_tridiag(std::vector<double>& A,
                                 std::size_t p,
                                 std::vector<double>& diag,
                                 std::vector<double>& subd,
                                 std::vector<double>& Q) {
    diag.assign(p, 0.0);
    subd.assign(p > 0 ? p - 1 : 0, 0.0);
    Q.assign(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) Q[idx(i, p, i)] = 1.0;
    if (p <= 1U) {
        if (p == 1U) diag[0] = A[0];
        return;
    }

    std::vector<double> v(p, 0.0);
    std::vector<double> w(p, 0.0);
    for (std::size_t k = 0; k + 2U <= p; ++k) {
        const std::size_t m = p - k - 1;  // length of vector below A[k, k]
        // Compute alpha = -sign(A[k+1, k]) * ||A[k+1:p, k]||
        double sigma = 0.0;
        for (std::size_t i = k + 1; i < p; ++i) {
            const double x = A[idx(i, p, k)];
            sigma += x * x;
        }
        const double sub = A[idx(k + 1, p, k)];
        if (sigma <= std::numeric_limits<double>::min()) {
            // Already tridiagonal at this column; skip.
            continue;
        }
        const double norm_x = std::sqrt(sigma);
        const double alpha = (sub >= 0.0 ? -norm_x : norm_x);
        // v = x - alpha*e_1
        std::fill(v.begin(), v.end(), 0.0);
        for (std::size_t i = k + 1; i < p; ++i) {
            v[i] = A[idx(i, p, k)];
        }
        v[k + 1] -= alpha;
        // ||v||^2 = 2 * (sigma - alpha * sub)
        const double vtv = 2.0 * (sigma - alpha * sub);
        if (vtv <= std::numeric_limits<double>::min()) continue;
        const double beta = 2.0 / vtv;

        // Apply H = I - beta * v v^T from both sides:
        //   A := H * A * H
        // Compute p_vec = beta * A * v   (only lower trailing p - k - 1 block
        // matters; but since A is dense symmetric we work over the whole
        // trailing block A[k+1:p, k+1:p] for now). Actually since v is zero
        // above index k, A * v only touches rows/cols k+1:p.
        // 1. w = beta * A[k+1:p, k+1:p] * v[k+1:p]
        std::fill(w.begin(), w.end(), 0.0);
        for (std::size_t i = k + 1; i < p; ++i) {
            double sum = 0.0;
            for (std::size_t j = k + 1; j < p; ++j) {
                sum += A[idx(i, p, j)] * v[j];
            }
            w[i] = beta * sum;
        }
        // 2. K = beta/2 * v^T * w
        double k_val = 0.0;
        for (std::size_t i = k + 1; i < p; ++i) {
            k_val += v[i] * w[i];
        }
        k_val *= 0.5 * beta;
        for (std::size_t i = k + 1; i < p; ++i) {
            w[i] -= k_val * v[i];
        }
        // 3. A := A - v w^T - w v^T  (in trailing block)
        for (std::size_t i = k + 1; i < p; ++i) {
            const double vi = v[i];
            const double wi = w[i];
            for (std::size_t j = k + 1; j < p; ++j) {
                A[idx(i, p, j)] -= vi * w[j] + wi * v[j];
            }
        }
        // 4. Zero out below-subdiagonal entries in column k (and row k by
        //    symmetry); record subdiagonal value.
        A[idx(k + 1, p, k)] = alpha;
        A[idx(k, p, k + 1)] = alpha;
        for (std::size_t i = k + 2; i < p; ++i) {
            A[idx(i, p, k)] = 0.0;
            A[idx(k, p, i)] = 0.0;
        }
        // 5. Accumulate Q := Q * H = Q - (beta * Q * v) v^T
        //    Compute u = Q * v (length p), then Q -= beta * u v^T.
        std::fill(w.begin(), w.end(), 0.0);
        for (std::size_t i = 0; i < p; ++i) {
            double sum = 0.0;
            for (std::size_t j = k + 1; j < p; ++j) {
                sum += Q[idx(i, p, j)] * v[j];
            }
            w[i] = beta * sum;
        }
        for (std::size_t i = 0; i < p; ++i) {
            const double wi = w[i];
            for (std::size_t j = k + 1; j < p; ++j) {
                Q[idx(i, p, j)] -= wi * v[j];
            }
        }
        (void)m;
    }
    for (std::size_t i = 0; i < p; ++i) {
        diag[i] = A[idx(i, p, i)];
    }
    for (std::size_t i = 0; i + 1 < p; ++i) {
        subd[i] = A[idx(i + 1, p, i)];
    }
}

// Symmetric tridiagonal QL with implicit Wilkinson shift (Numerical
// Recipes `tqli`). Operates on the tridiagonal (`diag`, `subd`) produced
// by `householder_tridiag`, accumulating Givens rotations into `Q`
// (already initialized to the Householder transform) so that the final
// `Q` columns are the eigenvectors of the original symmetric matrix.
// On exit `diag` holds the eigenvalues (unsorted).
//
// `subd` is conceptually of length p-1 but the algorithm pads with a
// trailing zero so subd[i] is accessed for i in [0, p-1]; in our storage
// we keep subd at length p with subd[p-1] = 0.
[[nodiscard]] static n4m_status_t symmetric_qr_tridiag(
    ::n4m::core::Context& ctx,
    std::vector<double>& diag,
    std::vector<double>& subd,
    std::vector<double>& Q,
    std::size_t p,
    double tol,
    const char* label) {
    if (p <= 1U) return N4M_OK;
    // Pad subd to length p (treat the last subdiagonal as 0).
    subd.resize(p, 0.0);
    constexpr std::size_t kMaxIter = 30U;  // per-eigenvalue cap (NR)
    const double safe_tol = std::max(tol, 1e-15);

    for (std::size_t l = 0; l < p; ++l) {
        std::size_t iter = 0;
        std::size_t m = 0;
        while (true) {
            // Find m >= l with subd[m] negligible relative to the
            // surrounding diagonals.
            for (m = l; m + 1 < p; ++m) {
                const double dd = std::fabs(diag[m]) + std::fabs(diag[m + 1]);
                if (std::fabs(subd[m]) <= safe_tol * dd) {
                    break;
                }
            }
            if (m == l) break;
            if (iter++ >= kMaxIter) {
                ctx.set_errorf("%s symmetric QR failed to converge", label);
                return N4M_ERR_CONVERGENCE_FAILED;
            }
            // Wilkinson shift from the trailing 2x2 starting at l.
            double g = (diag[l + 1] - diag[l]) / (2.0 * subd[l]);
            double r = std::hypot(g, 1.0);
            const double sign_r = (g >= 0.0 ? 1.0 : -1.0);
            g = diag[m] - diag[l] + subd[l] / (g + sign_r * r);
            double s = 1.0;
            double c = 1.0;
            double pp = 0.0;
            // Implicit QL sweep, walking from index m-1 down to l.
            for (std::size_t ii = m; ii-- > l;) {
                double f = s * subd[ii];
                double b = c * subd[ii];
                r = std::hypot(f, g);
                subd[ii + 1] = r;
                if (r == 0.0) {
                    diag[ii + 1] -= pp;
                    subd[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = diag[ii + 1] - pp;
                r = (diag[ii] - g) * s + 2.0 * c * b;
                pp = s * r;
                diag[ii + 1] = g + pp;
                g = c * r - b;
                // Accumulate rotation into eigenvector matrix Q.
                for (std::size_t row = 0; row < p; ++row) {
                    const double qi = Q[idx(row, p, ii + 1)];
                    const double qj = Q[idx(row, p, ii)];
                    Q[idx(row, p, ii + 1)] = s * qj + c * qi;
                    Q[idx(row, p, ii)]     = c * qj - s * qi;
                }
                if (ii == l) break;  // bounded explicitly because ii is size_t
            }
            if (r == 0.0 && m > l) continue;
            diag[l] -= pp;
            subd[l] = g;
            subd[m] = 0.0;
        }
    }
    return N4M_OK;
}

// Full symmetric eigendecomposition: matrix -> (eigvals, eigvecs) such
// that  matrix = eigvecs * diag(eigvals) * eigvecs^T. Composes Householder
// tridiagonalization with implicit-shift symmetric QR; substantially
// faster than cyclic Jacobi at p >= a few hundred where Jacobi's many
// full-matrix sweeps dominate cost.
[[nodiscard]] static n4m_status_t symmetric_eigh(
    ::n4m::core::Context& ctx,
    std::vector<double> matrix,
    std::size_t p,
    double tol,
    const char* label,
    std::vector<double>& eigvals_out,
    std::vector<double>& eigvecs_out) {
    eigvals_out.assign(p, 0.0);
    eigvecs_out.assign(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) eigvecs_out[idx(i, p, i)] = 1.0;
    if (p == 0U) return N4M_OK;
    if (p == 1U) {
        eigvals_out[0] = matrix[0];
        return N4M_OK;
    }

    std::vector<double> diag;
    std::vector<double> subd;
    householder_tridiag(matrix, p, diag, subd, eigvecs_out);
    const n4m_status_t status =
        symmetric_qr_tridiag(ctx, diag, subd, eigvecs_out, p,
                              std::max(tol, 1e-14), label);
    if (status != N4M_OK) return status;
    eigvals_out = std::move(diag);
    return N4M_OK;
}

// Apply the diPLSlib regularizer in the eigenbasis V:
//   reg = I + alpha * V * diag(|d|) * V^T
//       = V * (I + alpha * |d|) * V^T,
//   reg^-1 * w_pls = V * (V^T w_pls / (1 + alpha * |d|)).
// `eigvecs` is (p x p) row-major with eigvecs[row, col] at idx(row, p, col)
// — column `col` is the eigenvector for eigvals[col]. `w_pls` is the input
// ordinary-PLS weight; `w_out` receives reg^-1 w_pls.
static void apply_diplslib_penalty(const std::vector<double>& eigvecs,
                                    const std::vector<double>& eigvals,
                                    const std::vector<double>& w_pls,
                                    double alpha,
                                    std::size_t p,
                                    std::vector<double>& w_out) {
    // proj = V^T * w_pls  (length p)
    std::vector<double> proj(p, 0.0);
    n4m::linalg::gemv(n4m::linalg::Trans_Yes,
                       p, p, 1.0,
                       eigvecs.data(), w_pls.data(), 0.0, proj.data());
    for (std::size_t i = 0; i < p; ++i) {
        const double denom = 1.0 + alpha * std::fabs(eigvals[i]);
        proj[i] /= denom;
    }
    // w_out = V * proj
    w_out.assign(p, 0.0);
    n4m::linalg::gemv(n4m::linalg::Trans_No,
                       p, p, 1.0,
                       eigvecs.data(), proj.data(), 0.0, w_out.data());
}

// Nikzad-Langerodi 2018 di-PLS (matches `diPLSlib.models.DIPLS` with
// `centering=True`, `rescale='Target'`, single ndarray target domain).
//
// Per component i:
//   w_pls = (y^T x) / (y^T y)                  (1 x k)
//   if lambda > 0:
//       rot = (1/ns Xs^T Xs - 1/nt Xt^T Xt)    (k x k, symmetric)
//       D   = V diag(|eigvals(rot)|) V^T       (convex relaxation)
//       reg = I + (lambda / (y^T y)) * D
//       w   = solve(reg^T, w_pls^T) = V * (V^T w_pls) / (1 + alpha |d|)
//   w  /= ||w||
//   t   = x w        ; ts = xs w ; tt = xt w
//   c   = (y^T t) / (t^T t)
//   p   = (t^T x) / (t^T t)
//   ps  = (ts^T xs) / (ts^T ts)
//   pt  = (tt^T xt) / (tt^T tt)
//   deflate: x -= t p, xs -= ts ps, xt -= tt pt, y -= t c
//
// At inference time the regression vector is `b = W * (P^T W)^{-1} * C` and
// predictions follow `yhat = (X - mu_target) @ b + mean(y_train)`. To make
// `predict_into` produce the same number we store `mu_target` in
// `model.x_mean` (instead of the source mean used by every other PLS path)
// and write `coefficients = b`, `y_mean = mean(y_train)`.
static n4m_status_t fit_di_pls_diplslib(Context& ctx,
                                         const Config& cfg,
                                         const n4m_matrix_view_t& X_source,
                                         const n4m_matrix_view_t& Y_source,
                                         const n4m_matrix_view_t& X_target,
                                         std::unique_ptr<Model>& out_model) {
    Config local = cfg;
    local.algorithm = N4M_ALGO_PLS_REGRESSION;

    n4m_status_t status = validate_fit_request(ctx, local, X_source, Y_source);
    if (status != N4M_OK) return status;

    std::vector<double> x_source_buf;
    std::vector<double> y_source_buf;
    std::vector<double> x_target_buf;
    status = copy_matrix_checked(ctx, X_source, "X_source", x_source_buf);
    if (status != N4M_OK) return status;
    status = copy_matrix_checked(ctx, Y_source, "Y_source", y_source_buf);
    if (status != N4M_OK) return status;
    status = copy_matrix_checked(ctx, X_target, "X_target", x_target_buf);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X_source.rows);
    const std::size_t n_target = static_cast<std::size_t>(X_target.rows);
    const std::size_t p = static_cast<std::size_t>(X_source.cols);
    const std::size_t q = static_cast<std::size_t>(Y_source.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);
    const double lambda = cfg.di_lambda;

    if (q != 1U) {
        ctx.set_error("DI-PLS (diPLSlib algorithm) currently supports a "
                       "single response variable (q=1)");
        return N4M_ERR_UNSUPPORTED;
    }
    if (n_target == 0U) {
        ctx.set_error("DI-PLS X_target must have at least one row");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    // diPLSlib has no notion of scaling — it strictly centers. Refuse the
    // request rather than silently producing a result that would diverge
    // from the canonical reference; tooling that needs scaled DI-PLS can
    // fall back to `cfg.di_pls_legacy = 1` (the legacy SIMPLS path applies
    // standard PLS-regression scale correction at the end).
    if (cfg.scale_x != 0 || cfg.scale_y != 0) {
        ctx.set_error("DI-PLS (diPLSlib algorithm) does not support scaling; "
                       "set cfg.scale_x = 0 and cfg.scale_y = 0, or use "
                       "cfg.di_pls_legacy = 1 for the pre-0.97.4 SIMPLS path");
        return N4M_ERR_UNSUPPORTED;
    }

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X_source.rows;
    model->n_features = static_cast<std::int32_t>(X_source.cols);
    model->n_targets = static_cast<std::int32_t>(Y_source.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    // Center xs by mu_s and xt by mu_t. y is centered by b0 = mean(y).
    // Source mean (for diagnostics + the source-centering step on x/xs).
    std::vector<double> mu_s_buf;
    std::vector<double> mu_s_scale;
    std::vector<double> xs;  // (n x p), centered by mu_s
    center_scale_in_place(x_source_buf, n, p,
                           cfg.center_x != 0, /*scale=*/false,
                           mu_s_buf, mu_s_scale, xs);

    std::vector<double> mu_t(p, 0.0);
    if (cfg.center_x != 0) {
        for (std::size_t row = 0; row < n_target; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                mu_t[feature] += x_target_buf[idx(row, p, feature)];
            }
        }
        const double inv_nt = 1.0 / static_cast<double>(n_target);
        for (double& value : mu_t) value *= inv_nt;
    }
    std::vector<double> xt(n_target * p, 0.0);
    for (std::size_t row = 0; row < n_target; ++row) {
        for (std::size_t feature = 0; feature < p; ++feature) {
            xt[idx(row, p, feature)] =
                x_target_buf[idx(row, p, feature)] - mu_t[feature];
        }
    }

    // y is (n x 1) so we keep it as a flat vector.
    std::vector<double> y_mean_buf;
    std::vector<double> y_scale_buf;
    std::vector<double> yk;
    center_scale_in_place(y_source_buf, n, q,
                           cfg.center_y != 0, /*scale=*/false,
                           y_mean_buf, y_scale_buf, yk);

    // We deflate two distinct copies of x: one as the "x" used for the
    // PLS regression (= xs in diPLSlib code), one as the "xs" used to
    // build the convex relaxation. In our setup xs == x (matches the
    // pls4all parity adapter that passes the same matrix for both).
    std::vector<double> x = xs;  // deflated alongside xs

    // Per-component buffers.
    std::vector<double> W(p * a, 0.0);
    std::vector<double> P(p * a, 0.0);
    std::vector<double> C(a, 0.0);

    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    if (cfg.store_scores != 0) {
        resize_fill(scores_t, n * a, 0.0);
        resize_fill(y_scores_u, n * a, 0.0);
    }

    std::vector<double> w_pls(p, 0.0);
    std::vector<double> w_comp(p, 0.0);
    std::vector<double> t_score(n, 0.0);
    std::vector<double> ts_score(n, 0.0);
    std::vector<double> tt_score(n_target, 0.0);
    std::vector<double> p_load(p, 0.0);
    std::vector<double> ps_load(p, 0.0);
    std::vector<double> pt_load(p, 0.0);

    // Convex-relaxation buffers (only used when lambda > 0).
    std::vector<double> rot;
    std::vector<double> eigvals;
    std::vector<double> eigvecs;

    for (std::size_t comp = 0; comp < a; ++comp) {
        // y^T y  (scalar — q == 1).
        double yty = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            yty += yk[row] * yk[row];
        }
        if (!(yty > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("DI-PLS y residual vanished at component %llu",
                            ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        // w_pls = (y^T x) / (y^T y), shape (1 x p).
        std::fill(w_pls.begin(), w_pls.end(), 0.0);
        n4m::linalg::gemv(n4m::linalg::Trans_Yes,
                          n, p, 1.0,
                          x.data(), yk.data(), 0.0, w_pls.data());
        const double inv_yty = 1.0 / yty;
        for (double& value : w_pls) value *= inv_yty;

        if (lambda > 0.0) {
            // rot = (1/ns) xs^T xs - (1/nt) xt^T xt   (k x k, symmetric).
            rot.assign(p * p, 0.0);
            n4m::linalg::gemm(n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
                               p, p, n,
                               1.0 / static_cast<double>(n),
                               xs.data(), p,
                               xs.data(), p,
                               0.0,
                               rot.data(), p);
            n4m::linalg::gemm(n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
                               p, p, n_target,
                               -1.0 / static_cast<double>(n_target),
                               xt.data(), p,
                               xt.data(), p,
                               1.0,
                               rot.data(), p);

            // Eigh tol is tightened to 1e-10 — well below the 1e-6
            // prediction parity budget but loose enough to keep the
            // implicit-shift QR sweep count near the textbook ~30 per
            // eigenvalue ceiling.
            const double eigh_tol = std::min(cfg.tol, 1e-10);
            status = symmetric_eigh(ctx, rot, p, eigh_tol,
                                     "DI-PLS convex relaxation",
                                     eigvals, eigvecs);
            if (status != N4M_OK) return status;

            const double alpha = lambda * inv_yty;
            apply_diplslib_penalty(eigvecs, eigvals, w_pls, alpha, p, w_comp);
        } else {
            w_comp = w_pls;
        }

        // Normalize w.
        const double w_norm = std::sqrt(squared_norm(w_comp));
        if (!(w_norm > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("DI-PLS direction vanished at component %llu",
                            ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : w_comp) value /= w_norm;

        // t = x w
        n4m::linalg::gemv(n4m::linalg::Trans_No, n, p, 1.0,
                          x.data(), w_comp.data(), 0.0, t_score.data());
        // ts = xs w
        n4m::linalg::gemv(n4m::linalg::Trans_No, n, p, 1.0,
                          xs.data(), w_comp.data(), 0.0, ts_score.data());
        // tt = xt w
        n4m::linalg::gemv(n4m::linalg::Trans_No, n_target, p, 1.0,
                          xt.data(), w_comp.data(), 0.0, tt_score.data());

        const double ttt = squared_norm(t_score);
        if (!(ttt > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("DI-PLS x score vanished at component %llu",
                            ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double tsts = squared_norm(ts_score);
        // tt^T tt can be zero if the algorithm exhausts target signal —
        // diPLSlib treats that by zeroing pt deflation; emulate.
        const double tttt = squared_norm(tt_score);

        // c = (y^T t) / (t^T t)
        double ytt = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            ytt += yk[row] * t_score[row];
        }
        const double c_value = ytt / ttt;

        // p = (t^T x) / (t^T t)
        n4m::linalg::gemv(n4m::linalg::Trans_Yes, n, p, 1.0,
                          x.data(), t_score.data(), 0.0, p_load.data());
        for (double& value : p_load) value /= ttt;

        // ps = (ts^T xs) / (ts^T ts)
        if (tsts > std::numeric_limits<double>::epsilon()) {
            n4m::linalg::gemv(n4m::linalg::Trans_Yes, n, p, 1.0,
                              xs.data(), ts_score.data(), 0.0, ps_load.data());
            for (double& value : ps_load) value /= tsts;
        } else {
            std::fill(ps_load.begin(), ps_load.end(), 0.0);
        }

        // pt = (tt^T xt) / (tt^T tt)
        if (tttt > std::numeric_limits<double>::epsilon()) {
            n4m::linalg::gemv(n4m::linalg::Trans_Yes, n_target, p, 1.0,
                              xt.data(), tt_score.data(), 0.0, pt_load.data());
            for (double& value : pt_load) value /= tttt;
        } else {
            std::fill(pt_load.begin(), pt_load.end(), 0.0);
        }

        // Deflate x, xs, xt, y.
        // x  -= t * p
        n4m::linalg::ger(n, p, -1.0, t_score.data(), p_load.data(),
                          x.data(), p);
        // xs -= ts * ps
        n4m::linalg::ger(n, p, -1.0, ts_score.data(), ps_load.data(),
                          xs.data(), p);
        // xt -= tt * pt
        if (tttt > std::numeric_limits<double>::epsilon()) {
            n4m::linalg::ger(n_target, p, -1.0, tt_score.data(), pt_load.data(),
                              xt.data(), p);
        }
        // y  -= t * c
        for (std::size_t row = 0; row < n; ++row) {
            yk[row] -= t_score[row] * c_value;
        }

        // Store W, P, C.
        for (std::size_t feature = 0; feature < p; ++feature) {
            W[idx(feature, a, comp)] = w_comp[feature];
            P[idx(feature, a, comp)] = p_load[feature];
        }
        C[comp] = c_value;

        if (cfg.store_scores != 0) {
            for (std::size_t row = 0; row < n; ++row) {
                scores_t[idx(row, a, comp)] = t_score[row];
            }
        }
    }

    // b = W * inv(P^T W) * C   -- shape (p x 1).
    // First build P^T W (a x a), then invert.
    std::vector<double> ptw(a * a, 0.0);
    for (std::size_t row_comp = 0; row_comp < a; ++row_comp) {
        for (std::size_t col_comp = 0; col_comp < a; ++col_comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += P[idx(feature, a, row_comp)] * W[idx(feature, a, col_comp)];
            }
            ptw[idx(row_comp, a, col_comp)] = sum;
        }
    }
    std::vector<double> ptw_inv;
    if (!invert_square(ptw, a, ptw_inv)) {
        ctx.set_error("DI-PLS failed to invert P^T W");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    // rotations = W * ptw_inv   (p x a)
    std::vector<double> rotations(p * a, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < a; ++inner) {
                sum += W[idx(feature, a, inner)] * ptw_inv[idx(inner, a, comp)];
            }
            rotations[idx(feature, a, comp)] = sum;
        }
    }
    // b = rotations * C   (p x 1)
    std::vector<double> b(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double sum = 0.0;
        for (std::size_t comp = 0; comp < a; ++comp) {
            sum += rotations[idx(feature, a, comp)] * C[comp];
        }
        b[feature] = sum;
    }

    // Predictions follow `(X - mu_t) @ b + mean(y_train)`. Store mu_t in
    // x_mean (overriding the source-mean populated by center_scale_in_place
    // on `mu_s_buf`) so that `predict_into` reproduces diPLSlib's
    // `rescale='Target'` exactly. y_mean already carries mean(y_train) (q=1).
    model->x_mean = (cfg.center_x != 0) ? mu_t : std::vector<double>(p, 0.0);
    model->x_scale.assign(p, 1.0);
    model->y_mean = std::move(y_mean_buf);
    model->y_scale = std::move(y_scale_buf);

    // Store fitted matrices on the model in the same row-major (p x a)
    // layout used by every other PLS path so downstream code (the SHAP
    // wrapper, bundle export, scores accessors) Just Works.
    model->weights_w = std::move(W);
    model->loadings_p = std::move(P);

    // Coefficients (p x q == p x 1) with the standard `b * y_scale /
    // x_scale` correction — but here x_scale = 1 and y_scale = 1 (we
    // disabled scale_x/scale_y above), so it's just `b`.
    model->coefficients.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        model->coefficients[feature] = b[feature];
    }

    model->rotations_r = std::move(rotations);

    // y_loadings q is (q x a); for q==1 this is just C[i] per component.
    model->y_loadings_q.assign(a, 0.0);
    for (std::size_t comp = 0; comp < a; ++comp) {
        model->y_loadings_q[idx(0, a, comp)] = C[comp];
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        // y_scores_u left empty; diPLSlib doesn't expose them.
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_di_pls(Context& ctx,
                         const Config& cfg,
                         const n4m_matrix_view_t& X_source,
                         const n4m_matrix_view_t& Y_source,
                         const n4m_matrix_view_t& X_target,
                         std::unique_ptr<Model>& out_model) {
    out_model.reset();
    if (cfg.solver != N4M_SOLVER_SIMPLS) {
        ctx.set_error("DI-PLS currently requires the SIMPLS solver");
        return N4M_ERR_UNSUPPORTED;
    }
    if (cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("DI-PLS requires regression deflation");
        return N4M_ERR_UNSUPPORTED;
    }
    if (!(cfg.di_lambda >= 0.0) || !std::isfinite(cfg.di_lambda)) {
        ctx.set_error("di_lambda must be >= 0 and finite");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X_source.cols != X_target.cols) {
        ctx.set_error("X_source and X_target must have the same number of columns");
        return N4M_ERR_SHAPE_MISMATCH;
    }

    if (cfg.di_pls_legacy != 0) {
        return fit_di_pls_legacy(ctx, cfg, X_source, Y_source, X_target,
                                  out_model);
    }
    return fit_di_pls_diplslib(ctx, cfg, X_source, Y_source, X_target,
                                out_model);
}

// Pre-0.97.4 sparse SIMPLS: soft-threshold the dominant left-singular
// direction of the running X^T Y at every component with the absolute
// `sparsity_lambda`, then renormalize. Kept opt-in via
// `cfg.sparse_simpls_legacy` for reproducibility of older bundles; the
// default path is the Chun & Keles spls algorithm in
// `fit_pls_sparse_simpls_chun_keles`, which matches the R `spls`
// package.
static n4m_status_t fit_pls_sparse_simpls_legacy(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) return status;
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);
    const double lambda = cfg.sparsity_lambda;

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    std::vector<double> xk;
    std::vector<double> yk;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yk);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    std::vector<double> covariance;
    resize_fill(covariance, p * q, 0.0);
    // Cross-covariance: covariance = X^T * Y  (p x q).
    // X is (n x p) row-major; Y is (n x q) row-major.
    n4m::linalg::gemm(
        n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
        p, q, n,
        1.0,
        xk.data(), p,
        yk.data(), q,
        0.0,
        covariance.data(), q);

    std::vector<double> basis_v;
    resize_fill(basis_v, p * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> direction;
        status = dominant_left_singular_direction(ctx, covariance, p, q,
                                                  cfg.max_iter, cfg.tol,
                                                  comp, direction);
        if (status != N4M_OK) return status;

        // Sparse step: soft-threshold direction with lambda, renormalize.
        for (double& value : direction) {
            const double abs_v = std::fabs(value);
            value = (abs_v > lambda) ? std::copysign(abs_v - lambda, value) : 0.0;
        }
        double dir_norm = std::sqrt(squared_norm(direction));
        if (dir_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf(
                "sparse SIMPLS direction collapsed to zero at component %llu "
                "(lambda %.3e too large for this dataset)",
                ull(comp), lambda);
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : direction) value /= dir_norm;

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, direction, x_score);
        const double x_score_norm = std::sqrt(squared_norm(x_score));
        if (x_score_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("sparse SIMPLS X score vanished at component %llu",
                           ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : x_score) value /= x_score_norm;
        for (double& value : direction) value /= x_score_norm;

        // x_loadings = X^T * x_score  (O(n*p) — was a hand-written
        // strided loop walking xk column-by-column; replaced with a
        // single BLAS gemv via linalg::gemv when libn4m is built with
        // PLS4ALL_WITH_BLAS, which closes the perf gap vs sklearn at
        // mid-to-large sizes. Scalar fallback in linalg::gemv preserves
        // bit-for-bit parity when BLAS is unavailable, and BLAS-vs-
        // scalar diff is the same class already accepted on the cross-
        // covariance gemm earlier in this function.
        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes, n, p, 1.0,
            xk.data(), x_score.data(), 0.0, x_loadings.data());

        // y_loadings = Y^T * x_score  (O(n*q) — same pattern; marginal
        // for PLS1 but kept symmetric so the SIMPLS-family hot path is
        // fully BLAS-routed).
        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_Yes, n, q, 1.0,
            yk.data(), x_score.data(), 0.0, y_loadings.data());

        std::vector<double> v = x_loadings;
        if (comp > 0U) {
            for (std::size_t prev = 0; prev < comp; ++prev) {
                double projection = 0.0;
                for (std::size_t feature = 0; feature < p; ++feature) {
                    projection += basis_v[idx(feature, a, prev)] * v[feature];
                }
                for (std::size_t feature = 0; feature < p; ++feature) {
                    v[feature] -= basis_v[idx(feature, a, prev)] * projection;
                }
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("sparse SIMPLS deflation basis vanished at "
                           "component %llu", ull(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : v) value /= v_norm;

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_loading_ss = squared_norm(y_loadings);
        if (y_loading_ss > std::numeric_limits<double>::epsilon()) {
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += yk[idx(row, q, target)] * y_loadings[target];
                }
                y_scores[row] = sum / y_loading_ss;
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = direction[feature];
            model->loadings_p[idx(feature, a, comp)] = x_loadings[feature];
            basis_v[idx(feature, a, comp)] = v[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            model->y_loadings_q[idx(target, a, comp)] = y_loadings[target];
        }
        for (std::size_t row = 0; row < n; ++row) {
            scores_t[idx(row, a, comp)] = x_score[row];
            y_scores_u[idx(row, a, comp)] = y_scores[row];
        }

        std::vector<double> v_transpose_s;
        resize_fill(v_transpose_s, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += v[feature] * covariance[idx(feature, q, target)];
            }
            v_transpose_s[target] = sum;
        }
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                covariance[idx(feature, q, target)] -=
                    v[feature] * v_transpose_s[target];
            }
        }
    }

    status = compute_rotations_and_coefficients(ctx, *model, p, q, a);
    if (status != N4M_OK) return status;

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

// --- Chun & Keles 2010 sparse PLS (matches R `spls::spls`) --------------
//
// The R `spls` algorithm is variable-selection + ordinary PLS rather than
// a per-component soft-threshold of the SIMPLS recurrence. Concretely,
// for each k in 1..K it (1) computes a sparse direction vector `what` by
// soft-thresholding `Z = X1^T Y1` with `eta * max(|Z|)`, (2) unions the
// non-zero indices with the previously active set `A`, then (3) fits an
// ordinary SIMPLS regression of `Y` onto `X[, A]` with `min(k, |A|)`
// components, and (4) deflates `Y1 = Y - X @ betahat` (the `pls2` default
// `select` mode). The final regression coefficient is `betahat` from the
// last iteration, embedded as zeros for the inactive features.

namespace {

// Soft universal threshold of Chun & Keles: `ust(b, eta) = sign(b) *
// max(|b| - eta * max(|b|), 0)`. Operates in-place over a contiguous
// span; `len` is the number of doubles in `b`.
void chun_keles_ust(double* b, std::size_t len, double eta) noexcept {
    if (len == 0) return;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < len; ++i) {
        const double v = std::fabs(b[i]);
        if (v > max_abs) max_abs = v;
    }
    if (eta >= 1.0 || max_abs == 0.0) {
        // R `ust` returns zeros whenever eta >= 1 or input is identically
        // zero; mirror that.
        std::fill(b, b + len, 0.0);
        return;
    }
    const double threshold = eta * max_abs;
    for (std::size_t i = 0; i < len; ++i) {
        const double v = b[i];
        const double av = std::fabs(v);
        if (av > threshold) {
            b[i] = std::copysign(av - threshold, v);
        } else {
            b[i] = 0.0;
        }
    }
}

// Sparse direction vector from Z (p x q): median-normalize, then either
// apply `ust` (q == 1) or run the kappa=0.5 power iteration (q > 1).
// Writes the p-vector into `out`.
n4m_status_t chun_keles_spls_dv(
    Context& ctx,
    const std::vector<double>& Z,
    std::size_t p,
    std::size_t q,
    double eta,
    double eps,
    std::int32_t max_iter,
    std::vector<double>& out) {
    resize_fill(out, p, 0.0);
    if (p == 0 || q == 0) return N4M_OK;

    // Median of |Z| — R `spls.dv` divides by this to keep the threshold
    // scale-invariant. R's `median` averages the two middle elements for
    // even-length inputs; replicate that to stay bit-for-bit.
    std::vector<double> abs_z;
    abs_z.reserve(p * q);
    for (double value : Z) {
        abs_z.push_back(std::fabs(value));
    }
    std::sort(abs_z.begin(), abs_z.end());
    double median_abs;
    if (abs_z.empty()) {
        median_abs = 0.0;
    } else if ((abs_z.size() % 2U) == 1U) {
        median_abs = abs_z[abs_z.size() / 2U];
    } else {
        const std::size_t hi = abs_z.size() / 2U;
        median_abs = 0.5 * (abs_z[hi - 1U] + abs_z[hi]);
    }
    if (median_abs == 0.0 || !std::isfinite(median_abs)) {
        median_abs = 1.0;
    }

    std::vector<double> z_scaled(p * q, 0.0);
    for (std::size_t i = 0; i < p * q; ++i) {
        z_scaled[i] = Z[i] / median_abs;
    }

    if (q == 1U) {
        for (std::size_t i = 0; i < p; ++i) {
            out[i] = z_scaled[i];
        }
        chun_keles_ust(out.data(), p, eta);
        return N4M_OK;
    }

    // q > 1, kappa == 0.5 branch. R loops:
    //   M = Z Z^T  (p x p)
    //   c = rep(10, p, 1); c_old = c
    //   repeat:
    //     a = svd(M c).u %*% t(svd(M c).v)   ; for p x 1 input this is M c / |M c|.
    //     c_new = ust(M a, eta)
    //     dis  = max|c_new - c_old|;  break if dis < eps or i > maxstep
    std::vector<double> M(p * p, 0.0);
    // M = z_scaled @ z_scaled^T (p x p). z_scaled is p x q row-major.
    n4m::linalg::gemm(
        n4m::linalg::Trans_No, n4m::linalg::Trans_Yes,
        p, p, q,
        1.0,
        z_scaled.data(), q,
        z_scaled.data(), q,
        0.0,
        M.data(), p);

    std::vector<double> c_curr(p, 10.0);
    std::vector<double> c_prev = c_curr;
    std::vector<double> mc(p, 0.0);
    std::vector<double> a_vec(p, 0.0);
    const std::int32_t step_cap = (max_iter > 0) ? max_iter : 100;
    bool converged = false;
    for (std::int32_t step = 0; step < step_cap; ++step) {
        // mc = M c
        n4m::linalg::gemv(
            n4m::linalg::Trans_No, p, p, 1.0,
            M.data(), c_curr.data(), 0.0, mc.data());
        double mc_norm_sq = 0.0;
        for (double v : mc) mc_norm_sq += v * v;
        const double mc_norm = std::sqrt(mc_norm_sq);
        if (mc_norm <= std::numeric_limits<double>::epsilon()) {
            // Power iteration collapsed; stop with zeros.
            std::fill(c_curr.begin(), c_curr.end(), 0.0);
            converged = true;
            break;
        }
        const double inv_norm = 1.0 / mc_norm;
        for (std::size_t i = 0; i < p; ++i) {
            a_vec[i] = mc[i] * inv_norm;
        }
        std::vector<double> ma(p, 0.0);
        n4m::linalg::gemv(
            n4m::linalg::Trans_No, p, p, 1.0,
            M.data(), a_vec.data(), 0.0, ma.data());
        chun_keles_ust(ma.data(), p, eta);
        double dis = 0.0;
        for (std::size_t i = 0; i < p; ++i) {
            const double d = std::fabs(ma[i] - c_prev[i]);
            if (d > dis) dis = d;
        }
        c_prev = ma;
        c_curr = ma;
        if (dis < eps) {
            converged = true;
            break;
        }
    }
    (void)converged;  // R does not error on non-convergence; we accept the
                      // last iterate too.
    for (std::size_t i = 0; i < p; ++i) {
        out[i] = c_curr[i];
    }
    (void)ctx;
    return N4M_OK;
}

// Build a contiguous (n x m) row-major submatrix of `src` (n x p) by
// picking the columns listed in `cols`. `src` and `dst` are row-major
// f64 buffers; `dst` is resized to n*m.
void slice_columns(const std::vector<double>& src,
                   std::size_t n,
                   std::size_t p,
                   const std::vector<std::size_t>& cols,
                   std::vector<double>& dst) {
    const std::size_t m = cols.size();
    dst.assign(n * m, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        const double* in_row = src.data() + row * p;
        double* out_row = dst.data() + row * m;
        for (std::size_t k = 0; k < m; ++k) {
            out_row[k] = in_row[cols[k]];
        }
    }
}

}  // namespace

[[nodiscard]] static n4m_status_t fit_pls_sparse_simpls_chun_keles(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    n4m_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != N4M_OK) return status;

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != N4M_OK) return status;
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t K = static_cast<std::size_t>(cfg.n_components);
    const double eta = cfg.sparsity_lambda;

    if (eta < 0.0 || eta >= 1.0 || !std::isfinite(eta)) {
        ctx.set_errorf(
            "sparse PLS (Chun & Keles): sparsity_lambda must be in [0, 1) "
            "(got %g). For the per-component absolute-threshold variant set "
            "cfg.sparse_simpls_legacy = 1.",
            eta);
        return N4M_ERR_INVALID_ARGUMENT;
    }

    auto model = std::make_unique<Model>();
    model->algorithm = cfg.algorithm;
    model->solver = cfg.solver;
    model->deflation = cfg.deflation;
    model->n_samples = X.rows;
    model->n_features = static_cast<std::int32_t>(X.cols);
    model->n_targets = static_cast<std::int32_t>(Y.cols);
    model->n_components = cfg.n_components;
    model->center_x = cfg.center_x;
    model->scale_x = cfg.scale_x;
    model->center_y = cfg.center_y;
    model->scale_y = cfg.scale_y;
    model->store_scores = cfg.store_scores;
    model->tol = cfg.tol;
    model->max_iter = cfg.max_iter;

    // Center / scale per cfg. R `spls::spls` defaults: center X, center Y,
    // scale.x = TRUE, scale.y = FALSE. We honour cfg here so callers can
    // override; the registry uses the defaults from `n4m_config_create`
    // which match R.
    std::vector<double> xc;   // centered+scaled X (n x p)
    std::vector<double> yc;   // centered Y (n x q)
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xc);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, yc);

    // R's outer state mirrors `x1, y1` — both start as the scaled-centered
    // X, Y. y1 is deflated each iteration; x1 is unchanged in the `pls2`
    // default `select`. We keep `xc` constant and only carry `y1`.
    std::vector<double> y1 = yc;

    // betahat is in standardized-X space (p x q row-major).
    std::vector<double> betahat(p * q, 0.0);
    std::vector<bool> ever_active(p, false);
    std::vector<std::size_t> active_indices;
    std::vector<double> Z(p * q, 0.0);
    std::vector<double> what;

    const double inner_eps = (cfg.tol > 0.0) ? cfg.tol : 1e-4;
    const std::int32_t inner_max_iter =
        (cfg.max_iter > 0) ? cfg.max_iter : 100;

    for (std::size_t k = 1; k <= K; ++k) {
        // Z = xc^T y1   (p x q)
        n4m::linalg::gemm(
            n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
            p, q, n,
            1.0,
            xc.data(), p,
            y1.data(), q,
            0.0,
            Z.data(), q);

        status = chun_keles_spls_dv(ctx, Z, p, q, eta, inner_eps,
                                    inner_max_iter, what);
        if (status != N4M_OK) return status;

        // Active set: union of new non-zeros and ever-active mask.
        for (std::size_t j = 0; j < p; ++j) {
            if (what[j] != 0.0) ever_active[j] = true;
        }
        active_indices.clear();
        for (std::size_t j = 0; j < p; ++j) {
            if (ever_active[j]) active_indices.push_back(j);
        }
        if (active_indices.empty()) {
            // No features ever picked. R returns betahat=0 here; just skip
            // the inner PLS but keep iterating (subsequent steps may pick
            // features, though for pls2 deflation y1 stays unchanged and so
            // the loop is idempotent — equivalent to early-stop).
            continue;
        }

        // Build xA (n x m) row-major. m = |A|.
        const std::size_t m = active_indices.size();
        std::vector<double> xA;
        slice_columns(xc, n, p, active_indices, xA);

        // Fit ordinary SIMPLS on (xA, yc) with min(k, m) components.
        const std::int32_t ncomp = static_cast<std::int32_t>(
            std::min(k, m));
        n4m_matrix_view_t xa_view{
            xA.data(), static_cast<std::int64_t>(n), static_cast<std::int64_t>(m),
            static_cast<std::int64_t>(m), 1, N4M_DTYPE_F64, 0};
        std::vector<double> yc_copy = yc;
        n4m_matrix_view_t yc_view{
            yc_copy.data(), static_cast<std::int64_t>(n), static_cast<std::int64_t>(q),
            static_cast<std::int64_t>(q), 1, N4M_DTYPE_F64, 0};

        Config inner = cfg;
        inner.algorithm = N4M_ALGO_PLS_REGRESSION;
        inner.solver = N4M_SOLVER_SIMPLS;
        inner.deflation = N4M_DEFLATION_REGRESSION;
        inner.n_components = ncomp;
        // R `plsr(scale=FALSE)` does NOT scale, but `plsr` always centers
        // (intercept). Both inputs are already centered so re-centering is
        // idempotent; keep it on for symmetry with `plsr`.
        inner.center_x = 1;
        inner.scale_x = 0;
        inner.center_y = 1;
        inner.scale_y = 0;
        inner.store_scores = 0;
        inner.store_diagnostics = 0;
        inner.sparsity_lambda = 0.0;
        inner.sparse_simpls_legacy = 0;

        std::unique_ptr<Model> inner_model;
        status = fit_pls_regression_simpls(
            ctx, inner, xa_view, yc_view, inner_model);
        if (status != N4M_OK) return status;

        // Embed inner coefficients (m x q) back into betahat (p x q).
        std::fill(betahat.begin(), betahat.end(), 0.0);
        for (std::size_t r = 0; r < m; ++r) {
            const std::size_t j = active_indices[r];
            for (std::size_t t = 0; t < q; ++t) {
                betahat[idx(j, q, t)] =
                    inner_model->coefficients[idx(r, q, t)];
            }
        }

        // Deflate y1 = yc - xc @ betahat (pls2 default `select`).
        y1 = yc;
        n4m::linalg::gemm(
            n4m::linalg::Trans_No, n4m::linalg::Trans_No,
            n, q, p,
            -1.0,
            xc.data(), p,
            betahat.data(), q,
            1.0,
            y1.data(), q);
    }

    // Map standardized-space betahat back to raw-centered-X coefficients
    // for `predict_into` (which computes y_hat = y_mean + (X-x_mean) @
    // coefficients). With `scale_y=0` and `center_y=1`, y_mean is mu and
    // model.y_scale[t] is 1, so coefficients[j, t] = betahat[j, t] /
    // x_scale[j] reproduces R's prediction
    // (newx - meanx) / normx @ betahat[A,:] + mu.
    resize_fill(model->coefficients, p * q, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        const double inv_x_scale =
            (cfg.scale_x != 0 && model->x_scale[j] != 0.0)
                ? (1.0 / model->x_scale[j])
                : 1.0;
        for (std::size_t t = 0; t < q; ++t) {
            const double y_factor =
                (cfg.scale_y != 0) ? model->y_scale[t] : 1.0;
            model->coefficients[idx(j, q, t)] =
                betahat[idx(j, q, t)] * inv_x_scale * y_factor;
        }
    }

    // Leave weights_w / loadings_p / rotations_r / scores untouched —
    // they aren't well-defined for the variable-selection chassis (the
    // active set differs per iteration). The result handle's "weights_w"
    // entry is conditional in c_api_method_result.cpp and is omitted when
    // the vector is empty.
    (void)inner_max_iter;  // suppress unused warnings if changed later

    out_model = std::move(model);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t fit_pls_sparse_simpls(Context& ctx,
                                    const Config& cfg,
                                    const n4m_matrix_view_t& X,
                                    const n4m_matrix_view_t& Y,
                                    std::unique_ptr<Model>& out_model) {
    out_model.reset();
    if (cfg.solver != N4M_SOLVER_SIMPLS) {
        ctx.set_error("sparse PLS currently requires the SIMPLS solver");
        return N4M_ERR_UNSUPPORTED;
    }
    if (cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("sparse PLS requires regression deflation");
        return N4M_ERR_UNSUPPORTED;
    }
    if (!(cfg.sparsity_lambda >= 0.0) || !std::isfinite(cfg.sparsity_lambda)) {
        ctx.set_error("sparsity_lambda must be >= 0 and finite");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    // lambda == 0 is plain SIMPLS in both algorithms; route to the
    // standard solver so we share its test coverage.
    if (cfg.sparsity_lambda == 0.0) {
        Config relaxed = cfg;
        relaxed.algorithm = N4M_ALGO_PLS_REGRESSION;
        return fit_pls_regression_simpls(ctx, relaxed, X, Y, out_model);
    }

    if (cfg.sparse_simpls_legacy != 0) {
        return fit_pls_sparse_simpls_legacy(ctx, cfg, X, Y, out_model);
    }
    return fit_pls_sparse_simpls_chun_keles(ctx, cfg, X, Y, out_model);
}

n4m_status_t fit_model(Context& ctx,
                       const Config& cfg,
                       const n4m_matrix_view_t& X,
                       const n4m_matrix_view_t& Y,
                       std::unique_ptr<Model>& out_model) {
    if (cfg.algorithm == N4M_ALGO_SPARSE_PLS) {
        return fit_pls_sparse_simpls(ctx, cfg, X, Y, out_model);
    }
    if (cfg.algorithm == N4M_ALGO_PCR) {
        return fit_pcr_svd(ctx, cfg, X, Y, out_model);
    }
    if (cfg.algorithm == N4M_ALGO_PLS_SVD) {
        return fit_pls_svd(ctx, cfg, X, Y, out_model);
    }
    if (cfg.algorithm == N4M_ALGO_PLS_CANONICAL) {
        if (cfg.solver == N4M_SOLVER_SVD) {
            return fit_pls_canonical_svd(ctx, cfg, X, Y, out_model);
        }
        return fit_pls_canonical_nipals(ctx, cfg, X, Y, out_model);
    }
    if (cfg.algorithm == N4M_ALGO_OPLS ||
        cfg.algorithm == N4M_ALGO_OPLS_DA) {
        return fit_opls_nipals(ctx, cfg, X, Y, out_model);
    }
    if (cfg.solver == N4M_SOLVER_KERNEL_ALGORITHM ||
        cfg.solver == N4M_SOLVER_WIDE_KERNEL) {
        return fit_pls_regression_kernel(ctx, cfg, X, Y, out_model);
    }
    if (cfg.solver == N4M_SOLVER_SIMPLS) {
        return fit_pls_regression_simpls(ctx, cfg, X, Y, out_model);
    }
    if (cfg.solver == N4M_SOLVER_ORTHOGONAL_SCORES) {
        return fit_pls_regression_orthogonal_scores(ctx, cfg, X, Y, out_model);
    }
    if (cfg.solver == N4M_SOLVER_POWER) {
        return fit_pls_regression_power(ctx, cfg, X, Y, out_model);
    }
    if (cfg.solver == N4M_SOLVER_RANDOMIZED_SVD) {
        return fit_pls_regression_randomized_svd(ctx, cfg, X, Y, out_model);
    }
    if (cfg.solver == N4M_SOLVER_SVD) {
        return fit_pls_regression_svd(ctx, cfg, X, Y, out_model);
    }
    return fit_pls_regression_nipals(ctx, cfg, X, Y, out_model);
}

n4m_status_t transform_into(Context& ctx,
                            const Model& model,
                            const n4m_matrix_view_t& X,
                            n4m_matrix_view_t& out_scores) {
    const n4m_status_t status =
        validate_apply_input(ctx, model, X, out_scores,
                             static_cast<std::int64_t>(model.n_components),
                             "transform");
    if (status != N4M_OK) {
        return status;
    }

    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            double sum = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                const double x_value = read_value(X, i, j);
                if (!std::isfinite(x_value)) {
                    ctx.set_errorf("X contains NaN or Inf at row %llu col %llu",
                                   ull(i),
                                   ull(j));
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                const double normalized = (x_value - model.x_mean[j]) / model.x_scale[j];
                sum += normalized * model.rotations_r[idx(j, a, comp)];
            }
            write_value(out_scores, i, comp, sum);
        }
    }
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t model_array(Context& ctx,
                         const Model& model,
                         n4m_model_array_t which,
                         std::unique_ptr<Array>& out) {
    out.reset();

    auto make = [](std::int64_t rows,
                   std::int64_t cols,
                   const std::vector<double>& values) {
        auto arr = std::make_unique<Array>();
        arr->dtype = N4M_DTYPE_F64;
        arr->rows = rows;
        arr->cols = cols;
        arr->values = values;
        return arr;
    };

    switch (which) {
        case N4M_MODEL_COEFFICIENTS:
            out = make(model.n_features, model.n_targets, model.coefficients);
            break;
        case N4M_MODEL_INTERCEPT:
            out = make(1, model.n_targets, model.y_mean);
            break;
        case N4M_MODEL_X_MEAN:
            out = make(1, model.n_features, model.x_mean);
            break;
        case N4M_MODEL_X_SCALE:
            out = make(1, model.n_features, model.x_scale);
            break;
        case N4M_MODEL_Y_MEAN:
            out = make(1, model.n_targets, model.y_mean);
            break;
        case N4M_MODEL_Y_SCALE:
            out = make(1, model.n_targets, model.y_scale);
            break;
        case N4M_MODEL_WEIGHTS_W:
            out = make(model.n_features, model.n_components, model.weights_w);
            break;
        case N4M_MODEL_LOADINGS_P:
            out = make(model.n_features, model.n_components, model.loadings_p);
            break;
        case N4M_MODEL_Y_LOADINGS_Q:
            out = make(model.n_targets, model.n_components, model.y_loadings_q);
            break;
        case N4M_MODEL_ROTATIONS_R:
            out = make(model.n_features, model.n_components, model.rotations_r);
            break;
        case N4M_MODEL_SCORES_T:
            if (model.store_scores == 0 || model.scores_t.empty()) {
                ctx.set_error("scores_T were not stored; enable store_scores before fit");
                return N4M_ERR_UNSUPPORTED;
            }
            out = make(model.n_samples, model.n_components, model.scores_t);
            break;
        case N4M_MODEL_Y_SCORES_U:
            if (model.store_scores == 0 || model.y_scores_u.empty()) {
                ctx.set_error("y_scores_U were not stored; enable store_scores before fit");
                return N4M_ERR_UNSUPPORTED;
            }
            out = make(model.n_samples, model.n_components, model.y_scores_u);
            break;
        default:
            ctx.set_errorf("invalid model array tag %d", static_cast<int>(which));
            return N4M_ERR_INVALID_ARGUMENT;
    }

    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
