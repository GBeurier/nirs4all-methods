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

    // Build a temporary regression Config so validate_fit_request accepts
    // the source dataset using the regression chassis.
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

        // x_loadings = X^T * x_score, y_loadings = Y^T * x_score
        // (BLAS gemv — see comment on fit_pls_regression_simpls for
        // perf/parity rationale).
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

    // When lambda == 0 the sparse path is identical to plain SIMPLS — call
    // the existing solver to avoid behavioural drift and reuse its test
    // coverage.
    if (cfg.sparsity_lambda == 0.0) {
        Config relaxed = cfg;
        relaxed.algorithm = N4M_ALGO_PLS_REGRESSION;
        return fit_pls_regression_simpls(ctx, relaxed, X, Y, out_model);
    }

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
