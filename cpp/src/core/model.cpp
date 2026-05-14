// SPDX-License-Identifier: CeCILL-2.1

#include "core/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

constexpr double kZeroTolerance = 10.0 * std::numeric_limits<double>::epsilon();

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

[[nodiscard]] double read_value(const p4a_matrix_view_t& v,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(v.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(v.data);
    return static_cast<double>(ptr[uoff]);
}

void write_value(p4a_matrix_view_t& v,
                 std::size_t row,
                 std::size_t col,
                 double value) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == P4A_DTYPE_F64) {
        auto* ptr = static_cast<double*>(v.data);
        ptr[uoff] = value;
        return;
    }
    auto* ptr = static_cast<float*>(v.data);
    ptr[uoff] = static_cast<float>(value);
}

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& v,
                                               const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(v);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (v.dtype != P4A_DTYPE_F64 && v.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_matrix_checked(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& src,
                                               const char* name,
                                               std::vector<double>& out) {
    const std::size_t rows = static_cast<std::size_t>(src.rows);
    const std::size_t cols = static_cast<std::size_t>(src.cols);
    resize_fill(out, rows * cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double value = read_value(src, i, j);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               ull(i),
                               ull(j));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(i, cols, j)] = value;
        }
    }
    return P4A_OK;
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
        for (std::size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                sum += original[idx(i, cols, j)];
            }
            mean[j] = sum / static_cast<double>(rows);
        }
    }

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            standardized[idx(i, cols, j)] -= mean[j];
        }
    }

    if (scale_enabled) {
        for (std::size_t j = 0; j < cols; ++j) {
            double sumsq = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                const double value = standardized[idx(i, cols, j)];
                sumsq += value * value;
            }
            const double denom = static_cast<double>(rows - 1U);
            const double stddev = std::sqrt(sumsq / denom);
            scale[j] = (stddev == 0.0 || !std::isfinite(stddev)) ? 1.0 : stddev;
        }
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                standardized[idx(i, cols, j)] /= scale[j];
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
    resize_fill(out, rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            sum += matrix[idx(i, cols, j)] * vector[j];
        }
        out[i] = sum;
    }
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

[[nodiscard]] p4a_status_t largest_symmetric_eigenvector(
    ::pls4all::core::Context& ctx,
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
        return P4A_ERR_INTERNAL;
    }
    if (n == 1U) {
        eigenvalue = symmetric[0];
        eigenvector[0] = 1.0;
        if (!(eigenvalue > std::numeric_limits<double>::epsilon())) {
            ctx.set_errorf("%s covariance vanished at component %llu", label, ull(component));
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        return P4A_OK;
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
        return P4A_ERR_CONVERGENCE_FAILED;
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
        return P4A_ERR_NUMERICAL_FAILURE;
    }

    for (std::size_t i = 0; i < n; ++i) {
        eigenvector[i] = vectors[idx(i, n, best)];
    }
    const double norm = std::sqrt(squared_norm(eigenvector));
    if (norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("%s eigenvector vanished at component %llu", label, ull(component));
        return P4A_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : eigenvector) {
        value /= norm;
    }
    eigenvalue = best_value;
    return P4A_OK;
}

[[nodiscard]] p4a_status_t dominant_svd_pair(
    ::pls4all::core::Context& ctx,
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
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        const double norm = std::sqrt(norm_sq);
        for (double& value : left) {
            value /= norm;
        }
        right[0] = 1.0;
        canonicalize_svd_pair_sign(left, right);
        return P4A_OK;
    }

    double singular_value = 0.0;
    p4a_status_t status = P4A_OK;
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
        if (status != P4A_OK) {
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
        if (status != P4A_OK) {
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
        return P4A_ERR_NUMERICAL_FAILURE;
    }
    double left_norm = std::sqrt(squared_norm(left));
    double right_norm = std::sqrt(squared_norm(right));
    if (left_norm <= std::numeric_limits<double>::epsilon() ||
        right_norm <= std::numeric_limits<double>::epsilon()) {
        ctx.set_errorf("SVD singular vectors vanished at component %llu", ull(component));
        return P4A_ERR_NUMERICAL_FAILURE;
    }
    for (double& value : left) {
        value /= left_norm;
    }
    for (double& value : right) {
        value /= right_norm;
    }
    canonicalize_svd_pair_sign(left, right);
    return P4A_OK;
}

[[nodiscard]] p4a_status_t dominant_left_singular_direction(
    ::pls4all::core::Context& ctx,
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
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        canonicalize_direction_sign(out);
        return P4A_OK;
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
        return P4A_ERR_NUMERICAL_FAILURE;
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
            return P4A_ERR_NUMERICAL_FAILURE;
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
            return P4A_ERR_NUMERICAL_FAILURE;
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
        return P4A_ERR_CONVERGENCE_FAILED;
    }
    canonicalize_direction_sign(out);
    return P4A_OK;
}

[[nodiscard]] p4a_status_t compute_rotations_and_coefficients(
    ::pls4all::core::Context& ctx,
    ::pls4all::core::Model& model,
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
        return P4A_ERR_NUMERICAL_FAILURE;
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
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_fit_request(::pls4all::core::Context& ctx,
                                                const ::pls4all::core::Config& cfg,
                                                const p4a_matrix_view_t& X,
                                                const p4a_matrix_view_t& Y) noexcept {
    const bool supported_pls =
        cfg.algorithm == P4A_ALGO_PLS_REGRESSION &&
        (cfg.solver == P4A_SOLVER_NIPALS ||
         cfg.solver == P4A_SOLVER_SIMPLS ||
         cfg.solver == P4A_SOLVER_KERNEL_ALGORITHM ||
         cfg.solver == P4A_SOLVER_SVD);
    const bool supported_pcr =
        cfg.algorithm == P4A_ALGO_PCR && cfg.solver == P4A_SOLVER_SVD;
    if (!supported_pls && !supported_pcr) {
        ctx.set_error(
            "this release supports P4A_ALGO_PLS_REGRESSION with P4A_SOLVER_NIPALS, "
            "P4A_SOLVER_SIMPLS, P4A_SOLVER_KERNEL_ALGORITHM or P4A_SOLVER_SVD, "
            "plus P4A_ALGO_PCR with P4A_SOLVER_SVD");
        return P4A_ERR_UNSUPPORTED;
    }
    if (cfg.deflation != P4A_DEFLATION_REGRESSION) {
        ctx.set_error("this release supports only P4A_DEFLATION_REGRESSION");
        return P4A_ERR_UNSUPPORTED;
    }
    if (cfg.pipeline != nullptr || cfg.operator_bank != nullptr || cfg.gating_strategy != nullptr) {
        ctx.set_error("pipelines, operator banks and gating strategies are not yet supported by model_fit");
        return P4A_ERR_UNSUPPORTED;
    }

    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.dtype != Y.dtype) {
        ctx.set_error("X and Y must have the same floating dtype");
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (X.rows < 2 || X.cols < 1 || Y.cols < 1) {
        ctx.set_error("fit requires at least 2 rows, 1 X column and 1 Y column");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("matrix column count exceeds the ABI limits");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t rank_upper_bound = std::min(X.rows, X.cols);
    if (static_cast<std::int64_t>(cfg.n_components) > rank_upper_bound) {
        ctx.set_errorf("n_components upper bound is %lld; got %d",
                       static_cast<long long>(rank_upper_bound),
                       static_cast<int>(cfg.n_components));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_apply_input(::pls4all::core::Context& ctx,
                                                const ::pls4all::core::Model& model,
                                                const p4a_matrix_view_t& X,
                                                const p4a_matrix_view_t& out,
                                                std::int64_t expected_cols,
                                                const char* op) noexcept {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, out, "output");
    if (status != P4A_OK) {
        return status;
    }
    if (X.cols != static_cast<std::int64_t>(model.n_features)) {
        ctx.set_errorf("%s expected %d X features, got %lld",
                       op,
                       static_cast<int>(model.n_features),
                       static_cast<long long>(X.cols));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (out.rows != X.rows || out.cols != expected_cols) {
        ctx.set_errorf("%s output shape must be (%lld, %lld)",
                       op,
                       static_cast<long long>(X.rows),
                       static_cast<long long>(expected_cols));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    return P4A_OK;
}

void fill_prediction(const ::pls4all::core::Model& model,
                     const std::vector<double>& X,
                     std::size_t rows,
                     std::vector<double>& predictions) {
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t q = static_cast<std::size_t>(model.n_targets);
    resize_fill(predictions, rows * q, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            double sum = model.y_mean[t];
            for (std::size_t j = 0; j < p; ++j) {
                const double centered = X[idx(i, p, j)] - model.x_mean[j];
                sum += centered * model.coefficients[idx(j, q, t)];
            }
            predictions[idx(i, q, t)] = sum;
        }
    }
}

void fill_transform(const ::pls4all::core::Model& model,
                    const std::vector<double>& X,
                    std::size_t rows,
                    std::vector<double>& scores) {
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    resize_fill(scores, rows * a, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            double sum = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                const double normalized = (X[idx(i, p, j)] - model.x_mean[j]) / model.x_scale[j];
                sum += normalized * model.rotations_r[idx(j, a, comp)];
            }
            scores[idx(i, a, comp)] = sum;
        }
    }
}

void copy_to_output(const std::vector<double>& values,
                    std::size_t rows,
                    std::size_t cols,
                    p4a_matrix_view_t& out) noexcept {
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            write_value(out, i, j, values[idx(i, cols, j)]);
        }
    }
}

}  // namespace

namespace pls4all::core {

p4a_status_t fit_pls_regression_nipals(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    p4a_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != P4A_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != P4A_OK) {
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
            return P4A_ERR_NUMERICAL_FAILURE;
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
                return P4A_ERR_NUMERICAL_FAILURE;
            }

            for (std::size_t feature = 0; feature < p; ++feature) {
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row) {
                    sum += xk[idx(row, p, feature)] * y_score[row];
                }
                x_weights[feature] = sum / y_score_ss;
            }

            const double x_weight_norm = std::sqrt(squared_norm(x_weights));
            if (x_weight_norm <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("X weights vanished at component %llu", ull(comp));
                return P4A_ERR_NUMERICAL_FAILURE;
            }
            const double x_weight_denom = x_weight_norm + std::numeric_limits<double>::epsilon();
            for (double& value : x_weights) {
                value /= x_weight_denom;
            }

            matrix_vector_product(xk, n, p, x_weights, x_score);
            const double x_score_ss = squared_norm(x_score);
            if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
                ctx.set_errorf("X score vanished at component %llu", ull(comp));
                return P4A_ERR_NUMERICAL_FAILURE;
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
                return P4A_ERR_NUMERICAL_FAILURE;
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
            return P4A_ERR_CONVERGENCE_FAILED;
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
        }

        matrix_vector_product(xk, n, p, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("X score vanished after sign normalization at component %llu",
                           ull(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        const double y_weight_ss = squared_norm(y_weights);
        if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("Y weights vanished after sign normalization at component %llu",
                           ull(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
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
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }
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
    if (status != P4A_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_pls_regression_kernel(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    p4a_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != P4A_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != P4A_OK) {
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
            return P4A_ERR_NUMERICAL_FAILURE;
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
                return P4A_ERR_NUMERICAL_FAILURE;
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
                return P4A_ERR_NUMERICAL_FAILURE;
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
                return P4A_ERR_NUMERICAL_FAILURE;
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
                return P4A_ERR_NUMERICAL_FAILURE;
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
            return P4A_ERR_CONVERGENCE_FAILED;
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
    if (status != P4A_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_pls_regression_svd(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    p4a_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != P4A_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != P4A_OK) {
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
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, x_weights, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SVD X score vanished at component %llu", ull(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_weight_ss = squared_norm(y_weights);
        if (y_weight_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SVD Y weights vanished at component %llu", ull(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
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
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }
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
    if (status != P4A_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_pcr_svd(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    p4a_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != P4A_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != P4A_OK) {
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
    std::vector<double> ys;
    center_scale_in_place(x_original, n, p, cfg.center_x != 0, cfg.scale_x != 0,
                          model->x_mean, model->x_scale, xk);
    center_scale_in_place(y_original, n, q, cfg.center_y != 0, cfg.scale_y != 0,
                          model->y_mean, model->y_scale, ys);

    resize_fill(model->weights_w, p * a, 0.0);
    resize_fill(model->loadings_p, p * a, 0.0);
    resize_fill(model->y_loadings_q, q * a, 0.0);
    std::vector<double> scores_t;
    std::vector<double> y_scores_u;
    resize_fill(scores_t, n * a, 0.0);
    resize_fill(y_scores_u, n * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> gram;
        resize_fill(gram, p * p, 0.0);
        for (std::size_t row = 0; row < p; ++row) {
            for (std::size_t col = row; col < p; ++col) {
                double sum = 0.0;
                for (std::size_t sample = 0; sample < n; ++sample) {
                    sum += xk[idx(sample, p, row)] * xk[idx(sample, p, col)];
                }
                gram[idx(row, p, col)] = sum;
                gram[idx(col, p, row)] = sum;
            }
        }

        std::vector<double> direction;
        double eigenvalue = 0.0;
        status = largest_symmetric_eigenvector(ctx, gram, p, cfg.max_iter, cfg.tol,
                                               "PCR X covariance", comp,
                                               direction, eigenvalue);
        if (status != P4A_OK) {
            return status;
        }
        canonicalize_direction_sign(direction);

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, direction, x_score);
        const double x_score_ss = squared_norm(x_score);
        if (x_score_ss <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("PCR X score vanished at component %llu", ull(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * xk[idx(row, p, feature)];
            }
            x_loadings[feature] = sum / x_score_ss;
        }

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += x_score[row] * ys[idx(row, q, target)];
            }
            y_loadings[target] = sum / x_score_ss;
        }

        std::vector<double> y_scores;
        resize_fill(y_scores, n, 0.0);
        const double y_loading_ss = squared_norm(y_loadings);
        if (y_loading_ss > std::numeric_limits<double>::epsilon()) {
            for (std::size_t row = 0; row < n; ++row) {
                double sum = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    sum += ys[idx(row, q, target)] * y_loadings[target];
                }
                y_scores[row] = sum / y_loading_ss;
            }
        }

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                xk[idx(row, p, feature)] -= x_score[row] * x_loadings[feature];
            }
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            model->weights_w[idx(feature, a, comp)] = direction[feature];
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
    if (status != P4A_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_pls_regression_simpls(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model) {
    out_model.reset();

    p4a_status_t status = validate_fit_request(ctx, cfg, X, Y);
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x_original;
    std::vector<double> y_original;
    status = copy_matrix_checked(ctx, X, "X", x_original);
    if (status != P4A_OK) {
        return status;
    }
    status = copy_matrix_checked(ctx, Y, "Y", y_original);
    if (status != P4A_OK) {
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
    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += xk[idx(row, p, feature)] * yk[idx(row, q, target)];
            }
            covariance[idx(feature, q, target)] = sum;
        }
    }

    std::vector<double> basis_v;
    resize_fill(basis_v, p * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        std::vector<double> direction;
        status = dominant_left_singular_direction(ctx, covariance, p, q, cfg.max_iter,
                                                  cfg.tol, comp, direction);
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> x_score;
        matrix_vector_product(xk, n, p, direction, x_score);
        const double x_score_norm = std::sqrt(squared_norm(x_score));
        if (x_score_norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_errorf("SIMPLS X score vanished at component %llu", ull(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : x_score) {
            value /= x_score_norm;
        }
        for (double& value : direction) {
            value /= x_score_norm;
        }

        std::vector<double> x_loadings;
        resize_fill(x_loadings, p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += xk[idx(row, p, feature)] * x_score[row];
            }
            x_loadings[feature] = sum;
        }

        std::vector<double> y_loadings;
        resize_fill(y_loadings, q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                sum += yk[idx(row, q, target)] * x_score[row];
            }
            y_loadings[target] = sum;
        }

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
            return P4A_ERR_NUMERICAL_FAILURE;
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
    if (status != P4A_OK) {
        return status;
    }

    if (cfg.store_scores != 0) {
        model->scores_t = std::move(scores_t);
        model->y_scores_u = std::move(y_scores_u);
    }

    out_model = std::move(model);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t predict_into(Context& ctx,
                          const Model& model,
                          const p4a_matrix_view_t& X,
                          p4a_matrix_view_t& out) {
    const p4a_status_t status =
        validate_apply_input(ctx, model, X, out,
                             static_cast<std::int64_t>(model.n_targets),
                             "predict");
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x;
    const p4a_status_t copy_status = copy_matrix_checked(ctx, X, "X", x);
    if (copy_status != P4A_OK) {
        return copy_status;
    }

    std::vector<double> predictions;
    fill_prediction(model, x, static_cast<std::size_t>(X.rows), predictions);
    copy_to_output(predictions,
                   static_cast<std::size_t>(X.rows),
                   static_cast<std::size_t>(model.n_targets),
                   out);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t transform_into(Context& ctx,
                            const Model& model,
                            const p4a_matrix_view_t& X,
                            p4a_matrix_view_t& out_scores) {
    const p4a_status_t status =
        validate_apply_input(ctx, model, X, out_scores,
                             static_cast<std::int64_t>(model.n_components),
                             "transform");
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> x;
    const p4a_status_t copy_status = copy_matrix_checked(ctx, X, "X", x);
    if (copy_status != P4A_OK) {
        return copy_status;
    }

    std::vector<double> scores;
    fill_transform(model, x, static_cast<std::size_t>(X.rows), scores);
    copy_to_output(scores,
                   static_cast<std::size_t>(X.rows),
                   static_cast<std::size_t>(model.n_components),
                   out_scores);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t model_array(Context& ctx,
                         const Model& model,
                         p4a_model_array_t which,
                         std::unique_ptr<Array>& out) {
    out.reset();

    auto make = [](std::int64_t rows,
                   std::int64_t cols,
                   const std::vector<double>& values) {
        auto arr = std::make_unique<Array>();
        arr->dtype = P4A_DTYPE_F64;
        arr->rows = rows;
        arr->cols = cols;
        arr->values = values;
        return arr;
    };

    switch (which) {
        case P4A_MODEL_COEFFICIENTS:
            out = make(model.n_features, model.n_targets, model.coefficients);
            break;
        case P4A_MODEL_INTERCEPT:
            out = make(1, model.n_targets, model.y_mean);
            break;
        case P4A_MODEL_X_MEAN:
            out = make(1, model.n_features, model.x_mean);
            break;
        case P4A_MODEL_X_SCALE:
            out = make(1, model.n_features, model.x_scale);
            break;
        case P4A_MODEL_Y_MEAN:
            out = make(1, model.n_targets, model.y_mean);
            break;
        case P4A_MODEL_Y_SCALE:
            out = make(1, model.n_targets, model.y_scale);
            break;
        case P4A_MODEL_WEIGHTS_W:
            out = make(model.n_features, model.n_components, model.weights_w);
            break;
        case P4A_MODEL_LOADINGS_P:
            out = make(model.n_features, model.n_components, model.loadings_p);
            break;
        case P4A_MODEL_Y_LOADINGS_Q:
            out = make(model.n_targets, model.n_components, model.y_loadings_q);
            break;
        case P4A_MODEL_ROTATIONS_R:
            out = make(model.n_features, model.n_components, model.rotations_r);
            break;
        case P4A_MODEL_SCORES_T:
            if (model.store_scores == 0 || model.scores_t.empty()) {
                ctx.set_error("scores_T were not stored; enable store_scores before fit");
                return P4A_ERR_UNSUPPORTED;
            }
            out = make(model.n_samples, model.n_components, model.scores_t);
            break;
        case P4A_MODEL_Y_SCORES_U:
            if (model.store_scores == 0 || model.y_scores_u.empty()) {
                ctx.set_error("y_scores_U were not stored; enable store_scores before fit");
                return P4A_ERR_UNSUPPORTED;
            }
            out = make(model.n_samples, model.n_components, model.y_scores_u);
            break;
        default:
            ctx.set_errorf("invalid model array tag %d", static_cast<int>(which));
            return P4A_ERR_INVALID_ARGUMENT;
    }

    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
