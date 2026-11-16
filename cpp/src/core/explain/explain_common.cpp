// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Shared validation and linear-algebra helpers of the explanation kernels.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/common/context.hpp"
#include "core/common/linalg.h"
#include "core/common/matrix_view.hpp"
#include "core/explain/explain_internal.hpp"

namespace n4m::explain {

n4m_status_t fail(n4m_context_t* ctx, n4m_status_t status, const char* message) noexcept {
    if (ctx != nullptr) {
        ctx->set_error(message);
    }
    return status;
}

bool checked_mul(int64_t a, int64_t b, int64_t& out) noexcept {
    if (a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool checked_add(int64_t a, int64_t b, int64_t& out) noexcept {
    if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

n4m_status_t validate_f64_view(n4m_context_t* ctx, const n4m_matrix_view_t* view,
                               const char* name, int64_t min_rows) {
    if (view == nullptr) {
        ctx->set_errorf("%s view is NULL", name);
        return N4M_ERR_NULL_POINTER;
    }
    const n4m_status_t status = n4m::core::validate_nonnull_view(*view);
    if (status != N4M_OK) {
        ctx->set_errorf("%s view is invalid (negative size, stride or NULL data)", name);
        return status;
    }
    if (view->dtype != N4M_DTYPE_F64) {
        ctx->set_errorf("%s must be an F64 view", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (view->rows < min_rows || view->cols < 1) {
        ctx->set_errorf("%s needs at least %lld row(s) and one column", name,
                        static_cast<long long>(min_rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (view->rows == 0) {
        return N4M_OK;
    }
    constexpr int64_t max_offset =
        std::numeric_limits<std::ptrdiff_t>::max() / static_cast<int64_t>(sizeof(double));
    const int64_t last_row = view->rows - 1;
    const int64_t last_col = view->cols - 1;
    if ((last_row > 0 && view->row_stride > max_offset / last_row)
        || (last_col > 0 && view->col_stride > max_offset / last_col)
        || last_row * view->row_stride > max_offset - last_col * view->col_stride) {
        ctx->set_errorf("%s strides overflow the addressable extent", name);
        return N4M_ERR_STRIDE_INVALID;
    }
    return N4M_OK;
}

bool copy_finite_rowmajor(const n4m_matrix_view_t& view, std::vector<double>& out) {
    const auto rows = static_cast<std::size_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    out.assign(rows * cols, 0.0);
    for (int64_t i = 0; i < view.rows; ++i) {
        for (int64_t j = 0; j < view.cols; ++j) {
            const double value = view_at(view, i, j);
            if (!std::isfinite(value)) {
                return false;
            }
            out[static_cast<std::size_t>(i) * cols + static_cast<std::size_t>(j)] = value;
        }
    }
    return true;
}

n4m_status_t validate_groups(n4m_context_t* ctx, const int64_t* feature_group,
                             int64_t n_features, int64_t n_groups,
                             std::vector<std::size_t>& group_of_feature) {
    const auto n_feat = static_cast<std::size_t>(n_features);
    group_of_feature.assign(n_feat, 0);
    if (feature_group == nullptr) {
        if (n_groups != n_features) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                        "feature_group is NULL, so n_groups must equal n_features");
        }
        for (std::size_t j = 0; j < n_feat; ++j) {
            group_of_feature[j] = j;
        }
        return N4M_OK;
    }
    if (n_groups < 1 || n_groups > n_features) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "n_groups must be in [1, n_features]");
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(n_groups), 0);
    for (std::size_t j = 0; j < n_feat; ++j) {
        const int64_t g = feature_group[j];
        if (g < 0 || g >= n_groups) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                        "feature_group values must lie in [0, n_groups)");
        }
        group_of_feature[j] = static_cast<std::size_t>(g);
        seen[static_cast<std::size_t>(g)] = 1;
    }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "every group must contain at least one feature");
    }
    return N4M_OK;
}

namespace {

// Factorizes `a` (n x n) in place and solves for every column of rhs.
// Returns false when R is numerically rank deficient.
bool qr_solve(std::vector<double>& a, std::size_t n, const std::vector<double>& rhs,
              std::size_t k, std::vector<double>& solution) {
    const auto n64 = static_cast<int64_t>(n);
    std::vector<double> tau(n, 0.0);
    if (n4m_householder_qr(a.data(), n64, n64, tau.data()) != N4M_OK) {
        return false;
    }
    double max_diag = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_diag = std::max(max_diag, std::fabs(a[i * n + i]));
    }
    if (!(max_diag > 0.0) || !std::isfinite(max_diag)) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(a[i * n + i]) <= 1e-12 * max_diag) {
            return false;
        }
    }
    solution.assign(n * k, 0.0);
    std::vector<double> b(n, 0.0);
    std::vector<double> x(n, 0.0);
    for (std::size_t col = 0; col < k; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            b[i] = rhs[i * k + col];
        }
        if (n4m_apply_qt(a.data(), n64, n64, tau.data(), b.data()) != N4M_OK
            || n4m_back_solve_R(a.data(), n64, n64, b.data(), x.data()) != N4M_OK) {
            return false;
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(x[i])) {
                return false;
            }
            solution[i * k + col] = x[i];
        }
    }
    return true;
}

}  // namespace

n4m_status_t solve_spd(std::vector<double> matrix, std::size_t n, const std::vector<double>& rhs,
                       std::size_t k, std::vector<double>& solution, bool& ridged) {
    ridged = false;
    std::vector<double> work = matrix;
    if (qr_solve(work, n, rhs, k, solution)) {
        return N4M_OK;
    }
    double trace = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        trace += matrix[i * n + i];
    }
    double ridge = 1e-8 * trace / static_cast<double>(n);
    if (!(ridge > 0.0) || !std::isfinite(ridge)) {
        ridge = 1e-8;
    }
    for (std::size_t i = 0; i < n; ++i) {
        matrix[i * n + i] += ridge;
    }
    ridged = true;
    if (qr_solve(matrix, n, rhs, k, solution)) {
        return N4M_OK;
    }
    return N4M_ERR_NUMERICAL_FAILURE;
}

}  // namespace n4m::explain
