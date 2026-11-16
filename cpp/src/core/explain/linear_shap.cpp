// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Exact interventional SHAP of an affine model f(z) = z . B + c.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/context.hpp"
#include "core/explain/explain_internal.hpp"

namespace n4m::explain {

n4m_status_t linear_shap(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                         const n4m_matrix_view_t* coefficients, const double* intercept,
                         const double* reference, const int64_t* feature_group, int64_t n_groups,
                         double* attributions, int64_t attributions_size, double* expected,
                         int64_t expected_size, double* predictions, int64_t predictions_size) {
    n4m_status_t status = validate_f64_view(ctx, X, "X", 1);
    if (status != N4M_OK) {
        return status;
    }
    status = validate_f64_view(ctx, coefficients, "coefficients", 1);
    if (status != N4M_OK) {
        return status;
    }
    if (coefficients->rows != X->cols) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                    "coefficients must have n_features rows (one column per output)");
    }
    if (reference == nullptr || attributions == nullptr || expected == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER,
                    "reference, attributions and expected must be non-NULL");
    }
    std::vector<std::size_t> group_of_feature;
    status = validate_groups(ctx, feature_group, X->cols, n_groups, group_of_feature);
    if (status != N4M_OK) {
        return status;
    }
    int64_t n_attr = 0;
    int64_t n_out = 0;
    if (!checked_mul(X->rows, n_groups, n_attr)
        || !checked_mul(n_attr, coefficients->cols, n_attr)
        || !checked_mul(X->rows, coefficients->cols, n_out)) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "output size overflows int64");
    }
    if (attributions_size < n_attr || expected_size < n_out
        || (predictions != nullptr && predictions_size < n_out)) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                    "output buffers are smaller than n_rows * n_groups * n_outputs "
                    "(attributions) or n_rows * n_outputs (expected, predictions)");
    }
    std::vector<double> x;
    std::vector<double> beta;
    if (!copy_finite_rowmajor(*X, x) || !copy_finite_rowmajor(*coefficients, beta)) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "X and coefficients must be finite");
    }
    const auto R = static_cast<std::size_t>(X->rows);
    const auto F = static_cast<std::size_t>(X->cols);
    const auto G = static_cast<std::size_t>(n_groups);
    const auto O = static_cast<std::size_t>(coefficients->cols);
    for (std::size_t j = 0; j < F; ++j) {
        if (!std::isfinite(reference[j])) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "reference must be finite");
        }
    }
    std::vector<double> base(O, 0.0);
    for (std::size_t o = 0; o < O; ++o) {
        if (intercept != nullptr) {
            if (!std::isfinite(intercept[o])) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "intercept must be finite");
            }
            base[o] = intercept[o];
        }
    }
    std::vector<double> e(O, 0.0);
    for (std::size_t o = 0; o < O; ++o) {
        double sum = 0.0;
        for (std::size_t j = 0; j < F; ++j) {
            sum += reference[j] * beta[j * O + o];
        }
        e[o] = sum + base[o];
    }
    std::vector<double> phi(R * G * O, 0.0);
    std::vector<double> pred(R * O, 0.0);
    for (std::size_t r = 0; r < R; ++r) {
        const double* xr = &x[r * F];
        double* phi_r = &phi[r * G * O];
        for (std::size_t j = 0; j < F; ++j) {
            const double dx = xr[j] - reference[j];
            const std::size_t g = group_of_feature[j];
            for (std::size_t o = 0; o < O; ++o) {
                phi_r[g * O + o] += beta[j * O + o] * dx;
            }
        }
        for (std::size_t o = 0; o < O; ++o) {
            double sum = base[o];
            for (std::size_t j = 0; j < F; ++j) {
                sum += xr[j] * beta[j * O + o];
            }
            pred[r * O + o] = sum;
        }
    }
    std::copy(phi.begin(), phi.end(), attributions);
    for (std::size_t r = 0; r < R; ++r) {
        std::copy(e.begin(), e.end(), expected + r * O);
    }
    if (predictions != nullptr) {
        std::copy(pred.begin(), pred.end(), predictions);
    }
    return N4M_OK;
}

}  // namespace n4m::explain
