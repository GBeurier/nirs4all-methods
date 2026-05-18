// SPDX-License-Identifier: CECILL-2.1

#include "core/pls_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/matrix_view.hpp"

namespace {

inline std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

// Project rows of X (n_rows x n_features) onto the model's latent space.
// Returns row-major scores (n_rows x n_components). Uses model.rotations_r
// and model.x_mean.
void project_to_scores(const ::pls4all::core::Model& model,
                       const std::vector<double>& X,
                       std::size_t n_rows,
                       std::vector<double>& out_scores) {
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    out_scores.assign(n_rows * a, 0.0);
    for (std::size_t row = 0; row < n_rows; ++row) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                const double centered =
                    X[idx(row, p, feature)] - model.x_mean[feature];
                sum += centered * model.rotations_r[idx(feature, a, comp)];
            }
            out_scores[idx(row, a, comp)] = sum;
        }
    }
}

[[nodiscard]] ::p4a_status_t copy_x(::pls4all::core::Context& ctx,
                                    const p4a_matrix_view_t& X,
                                    const ::pls4all::core::Model& model,
                                    std::vector<double>& out) {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(X);
    if (status != P4A_OK) {
        ctx.set_error("invalid X matrix view");
        return status;
    }
    if (X.dtype != P4A_DTYPE_F64) {
        ctx.set_error("X must be float64 for diagnostics");
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (X.cols != model.n_features) {
        ctx.set_errorf("X.cols (%lld) must match model.n_features (%d)",
                       static_cast<long long>(X.cols),
                       static_cast<int>(model.n_features));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = static_cast<std::size_t>(X.cols);
    out.assign(rows * cols, 0.0);
    const auto* data = static_cast<const double*>(X.data);
    if (data == nullptr) {
        if (rows > 0 && cols > 0) {
            ctx.set_error("X.data is null with non-zero shape");
            return P4A_ERR_NULL_POINTER;
        }
        return P4A_OK;
    }
    const std::size_t row_stride = static_cast<std::size_t>(X.row_stride);
    const std::size_t col_stride = static_cast<std::size_t>(X.col_stride);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double v = data[row * row_stride + col * col_stride];
            if (!std::isfinite(v)) {
                ctx.set_error("X contains NaN or Inf");
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[row * cols + col] = v;
        }
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t pls_hotelling_t2(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t* X_reference,
    std::vector<double>& out_t2) {
    out_t2.clear();
    if (model.rotations_r.empty()) {
        ctx.set_error("model is not fitted (rotations missing)");
        return P4A_ERR_NOT_FITTED;
    }
    std::vector<double> x_buf;
    p4a_status_t status = copy_x(ctx, X, model, x_buf);
    if (status != P4A_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t a = static_cast<std::size_t>(model.n_components);

    // Reference variances per component.
    std::vector<double> score_var(a, 0.0);
    if (X_reference != nullptr) {
        std::vector<double> ref_buf;
        status = copy_x(ctx, *X_reference, model, ref_buf);
        if (status != P4A_OK) return status;
        std::vector<double> ref_scores;
        project_to_scores(model, ref_buf,
                          static_cast<std::size_t>(X_reference->rows),
                          ref_scores);
        const std::size_t n_ref = static_cast<std::size_t>(X_reference->rows);
        if (n_ref < 2) {
            ctx.set_error("Hotelling T2 reference requires at least 2 rows");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        for (std::size_t comp = 0; comp < a; ++comp) {
            double mean = 0.0;
            for (std::size_t row = 0; row < n_ref; ++row) {
                mean += ref_scores[idx(row, a, comp)];
            }
            mean /= static_cast<double>(n_ref);
            double sumsq = 0.0;
            for (std::size_t row = 0; row < n_ref; ++row) {
                const double centered = ref_scores[idx(row, a, comp)] - mean;
                sumsq += centered * centered;
            }
            score_var[comp] = sumsq / static_cast<double>(n_ref - 1);
        }
    } else if (!model.scores_t.empty()) {
        const std::size_t n_train = static_cast<std::size_t>(model.n_samples);
        if (n_train < 2) {
            ctx.set_error("Hotelling T2 fallback requires >= 2 stored scores");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        for (std::size_t comp = 0; comp < a; ++comp) {
            double mean = 0.0;
            for (std::size_t row = 0; row < n_train; ++row) {
                mean += model.scores_t[idx(row, a, comp)];
            }
            mean /= static_cast<double>(n_train);
            double sumsq = 0.0;
            for (std::size_t row = 0; row < n_train; ++row) {
                const double centered =
                    model.scores_t[idx(row, a, comp)] - mean;
                sumsq += centered * centered;
            }
            score_var[comp] = sumsq / static_cast<double>(n_train - 1);
        }
    } else {
        ctx.set_error(
            "Hotelling T2 needs either X_reference or model.store_scores=1");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    for (std::size_t comp = 0; comp < a; ++comp) {
        if (!(score_var[comp] > 0.0)) {
            ctx.set_errorf(
                "Hotelling T2 reference variance is zero at component %llu",
                static_cast<unsigned long long>(comp));
            return P4A_ERR_NUMERICAL_FAILURE;
        }
    }

    std::vector<double> scores;
    project_to_scores(model, x_buf, n, scores);
    out_t2.assign(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        double sum = 0.0;
        for (std::size_t comp = 0; comp < a; ++comp) {
            const double t = scores[idx(row, a, comp)];
            sum += (t * t) / score_var[comp];
        }
        out_t2[row] = sum;
    }
    return P4A_OK;
}

p4a_status_t pls_q_residuals(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    std::vector<double>& out_q) {
    out_q.clear();
    if (model.loadings_p.empty() || model.rotations_r.empty()) {
        ctx.set_error("model is not fitted (loadings/rotations missing)");
        return P4A_ERR_NOT_FITTED;
    }
    std::vector<double> x_buf;
    p4a_status_t status = copy_x(ctx, X, model, x_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);

    std::vector<double> scores;
    project_to_scores(model, x_buf, n, scores);

    out_q.assign(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        double sumsq = 0.0;
        for (std::size_t feature = 0; feature < p; ++feature) {
            double reconstructed = 0.0;
            for (std::size_t comp = 0; comp < a; ++comp) {
                reconstructed += scores[idx(row, a, comp)] *
                                 model.loadings_p[idx(feature, a, comp)];
            }
            const double centered =
                x_buf[idx(row, p, feature)] - model.x_mean[feature];
            const double residual = centered - reconstructed;
            sumsq += residual * residual;
        }
        out_q[row] = sumsq;
    }
    return P4A_OK;
}

p4a_status_t pls_dmodx(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t* X_reference,
    std::vector<double>& out_dmodx) {
    out_dmodx.clear();
    std::vector<double> q;
    p4a_status_t status = pls_q_residuals(ctx, model, X, q);
    if (status != P4A_OK) return status;
    const std::size_t n = q.size();
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    if (p <= a) {
        ctx.set_error("DModX requires n_features > n_components");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const double dof = static_cast<double>(p - a);

    double sigma = 1.0;
    if (X_reference != nullptr) {
        std::vector<double> q_ref;
        status = pls_q_residuals(ctx, model, *X_reference, q_ref);
        if (status != P4A_OK) return status;
        const std::size_t n_ref = q_ref.size();
        if (n_ref < a + 2U) {
            ctx.set_error("DModX reference requires at least n_components + 2 rows");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        double sum = 0.0;
        for (double q_value : q_ref) sum += q_value;
        const double dof_ref =
            static_cast<double>(n_ref) - static_cast<double>(a) - 1.0;
        if (!(dof_ref > 0.0)) {
            ctx.set_error("DModX reference DoF is non-positive");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        sigma = std::sqrt(sum / (dof * dof_ref));
        if (!(sigma > 0.0)) sigma = 1.0;
    }

    out_dmodx.assign(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        out_dmodx[row] = std::sqrt(q[row] / dof) / sigma;
    }
    return P4A_OK;
}

}  // namespace pls4all::core
