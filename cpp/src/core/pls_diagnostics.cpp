// SPDX-License-Identifier: CECILL-2.1

#include "core/pls_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/matrix_view.hpp"

namespace {

[[maybe_unused]] inline std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

// Scores (n_rows x n_components, row-major) of rows already in the model's
// X space (model_space_x): t = x @ rotations_R. A null component (zero
// rotation column) has a zero score column.
void project_to_scores(const ::n4m::core::Model& model,
                       const std::vector<double>& x_model,
                       std::size_t n_rows,
                       std::vector<double>& out_scores) {
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    out_scores.assign(n_rows * a, 0.0);
    for (std::size_t row = 0; row < n_rows; ++row) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += x_model[idx(row, p, feature)] * model.rotations_r[idx(feature, a, comp)];
            }
            out_scores[idx(row, a, comp)] = sum;
        }
    }
}

// The latent X directions the model keeps: the components whose rotation
// column is nonzero. A PLS component flagged unresolved is a null component
// (zero W, P, R columns) and has no X direction; a PCR component flagged
// unresolved only for its Y side (u = Y q / q'q undefined or below the
// resolution) keeps a resolved X direction and is kept.
std::vector<std::size_t> latent_directions(const ::n4m::core::Model& model) {
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    std::vector<std::size_t> kept;
    for (std::size_t comp = 0; comp < a; ++comp) {
        for (std::size_t feature = 0; feature < p; ++feature) {
            if (model.rotations_r[idx(feature, a, comp)] != 0.0) {
                kept.push_back(comp);
                break;
            }
        }
    }
    return kept;
}

[[nodiscard]] ::n4m_status_t copy_x(::n4m::core::Context& ctx,
                                    const n4m_matrix_view_t& X,
                                    const ::n4m::core::Model& model,
                                    std::vector<double>& out) {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(X);
    if (status != N4M_OK) {
        ctx.set_error("invalid X matrix view");
        return status;
    }
    if (X.dtype != N4M_DTYPE_F64) {
        ctx.set_error("X must be float64 for diagnostics");
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (X.cols != model.n_features) {
        ctx.set_errorf("X.cols (%lld) must match model.n_features (%d)",
                       static_cast<long long>(X.cols),
                       static_cast<int>(model.n_features));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = static_cast<std::size_t>(X.cols);
    out.assign(rows * cols, 0.0);
    const auto* data = static_cast<const double*>(X.data);
    if (data == nullptr) {
        if (rows > 0 && cols > 0) {
            ctx.set_error("X.data is null with non-zero shape");
            return N4M_ERR_NULL_POINTER;
        }
        return N4M_OK;
    }
    const std::size_t row_stride = static_cast<std::size_t>(X.row_stride);
    const std::size_t col_stride = static_cast<std::size_t>(X.col_stride);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double v = data[row * row_stride + col * col_stride];
            if (!std::isfinite(v)) {
                ctx.set_error("X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[row * cols + col] = v;
        }
    }
    return N4M_OK;
}

// Rows of X in the model's X space, as n4m_model_transform maps them: the
// fitted SNV/Savitzky-Golay pipeline when the model owns one, then centring
// and scaling by the model's x_mean / x_scale.
[[nodiscard]] ::n4m_status_t model_space_x(::n4m::core::Context& ctx,
                                           const n4m_matrix_view_t& X,
                                           const ::n4m::core::Model& model,
                                           std::vector<double>& out) {
    n4m_status_t status = copy_x(ctx, X, model, out);
    if (status != N4M_OK) return status;
    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    if (model.has_snv_savgol_pipeline && rows > 0 && p > 0) {
        const std::vector<double> raw = out;
        n4m_matrix_view_t raw_view{};
        raw_view.data = const_cast<double*>(raw.data());
        raw_view.rows = X.rows;
        raw_view.cols = X.cols;
        raw_view.row_stride = X.cols;
        raw_view.col_stride = 1;
        raw_view.dtype = N4M_DTYPE_F64;
        n4m_matrix_view_t out_view = raw_view;
        out_view.data = out.data();
        status = model.pipeline.transform(ctx, raw_view, out_view);
        if (status != N4M_OK) return status;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t feature = 0; feature < p; ++feature) {
            double& value = out[idx(row, p, feature)];
            value = (value - model.x_mean[feature]) / model.x_scale[feature];
        }
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t pls_hotelling_t2(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t* X_reference,
    std::vector<double>& out_t2) {
    out_t2.clear();
    if (model.rotations_r.empty()) {
        ctx.set_error("model is not fitted (rotations missing)");
        return N4M_ERR_NOT_FITTED;
    }
    std::vector<double> x_buf;
    n4m_status_t status = model_space_x(ctx, X, model, x_buf);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t a = static_cast<std::size_t>(model.n_components);
    const std::vector<std::size_t> kept = latent_directions(model);

    // Reference score variance of every kept direction, from X_reference or
    // from the stored training scores.
    std::vector<double> ref_scores;
    std::size_t n_ref = 0;
    const std::vector<double>* reference = nullptr;
    if (X_reference != nullptr) {
        std::vector<double> ref_buf;
        status = model_space_x(ctx, *X_reference, model, ref_buf);
        if (status != N4M_OK) return status;
        n_ref = static_cast<std::size_t>(X_reference->rows);
        if (n_ref < 2) {
            ctx.set_error("Hotelling T2 reference requires at least 2 rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        project_to_scores(model, ref_buf, n_ref, ref_scores);
        reference = &ref_scores;
    } else if (!model.scores_t.empty()) {
        n_ref = static_cast<std::size_t>(model.n_samples);
        if (n_ref < 2) {
            ctx.set_error("Hotelling T2 fallback requires >= 2 stored scores");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        reference = &model.scores_t;
    } else {
        ctx.set_error(
            "Hotelling T2 needs either X_reference or model.store_scores=1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> score_var(kept.size(), 0.0);
    for (std::size_t k = 0; k < kept.size(); ++k) {
        const std::size_t comp = kept[k];
        double mean = 0.0;
        for (std::size_t row = 0; row < n_ref; ++row) {
            mean += (*reference)[idx(row, a, comp)];
        }
        mean /= static_cast<double>(n_ref);
        double sumsq = 0.0;
        for (std::size_t row = 0; row < n_ref; ++row) {
            const double centered = (*reference)[idx(row, a, comp)] - mean;
            sumsq += centered * centered;
        }
        score_var[k] = sumsq / static_cast<double>(n_ref - 1);
        if (!(score_var[k] > 0.0)) {
            ctx.set_errorf(
                "Hotelling T2 reference variance is zero at component %llu",
                static_cast<unsigned long long>(comp));
            return N4M_ERR_NUMERICAL_FAILURE;
        }
    }

    std::vector<double> scores;
    project_to_scores(model, x_buf, n, scores);
    out_t2.assign(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        double sum = 0.0;
        for (std::size_t k = 0; k < kept.size(); ++k) {
            const double t = scores[idx(row, a, kept[k])];
            sum += (t * t) / score_var[k];
        }
        out_t2[row] = sum;
    }
    return N4M_OK;
}

std::size_t pls_latent_directions(const Model& model) {
    return model.rotations_r.empty() ? 0U : latent_directions(model).size();
}

n4m_status_t pls_q_residuals(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    std::vector<double>& out_q) {
    out_q.clear();
    if (model.loadings_p.empty() || model.rotations_r.empty()) {
        ctx.set_error("model is not fitted (loadings/rotations missing)");
        return N4M_ERR_NOT_FITTED;
    }
    std::vector<double> x_buf;
    n4m_status_t status = model_space_x(ctx, X, model, x_buf);
    if (status != N4M_OK) return status;
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
            const double residual = x_buf[idx(row, p, feature)] - reconstructed;
            sumsq += residual * residual;
        }
        out_q[row] = sumsq;
    }
    return N4M_OK;
}

n4m_status_t pls_dmodx(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t* X_reference,
    std::vector<double>& out_dmodx) {
    out_dmodx.clear();
    std::vector<double> q;
    n4m_status_t status = pls_q_residuals(ctx, model, X, q);
    if (status != N4M_OK) return status;
    const std::size_t n = q.size();
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t k = pls_latent_directions(model);
    if (p <= k) {
        ctx.set_error("DModX requires n_features > the number of latent directions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const double dof = static_cast<double>(p - k);

    double sigma = 1.0;
    if (X_reference != nullptr) {
        std::vector<double> q_ref;
        status = pls_q_residuals(ctx, model, *X_reference, q_ref);
        if (status != N4M_OK) return status;
        const std::size_t n_ref = q_ref.size();
        if (n_ref < k + 2U) {
            ctx.set_error("DModX reference requires at least latent directions + 2 rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        double sum = 0.0;
        for (double q_value : q_ref) sum += q_value;
        const double dof_ref = static_cast<double>(n_ref - k - 1U);
        sigma = std::sqrt(sum / (dof * dof_ref));
        if (!(sigma > 0.0)) sigma = 1.0;
    }

    out_dmodx.assign(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        out_dmodx[row] = std::sqrt(q[row] / dof) / sigma;
    }
    return N4M_OK;
}

}  // namespace n4m::core
