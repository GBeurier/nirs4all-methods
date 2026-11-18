// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Model-intrinsic structure and importances (n4m/estimator.h, ABI 2.24).
// The VIP and selectivity-ratio numerics are the shared kernels of
// core/variable_importance.cpp (also behind
// n4m_feature_selection_variable_select_rank); this file only forms their
// inputs from a fitted estimator and rows the host supplies.

#include "core/estimator/intrinsic_importance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/model.hpp"
#include "core/variable_importance.hpp"

namespace n4m::estimator {

namespace {

std::size_t at(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

const core::Model* model_of(const n4m_estimator_s& est) noexcept {
    const n4m_model_t* model = est.adapter->linear_model();
    return model == nullptr ? nullptr : static_cast<const core::Model*>(model);
}

n4m_status_t unsupported(n4m_context_t* ctx, const char* what) {
    set_error(ctx, what);
    return N4M_ERR_UNSUPPORTED;
}

bool has_latent_arrays(const core::Model& m) noexcept {
    if (m.algorithm == N4M_ALGO_IMPORTED_LINEAR_PREDICTOR || m.n_components <= 0 ||
        m.n_features <= 0 || m.n_targets <= 0) {
        return false;
    }
    const auto p = static_cast<std::size_t>(m.n_features);
    const auto q = static_cast<std::size_t>(m.n_targets);
    const auto a = static_cast<std::size_t>(m.n_components);
    return m.weights_w.size() == p * a && m.rotations_r.size() == p * a &&
           m.loadings_p.size() == p * a && m.y_loadings_q.size() == q * a &&
           m.x_mean.size() == p && m.x_scale.size() == p;
}

// Dense finite copy of a caller view with `cols` columns and >= min_rows rows.
n4m_status_t dense_rows(n4m_context_t* ctx, const n4m_matrix_view_t* X, std::int64_t cols,
                        std::int64_t min_rows, std::vector<double>& out) {
    n4m_status_t st = check_input_view(ctx, X, "X");
    if (st != N4M_OK) return st;
    if (X->cols != cols || X->rows < min_rows) {
        ctx->set_errorf("X must have %lld columns and at least %lld rows (got %lldx%lld)",
                        static_cast<long long>(cols), static_cast<long long>(min_rows),
                        static_cast<long long>(X->rows), static_cast<long long>(X->cols));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const auto* data = static_cast<const double*>(X->data);
    out.resize(static_cast<std::size_t>(X->rows * X->cols));
    for (std::int64_t i = 0; i < X->rows; ++i) {
        for (std::int64_t j = 0; j < X->cols; ++j) {
            const double v = data[i * X->row_stride + j * X->col_stride];
            if (!std::isfinite(v)) {
                set_error(ctx, "X must contain only finite values");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[static_cast<std::size_t>(i * X->cols + j)] = v;
        }
    }
    return N4M_OK;
}

}  // namespace

n4m_status_t affine_map(n4m_context_t* ctx, const n4m_estimator_s& est,
                        std::vector<double>& coefficients, std::vector<double>& intercept) {
    if (!est.fitted) {
        set_error(ctx, "estimator is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    const core::Model* m = model_of(est);
    if ((est.adapter->capabilities() & N4M_CAP_AFFINE) == 0 || m == nullptr) {
        return unsupported(ctx, "the fitted estimator has no affine predictor (N4M_CAP_AFFINE)");
    }
    if (m->has_snv_savgol_pipeline) {
        return unsupported(ctx, "the model applies an internal non-affine preprocessing");
    }
    const auto p = static_cast<std::size_t>(m->n_features);
    const auto q = static_cast<std::size_t>(m->n_targets);
    if (m->coefficients.size() != p * q || m->x_mean.size() != p || m->y_mean.size() != q) {
        set_error(ctx, "fitted affine arrays are inconsistent");
        return N4M_ERR_INTERNAL;
    }
    // predict(x) = y_mean + (x - x_mean) B (core predict_into).
    coefficients = m->coefficients;
    intercept = m->y_mean;
    for (std::size_t t = 0; t < q; ++t) {
        for (std::size_t j = 0; j < p; ++j) {
            intercept[t] -= m->x_mean[j] * m->coefficients[at(j, q, t)];
        }
    }
    return N4M_OK;
}

n4m_status_t latent_structure(n4m_context_t* ctx, const n4m_estimator_s& est,
                              LatentStructure& out) {
    if (!est.fitted) {
        set_error(ctx, "estimator is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    const core::Model* m = model_of(est);
    if (m == nullptr || !has_latent_arrays(*m) || m->has_snv_savgol_pipeline) {
        return unsupported(ctx,
                           "the fitted estimator keeps no latent structure (W, R, P, Q)");
    }
    out.n_features = m->n_features;
    out.n_targets = m->n_targets;
    out.n_components = m->n_components;
    out.x_weights = &m->weights_w;
    out.x_rotations = &m->rotations_r;
    out.x_loadings = &m->loadings_p;
    out.y_loadings = &m->y_loadings_q;
    out.x_mean = &m->x_mean;
    out.x_scale = &m->x_scale;
    return N4M_OK;
}

n4m_status_t importance(n4m_context_t* ctx, const n4m_estimator_s& est,
                        const n4m_matrix_view_t* X, std::int32_t kind,
                        std::int64_t output_index, std::vector<double>& out) {
    if (kind < N4M_IMPORTANCE_VIP || kind > N4M_IMPORTANCE_STANDARDIZED_COEFFICIENTS) {
        set_error(ctx, "unknown importance kind");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!est.fitted) {
        set_error(ctx, "estimator is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    const std::int64_t p64 = est.adapter->n_features_in();
    const std::int64_t q64 = est.adapter->n_outputs();
    if (output_index < 0 || output_index >= q64) {
        set_error(ctx, "output_index is outside [0, n_outputs)");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    auto& core_ctx = static_cast<core::Context&>(*ctx);

    if (kind == N4M_IMPORTANCE_VIP) {
        LatentStructure latent;
        n4m_status_t st = latent_structure(ctx, est, latent);
        if (st != N4M_OK) return st;
        std::vector<double> x;
        st = dense_rows(ctx, X, p64, 1, x);
        if (st != N4M_OK) return st;
        const core::Model* latent_model = model_of(est);
        if (latent_model == nullptr) return unsupported(ctx, "the fitted estimator keeps no model");
        const core::Model& m = *latent_model;
        const auto n = static_cast<std::size_t>(X->rows);
        const auto p = static_cast<std::size_t>(p64);
        const auto a = static_cast<std::size_t>(m.n_components);
        // Scores of the supplied rows, T = ((X - x_mean) / x_scale) R (the
        // latent transform of the model).
        std::vector<double> scores(n * a, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                const double z = (x[at(i, p, j)] - m.x_mean[j]) / m.x_scale[j];
                for (std::size_t c = 0; c < a; ++c) {
                    scores[at(i, a, c)] += z * m.rotations_r[at(j, a, c)];
                }
            }
        }
        return core::compute_vip_from_scores(core_ctx, m, scores, n, out);
    }

    std::vector<double> B, b;
    n4m_status_t st = affine_map(ctx, est, B, b);
    if (st != N4M_OK) return st;
    std::vector<double> x;
    st = dense_rows(ctx, X, p64,
                    kind == N4M_IMPORTANCE_STANDARDIZED_COEFFICIENTS ? 2 : 1, x);
    if (st != N4M_OK) return st;
    const auto n = static_cast<std::size_t>(X->rows);
    const auto p = static_cast<std::size_t>(p64);
    const auto q = static_cast<std::size_t>(q64);
    const auto o = static_cast<std::size_t>(output_index);
    std::vector<double> mean(p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) mean[j] += x[at(i, p, j)];
    }
    for (double& v : mean) v /= static_cast<double>(n);

    if (kind == N4M_IMPORTANCE_STANDARDIZED_COEFFICIENTS) {
        out.assign(p, 0.0);
        for (std::size_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = x[at(i, p, j)] - mean[j];
                ss += d * d;
            }
            out[j] = B[at(j, q, o)] * std::sqrt(ss / static_cast<double>(n - 1));
        }
        return N4M_OK;
    }

    // Target-projection selectivity ratio (Rajalahti et al. 2009) in the
    // model space Z = (X - mean) / x_scale, with the model-space regression
    // vector B_model = x_scale * B[:, o].
    const core::Model* fitted_model = model_of(est);
    if (fitted_model == nullptr) return unsupported(ctx, "the fitted estimator keeps no model");
    const std::vector<double>& scale = fitted_model->x_scale;
    std::vector<double> w(p, 0.0);
    double w_norm = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        w[j] = B[at(j, q, o)] * scale[j];
        w_norm = std::hypot(w_norm, w[j]);
    }
    if (!(w_norm > 0.0)) {
        set_error(ctx, "the regression vector of this output is zero: the selectivity ratio "
                       "is undefined");
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (double& v : w) v /= w_norm;
    std::vector<double> z(n * p, 0.0), t(n, 0.0), loading(p, 0.0);
    double tt = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            z[at(i, p, j)] = (x[at(i, p, j)] - mean[j]) / scale[j];
            t[i] += z[at(i, p, j)] * w[j];
        }
        tt += t[i] * t[i];
    }
    if (!(tt > 0.0)) {
        set_error(ctx, "the target-projected scores of the supplied rows are zero: the "
                       "selectivity ratio is undefined");
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += z[at(i, p, j)] * t[i];
        loading[j] = s / tt;
    }
    return core::selectivity_ratio_from_reconstruction(core_ctx, z, n, p, t, loading, 1, out);
}

n4m_status_t pipeline_input_affine(n4m_context_t* ctx, const n4m_role_pipeline_s& pipeline,
                                   std::vector<double>& coefficients,
                                   std::vector<double>& intercept) {
    if (!pipeline.fitted) {
        set_error(ctx, "role pipeline is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    const std::size_t terminal = pipeline.steps.size() - 1;
    for (std::size_t s = 0; s < terminal; ++s) {
        const auto& step = pipeline.steps[s];
        if (step.state_index < 0) continue;  // sample filters act on training rows only
        const auto& adapter = *pipeline.states[static_cast<std::size_t>(step.state_index)]->adapter;
        if ((adapter.transform_traits() & N4M_CAP_AFFINE_TRANSFORM) == 0) {
            set_error(ctx, "the transform is not affine (T(x) = A x + c)");
            return pipeline_error(ctx, N4M_ERR_UNSUPPORTED, step_label(pipeline, s).c_str());
        }
    }
    std::vector<double> B, b;
    n4m_status_t st = affine_map(ctx, pipeline.terminal(), B, b);
    if (st != N4M_OK) return pipeline_error(ctx, st, step_label(pipeline, terminal).c_str());

    const std::int64_t p = pipeline.n_features;
    const std::int64_t m = pipeline.terminal().adapter->n_features_in();
    const auto q = b.size();
    if (pipeline.states.size() == 1) {  // no transform: the terminal map itself
        coefficients = std::move(B);
        intercept = std::move(b);
        return N4M_OK;
    }
    // Probe the composed chain z = A x + c: row 0 is the origin (z = c), row
    // 1 + k the unit row e_{j0 + k} (z = A e_j + c). Batches bound memory.
    const std::int64_t batch =
        std::max<std::int64_t>(1, std::min<std::int64_t>(p, (std::int64_t{1} << 21) /
                                                                std::max(p, m)));
    coefficients.assign(static_cast<std::size_t>(p) * q, 0.0);
    intercept.assign(q, 0.0);
    std::vector<double> origin;
    std::vector<double> rows, store;
    for (std::int64_t j0 = -1; j0 < p;) {
        const std::int64_t count = j0 < 0 ? 1 : std::min(batch, p - j0);
        rows.assign(static_cast<std::size_t>(count * p), 0.0);
        if (j0 >= 0) {
            for (std::int64_t k = 0; k < count; ++k) {
                rows[static_cast<std::size_t>(k * p + j0 + k)] = 1.0;
            }
        }
        n4m_matrix_view_t X{};
        n4m_matrix_view_init_rowmajor(&X, rows.data(), count, p, N4M_DTYPE_F64);
        n4m_matrix_view_t z{};
        st = pipeline_features(ctx, pipeline, &X, store, z);
        if (st != N4M_OK) return st;
        const auto* zd = static_cast<const double*>(z.data);
        auto zat = [&](std::int64_t i, std::int64_t c) {
            return zd[i * z.row_stride + c * z.col_stride];
        };
        if (j0 < 0) {
            origin.resize(static_cast<std::size_t>(m));
            for (std::int64_t c = 0; c < m; ++c) origin[static_cast<std::size_t>(c)] = zat(0, c);
            for (std::size_t t = 0; t < q; ++t) {
                double sum = b[t];
                for (std::int64_t c = 0; c < m; ++c) {
                    sum += origin[static_cast<std::size_t>(c)] *
                           B[at(static_cast<std::size_t>(c), q, t)];
                }
                intercept[t] = sum;
            }
            j0 = 0;
            continue;
        }
        for (std::int64_t k = 0; k < count; ++k) {
            const auto j = static_cast<std::size_t>(j0 + k);
            for (std::size_t t = 0; t < q; ++t) {
                double sum = 0.0;
                for (std::int64_t c = 0; c < m; ++c) {
                    const double a = zat(k, c) - origin[static_cast<std::size_t>(c)];
                    if (a != 0.0) sum += a * B[at(static_cast<std::size_t>(c), q, t)];
                }
                coefficients[at(j, q, t)] = sum;
            }
        }
        j0 += count;
    }
    return N4M_OK;
}

}  // namespace n4m::estimator
