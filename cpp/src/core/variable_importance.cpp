// SPDX-License-Identifier: CECILL-2.1

#include "core/variable_importance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/common/status.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] double read_value(const n4m_matrix_view_t& view,
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

[[nodiscard]] n4m_status_t validate_scores_available(::n4m::core::Context& ctx,
                                                     const ::n4m::core::Model& model) {
    if (model.store_scores == 0 || model.scores_t.empty()) {
        ctx.set_error("variable-importance metrics require store_scores=1 at fit time");
        return N4M_ERR_UNSUPPORTED;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_model_arrays(::n4m::core::Context& ctx,
                                                 const ::n4m::core::Model& model) {
    if (model.n_samples <= 0 || model.n_features <= 0 ||
        model.n_targets <= 0 || model.n_components <= 0) {
        ctx.set_error("fitted model dimensions are invalid for variable importance");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto n = static_cast<std::size_t>(model.n_samples);
    const auto p = static_cast<std::size_t>(model.n_features);
    const auto q = static_cast<std::size_t>(model.n_targets);
    const auto a = static_cast<std::size_t>(model.n_components);
    if (model.weights_w.size() != p * a ||
        model.loadings_p.size() != p * a ||
        model.y_loadings_q.size() != q * a ||
        model.scores_t.size() != n * a ||
        model.x_mean.size() != p ||
        model.x_scale.size() != p) {
        ctx.set_error("fitted model arrays are inconsistent for variable importance");
        return N4M_ERR_INTERNAL;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_x(::n4m::core::Context& ctx,
                                      const ::n4m::core::Model& model,
                                      const n4m_matrix_view_t& X) noexcept {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(X);
    if (status != N4M_OK) {
        ctx.set_errorf("X matrix view is invalid: %s",
                       ::n4m::core::status_to_string(status));
        return status;
    }
    if (X.dtype != N4M_DTYPE_F64 && X.dtype != N4M_DTYPE_F32) {
        ctx.set_error("X dtype must be f64 or f32");
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (X.rows != model.n_samples || X.cols != model.n_features) {
        ctx.set_errorf("X shape (%lld, %lld) must match fitted training shape (%lld, %d)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(X.cols),
                       static_cast<long long>(model.n_samples),
                       static_cast<int>(model.n_features));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t compute_vip_scores(Context& ctx,
                                const Model& model,
                                std::vector<double>& out) {
    try {
        out.clear();
        n4m_status_t status = validate_scores_available(ctx, model);
        if (status != N4M_OK) {
            return status;
        }
        status = validate_model_arrays(ctx, model);
        if (status != N4M_OK) {
            return status;
        }

        const auto n = static_cast<std::size_t>(model.n_samples);
        const auto p = static_cast<std::size_t>(model.n_features);
        const auto q = static_cast<std::size_t>(model.n_targets);
        const auto a = static_cast<std::size_t>(model.n_components);

        std::vector<double> component_ssy(a, 0.0);
        std::vector<double> weights_col_ss(a, 0.0);
        double total_ssy = 0.0;
        for (std::size_t comp = 0; comp < a; ++comp) {
            double tss = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                const double value = model.scores_t[idx(row, a, comp)];
                tss += value * value;
            }
            double qss = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                const double value = model.y_loadings_q[idx(target, a, comp)];
                qss += value * value;
            }
            double wss = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                const double w = model.weights_w[idx(feature, a, comp)];
                wss += w * w;
            }
            component_ssy[comp] = tss * qss;
            // R `plsVarSel::VIP` normalises each weight column to unit
            // sum-of-squares (WW = W*W / colSums(W*W)) before mixing with
            // the per-component Y-variance proxy. NIPALS naturally yields
            // unit-norm weight columns so the normalisation is a no-op,
            // but SIMPLS stores R = W / ||X w|| (||R_a|| != 1), so without
            // this rescale the VIP ranking diverges from R when the SIMPLS
            // solver is used.
            weights_col_ss[comp] = wss;
            total_ssy += component_ssy[comp];
        }
        if (total_ssy <= kEps) {
            ctx.set_error("Y variance explained by components vanished while computing VIP");
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        out.assign(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double weighted = 0.0;
            for (std::size_t comp = 0; comp < a; ++comp) {
                const double wss = weights_col_ss[comp];
                if (wss <= kEps) {
                    continue;
                }
                const double w = model.weights_w[idx(feature, a, comp)];
                weighted += component_ssy[comp] * (w * w) / wss;
            }
            out[feature] = std::sqrt(static_cast<double>(p) * weighted / total_ssy);
        }

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing VIP scores");
        out.clear();
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing VIP scores");
        out.clear();
        return N4M_ERR_INTERNAL;
    }
}

n4m_status_t compute_selectivity_ratio(Context& ctx,
                                       const Model& model,
                                       const n4m_matrix_view_t& X,
                                       std::vector<double>& out) {
    try {
        out.clear();
        n4m_status_t status = validate_scores_available(ctx, model);
        if (status != N4M_OK) {
            return status;
        }
        status = validate_model_arrays(ctx, model);
        if (status != N4M_OK) {
            return status;
        }
        status = validate_x(ctx, model, X);
        if (status != N4M_OK) {
            return status;
        }

        const auto n = static_cast<std::size_t>(model.n_samples);
        const auto p = static_cast<std::size_t>(model.n_features);
        const auto a = static_cast<std::size_t>(model.n_components);

        std::vector<double> explained(p, 0.0);
        std::vector<double> residual(p, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                double x_scaled = read_value(X, row, feature);
                if (!std::isfinite(x_scaled)) {
                    ctx.set_errorf("X contains NaN or Inf at row %llu col %llu",
                                   ull(row),
                                   ull(feature));
                    out.clear();
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                x_scaled -= model.x_mean[feature];
                x_scaled /= model.x_scale[feature];

                double reconstructed = 0.0;
                for (std::size_t comp = 0; comp < a; ++comp) {
                    reconstructed +=
                        model.scores_t[idx(row, a, comp)] *
                        model.loadings_p[idx(feature, a, comp)];
                }
                explained[feature] += reconstructed * reconstructed;
                const double error = x_scaled - reconstructed;
                residual[feature] += error * error;
            }
        }

        out.assign(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            out[feature] = explained[feature] / std::max(residual[feature], kEps);
        }

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing selectivity ratio");
        out.clear();
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing selectivity ratio");
        out.clear();
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
