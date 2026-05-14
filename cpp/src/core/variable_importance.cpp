// SPDX-License-Identifier: CeCILL-2.1

#include "core/variable_importance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
}

[[nodiscard]] p4a_status_t validate_scores_available(::pls4all::core::Context& ctx,
                                                     const ::pls4all::core::Model& model) {
    if (model.store_scores == 0 || model.scores_t.empty()) {
        ctx.set_error("variable-importance metrics require store_scores=1 at fit time");
        return P4A_ERR_UNSUPPORTED;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_model_arrays(::pls4all::core::Context& ctx,
                                                 const ::pls4all::core::Model& model) {
    if (model.n_samples <= 0 || model.n_features <= 0 ||
        model.n_targets <= 0 || model.n_components <= 0) {
        ctx.set_error("fitted model dimensions are invalid for variable importance");
        return P4A_ERR_INVALID_ARGUMENT;
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
        return P4A_ERR_INTERNAL;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_x(::pls4all::core::Context& ctx,
                                      const ::pls4all::core::Model& model,
                                      const p4a_matrix_view_t& X) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(X);
    if (status != P4A_OK) {
        ctx.set_errorf("X matrix view is invalid: %s",
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (X.dtype != P4A_DTYPE_F64 && X.dtype != P4A_DTYPE_F32) {
        ctx.set_error("X dtype must be f64 or f32");
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (X.rows != model.n_samples || X.cols != model.n_features) {
        ctx.set_errorf("X shape (%lld, %lld) must match fitted training shape (%lld, %d)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(X.cols),
                       static_cast<long long>(model.n_samples),
                       static_cast<int>(model.n_features));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t compute_vip_scores(Context& ctx,
                                const Model& model,
                                std::vector<double>& out) {
    try {
        out.clear();
        p4a_status_t status = validate_scores_available(ctx, model);
        if (status != P4A_OK) {
            return status;
        }
        status = validate_model_arrays(ctx, model);
        if (status != P4A_OK) {
            return status;
        }

        const auto n = static_cast<std::size_t>(model.n_samples);
        const auto p = static_cast<std::size_t>(model.n_features);
        const auto q = static_cast<std::size_t>(model.n_targets);
        const auto a = static_cast<std::size_t>(model.n_components);

        std::vector<double> component_ssy(a, 0.0);
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
            component_ssy[comp] = tss * qss;
            total_ssy += component_ssy[comp];
        }
        if (total_ssy <= kEps) {
            ctx.set_error("Y variance explained by components vanished while computing VIP");
            return P4A_ERR_NUMERICAL_FAILURE;
        }

        out.assign(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double weighted = 0.0;
            for (std::size_t comp = 0; comp < a; ++comp) {
                const double w = model.weights_w[idx(feature, a, comp)];
                weighted += component_ssy[comp] * w * w;
            }
            out[feature] = std::sqrt(static_cast<double>(p) * weighted / total_ssy);
        }

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing VIP scores");
        out.clear();
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing VIP scores");
        out.clear();
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t compute_selectivity_ratio(Context& ctx,
                                       const Model& model,
                                       const p4a_matrix_view_t& X,
                                       std::vector<double>& out) {
    try {
        out.clear();
        p4a_status_t status = validate_scores_available(ctx, model);
        if (status != P4A_OK) {
            return status;
        }
        status = validate_model_arrays(ctx, model);
        if (status != P4A_OK) {
            return status;
        }
        status = validate_x(ctx, model, X);
        if (status != P4A_OK) {
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
                    return P4A_ERR_INVALID_ARGUMENT;
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing selectivity ratio");
        out.clear();
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing selectivity ratio");
        out.clear();
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
