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

// Euclidean norm of a strided sequence, accumulated on values divided by
// their largest magnitude, so that it neither underflows nor overflows for
// any finite input whose norm is representable. 0 exactly for an all-zero
// sequence.
template <typename Value>
[[nodiscard]] double scaled_norm(std::size_t count, Value value) {
    double largest = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        largest = std::max(largest, std::fabs(value(i)));
    }
    if (!(largest > 0.0)) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double scaled = value(i) / largest;
        sum += scaled * scaled;
    }
    return largest * std::sqrt(sum);
}

// A component flagged in N4M_MODEL_UNRESOLVED_COMPONENTS: a PLS null
// component, or a PCR component whose Y side is not resolved.
[[nodiscard]] bool flagged_unresolved(const ::n4m::core::Model& model, std::size_t comp) {
    return comp < model.unresolved_components.size() &&
           model.unresolved_components[comp] != 0.0;
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

        // VIP_j = sqrt(p sum_a SSY_a (w_ja / ||w_a||)^2 / sum_a SSY_a), with
        // SSY_a = ||t_a||^2 ||q_a||^2 (R plsVarSel::VIP, which normalises each
        // weight column to unit sum of squares: NIPALS weights are unit
        // vectors already, SIMPLS stores R = W / ||X w||). Only the ratios
        // SSY_a / sum SSY and w_ja / ||w_a|| matter, so they are formed from
        // scaled norms: s_a = ||t_a|| ||q_a|| (the norm of the Y part the
        // component explains, representable inside the supported domain) and
        // SSY_a / sum SSY = (s_a / s_max)^2 / sum (s_b / s_max)^2. No
        // magnitude threshold is applied; a null component (flagged
        // unresolved, or with an exactly zero weight, score or Y-loading
        // column) does not contribute.
        std::vector<double> explained(a, 0.0);
        std::vector<double> weight_norm(a, 0.0);
        double largest = 0.0;
        for (std::size_t comp = 0; comp < a; ++comp) {
            weight_norm[comp] = scaled_norm(
                p, [&](std::size_t f) { return model.weights_w[idx(f, a, comp)]; });
            if (flagged_unresolved(model, comp) || weight_norm[comp] == 0.0) {
                continue;
            }
            const double t_norm = scaled_norm(
                n, [&](std::size_t row) { return model.scores_t[idx(row, a, comp)]; });
            const double q_norm = scaled_norm(
                q, [&](std::size_t target) { return model.y_loadings_q[idx(target, a, comp)]; });
            explained[comp] = t_norm * q_norm;
            largest = std::max(largest, explained[comp]);
        }
        if (!(largest > 0.0)) {
            ctx.set_error("no resolved component explains Y: VIP is undefined");
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        double total = 0.0;
        for (std::size_t comp = 0; comp < a; ++comp) {
            explained[comp] /= largest;
            explained[comp] *= explained[comp];
            total += explained[comp];
        }

        out.assign(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double weighted = 0.0;
            for (std::size_t comp = 0; comp < a; ++comp) {
                if (explained[comp] == 0.0) {
                    continue;
                }
                const double w = model.weights_w[idx(feature, a, comp)] / weight_norm[comp];
                weighted += explained[comp] * (w * w);
            }
            out[feature] = std::sqrt(static_cast<double>(p) * weighted / total);
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

        // SR_j = ||T P_j'||^2 / ||x_j - T P_j'||^2 on the model-space column
        // x_j = (X_j - x_mean_j) / x_scale_j. The ratio does not depend on the
        // units of x_j: both sums are accumulated on the column divided by the
        // exact power of two of its largest magnitude, then divided, with no
        // floor on the residual. A column reconstructed exactly (residual 0)
        // has SR = +inf when it carries explained variation and SR = 0 when
        // it is identically zero in model space (nothing to explain).
        std::vector<double> x_model(n * p, 0.0);
        std::vector<double> column_max(p, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                const double value = read_value(X, row, feature);
                if (!std::isfinite(value)) {
                    ctx.set_errorf("X contains NaN or Inf at row %llu col %llu",
                                   ull(row),
                                   ull(feature));
                    out.clear();
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                const double x_scaled = (value - model.x_mean[feature]) / model.x_scale[feature];
                x_model[idx(row, p, feature)] = x_scaled;
                column_max[feature] = std::max(column_max[feature], std::fabs(x_scaled));
            }
        }
        std::vector<int> column_exponent(p, 0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            if (column_max[feature] > 0.0) {
                column_exponent[feature] = std::ilogb(column_max[feature]);
            }
        }

        std::vector<double> explained(p, 0.0);
        std::vector<double> residual(p, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                double reconstructed = 0.0;
                for (std::size_t comp = 0; comp < a; ++comp) {
                    reconstructed +=
                        model.scores_t[idx(row, a, comp)] *
                        model.loadings_p[idx(feature, a, comp)];
                }
                const int e = column_exponent[feature];
                const double fitted = std::ldexp(reconstructed, -e);
                const double error = std::ldexp(x_model[idx(row, p, feature)] - reconstructed, -e);
                explained[feature] += fitted * fitted;
                residual[feature] += error * error;
            }
        }

        out.assign(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            if (residual[feature] > 0.0) {
                out[feature] = explained[feature] / residual[feature];
            } else {
                out[feature] = explained[feature] > 0.0
                                   ? std::numeric_limits<double>::infinity()
                                   : 0.0;
            }
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
