// SPDX-License-Identifier: CeCILL-2.1

#include "core/component_coefficients.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <vector>

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
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
        if (pivot_abs <= 0.0) {
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

[[nodiscard]] p4a_status_t validate_model(::pls4all::core::Context& ctx,
                                          const ::pls4all::core::Model& model) {
    if (model.n_features <= 0 || model.n_targets <= 0 || model.n_components <= 0) {
        ctx.set_error("fitted model dimensions are invalid for component coefficients");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(model.n_features);
    const auto q = static_cast<std::size_t>(model.n_targets);
    const auto a = static_cast<std::size_t>(model.n_components);
    if (model.weights_w.size() != p * a ||
        model.loadings_p.size() != p * a ||
        model.y_loadings_q.size() != q * a ||
        model.x_scale.size() != p ||
        model.y_scale.size() != q) {
        ctx.set_error("fitted model arrays are inconsistent for component coefficients");
        return P4A_ERR_INTERNAL;
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t compute_regression_coefficients_by_component(Context& ctx,
                                                          const Model& model,
                                                          std::vector<double>& out) {
    try {
        out.clear();
        const p4a_status_t status = validate_model(ctx, model);
        if (status != P4A_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(model.n_features);
        const auto q = static_cast<std::size_t>(model.n_targets);
        const auto a = static_cast<std::size_t>(model.n_components);
        out.assign(a * p * q, 0.0);

        for (std::size_t prefix = 1U; prefix <= a; ++prefix) {
            std::vector<double> ptw;
            resize_fill(ptw, prefix * prefix, 0.0);
            for (std::size_t row_comp = 0; row_comp < prefix; ++row_comp) {
                for (std::size_t col_comp = 0; col_comp < prefix; ++col_comp) {
                    double sum = 0.0;
                    for (std::size_t feature = 0; feature < p; ++feature) {
                        sum += model.loadings_p[idx(feature, a, row_comp)] *
                               model.weights_w[idx(feature, a, col_comp)];
                    }
                    ptw[idx(row_comp, prefix, col_comp)] = sum;
                }
            }

            std::vector<double> ptw_inv;
            if (!invert_square(ptw, prefix, ptw_inv)) {
                ctx.set_error("failed to invert P_k.T @ W_k while computing component coefficients");
                out.clear();
                return P4A_ERR_NUMERICAL_FAILURE;
            }

            const std::size_t block = (prefix - 1U) * p * q;
            for (std::size_t feature = 0; feature < p; ++feature) {
                for (std::size_t target = 0; target < q; ++target) {
                    double coefficient = 0.0;
                    for (std::size_t comp = 0; comp < prefix; ++comp) {
                        double rotation = 0.0;
                        for (std::size_t inner = 0; inner < prefix; ++inner) {
                            rotation += model.weights_w[idx(feature, a, inner)] *
                                        ptw_inv[idx(inner, prefix, comp)];
                        }
                        coefficient += rotation * model.y_loadings_q[idx(target, a, comp)];
                    }
                    out[block + idx(feature, q, target)] =
                        coefficient * model.y_scale[target] / model.x_scale[feature];
                }
            }
        }

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing component coefficients");
        out.clear();
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing component coefficients");
        out.clear();
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
