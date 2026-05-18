// SPDX-License-Identifier: CECILL-2.1

#include "core/pso_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/cross_validation.hpp"
#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/status.hpp"

namespace {

constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;
constexpr double kDoubleUnit = 1.0 / 9007199254740992.0;  // 2^-53

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols,
                              std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
}

[[nodiscard]] double uniform01(std::uint64_t& state) noexcept {
    return static_cast<double>(splitmix64_next(state) >> 11U) * kDoubleUnit;
}

[[nodiscard]] std::size_t random_bounded(std::uint64_t& state,
                                          std::size_t bound) noexcept {
    return static_cast<std::size_t>(splitmix64_next(state) %
                                     static_cast<std::uint64_t>(bound));
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

[[nodiscard]] p4a_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                   std::int64_t rows,
                                                   std::int64_t cols) noexcept {
    p4a_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                                const p4a_matrix_view_t& view,
                                                const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(view);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (view.dtype != P4A_DTYPE_F64 && view.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_request(::pls4all::core::Context& ctx,
        const ::pls4all::core::Config& cfg,
        const p4a_matrix_view_t& X,
        const p4a_matrix_view_t& Y,
        const ::pls4all::core::ValidationPlan& plan,
        std::int32_t n_swarm,
        std::int32_t n_iterations,
        double w, double c1, double c2, double v_max) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) return status;
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) return status;
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("PSO-PLS matrices must be non-empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (plan.n_samples != X.rows) {
        ctx.set_errorf("validation plan n_samples (%lld) must match X rows (%lld)",
                       static_cast<long long>(plan.n_samples),
                       static_cast<long long>(X.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (plan.folds.empty()) {
        ctx.set_error("PSO-PLS requires at least one validation fold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_swarm < 2) {
        ctx.set_errorf("n_swarm must be >= 2; got %d", n_swarm);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_iterations < 1) {
        ctx.set_errorf("n_iterations must be >= 1; got %d", n_iterations);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 ||
        static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(w) || w < 0.0 || w > 1.0) {
        ctx.set_error("inertia w must be in [0, 1]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(c1) || c1 < 0.0 || c1 > 4.0) {
        ctx.set_error("cognitive c1 must be in [0, 4]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(c2) || c2 < 0.0 || c2 > 4.0) {
        ctx.set_error("social c2 must be in [0, 4]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(v_max) || v_max <= 0.0) {
        ctx.set_error("v_max must be strictly positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("PSO-PLS feature count exceeds int32 result storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_columns(::pls4all::core::Context& ctx,
                                         const p4a_matrix_view_t& X,
                                         const std::vector<std::int64_t>& columns,
                                         std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("PSO-PLS column selection must not be empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    out.assign(rows * cols, 0.0);
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("PSO-PLS column index out of range");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("PSO-PLS X contains NaN or Inf");
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t subset_rmse(::pls4all::core::Context& ctx,
                                        const ::pls4all::core::Config& cfg,
                                        const p4a_matrix_view_t& X,
                                        const p4a_matrix_view_t& Y,
                                        const ::pls4all::core::ValidationPlan& plan,
                                        const std::vector<std::int64_t>& columns,
                                        double& out) {
    std::vector<double> subset_x;
    p4a_status_t status = copy_columns(ctx, X, columns, subset_x);
    if (status != P4A_OK) return status;
    p4a_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows,
                           static_cast<std::int64_t>(columns.size()));
    ::pls4all::core::CrossValidationResult cv;
    status = ::pls4all::core::cross_validate_regression(
        ctx, cfg, subset_view, Y, plan, cv);
    if (status != P4A_OK) return status;
    out = cv.metrics.rmse;
    return P4A_OK;
}

// Mask -> sorted index list (used to feed subset_rmse and the gbest export).
void mask_to_indices(const std::vector<std::uint8_t>& mask,
                     std::vector<std::int64_t>& out) {
    out.clear();
    out.reserve(mask.size());
    for (std::size_t d = 0; d < mask.size(); ++d) {
        if (mask[d] != 0U) {
            out.push_back(static_cast<std::int64_t>(d));
        }
    }
}

// Repair: ensure popcount(mask) >= n_components by flipping random zeros
// to ones. Consumes RNG state in a deterministic order.
void repair_mask(std::vector<std::uint8_t>& mask,
                 std::size_t min_active,
                 std::uint64_t& state) {
    std::size_t active = 0;
    for (std::uint8_t b : mask) {
        if (b != 0U) ++active;
    }
    while (active < min_active) {
        // Count zeros first so the bounded random pick is uniform.
        std::size_t zeros = 0;
        for (std::uint8_t b : mask) {
            if (b == 0U) ++zeros;
        }
        if (zeros == 0U) break;
        const std::size_t k = random_bounded(state, zeros);
        std::size_t seen = 0;
        for (std::size_t d = 0; d < mask.size(); ++d) {
            if (mask[d] == 0U) {
                if (seen == k) {
                    mask[d] = 1U;
                    ++active;
                    break;
                }
                ++seen;
            }
        }
    }
}

double sigmoid(double v) noexcept {
    // 1 / (1 + exp(-v)); v_max clipping upstream keeps v bounded.
    return 1.0 / (1.0 + std::exp(-v));
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_pso(Context& ctx,
                            const Config& cfg,
                            const p4a_matrix_view_t& X,
                            const p4a_matrix_view_t& Y,
                            const ValidationPlan& plan,
                            std::int32_t n_swarm_in,
                            std::int32_t n_iterations_in,
                            double w,
                            double c1,
                            double c2,
                            double v_max,
                            std::uint64_t seed,
                            PsoSelectionResult& out) {
    out = PsoSelectionResult{};
    p4a_status_t status = validate_request(ctx, cfg, X, Y, plan,
                                            n_swarm_in, n_iterations_in,
                                            w, c1, c2, v_max);
    if (status != P4A_OK) return status;

    const auto p = static_cast<std::size_t>(X.cols);
    const auto n_swarm = static_cast<std::size_t>(n_swarm_in);
    const auto n_iter = static_cast<std::size_t>(n_iterations_in);
    const auto min_active =
        static_cast<std::size_t>(std::max<std::int32_t>(1, cfg.n_components));

    std::uint64_t state = seed;

    // Particle storage: positions (n_swarm × p), velocities (n_swarm × p),
    // per-particle current RMSE, pbest mask + RMSE.
    std::vector<std::uint8_t> X_pos(n_swarm * p, 0U);
    std::vector<double>       V(n_swarm * p, 0.0);
    std::vector<std::uint8_t> pbest_x(n_swarm * p, 0U);
    std::vector<double>       pbest_rmse(n_swarm,
                                          std::numeric_limits<double>::infinity());
    std::vector<std::uint8_t> gbest_x(p, 0U);
    double gbest_rmse = std::numeric_limits<double>::infinity();

    // Inclusion-count accumulator (counts particle-iter cells with bit on).
    std::vector<std::uint64_t> inc_count(p, 0U);

    // --- Initialise positions (Bernoulli(0.5)) + repair --------------------
    std::vector<std::uint8_t> mask_buf(p, 0U);
    std::vector<std::int64_t> idx_buf;
    idx_buf.reserve(p);
    for (std::size_t i = 0; i < n_swarm; ++i) {
        for (std::size_t d = 0; d < p; ++d) {
            mask_buf[d] = (uniform01(state) < 0.5) ? 1U : 0U;
        }
        repair_mask(mask_buf, min_active, state);
        for (std::size_t d = 0; d < p; ++d) {
            X_pos[i * p + d] = mask_buf[d];
        }
        mask_to_indices(mask_buf, idx_buf);
        double rmse = std::numeric_limits<double>::infinity();
        status = subset_rmse(ctx, cfg, X, Y, plan, idx_buf, rmse);
        if (status != P4A_OK) return status;
        pbest_rmse[i] = rmse;
        for (std::size_t d = 0; d < p; ++d) {
            pbest_x[i * p + d] = mask_buf[d];
        }
        if (rmse < gbest_rmse) {
            gbest_rmse = rmse;
            gbest_x = mask_buf;
        }
    }

    out.best_rmse_by_iteration.assign(n_iter, 0.0);
    out.mean_rmse_by_iteration.assign(n_iter, 0.0);

    // --- Iterate -----------------------------------------------------------
    for (std::size_t t = 0; t < n_iter; ++t) {
        double sum_iter_rmse = 0.0;
        for (std::size_t i = 0; i < n_swarm; ++i) {
            for (std::size_t d = 0; d < p; ++d) {
                const double r1 = uniform01(state);
                const double r2 = uniform01(state);
                const double pbest_d =
                    static_cast<double>(pbest_x[i * p + d]);
                const double gbest_d = static_cast<double>(gbest_x[d]);
                const double xid = static_cast<double>(X_pos[i * p + d]);
                double vid = w * V[i * p + d]
                             + c1 * r1 * (pbest_d - xid)
                             + c2 * r2 * (gbest_d - xid);
                if (vid > v_max) vid = v_max;
                if (vid < -v_max) vid = -v_max;
                V[i * p + d] = vid;
                const double s = sigmoid(vid);
                const std::uint8_t bit = (uniform01(state) < s) ? 1U : 0U;
                X_pos[i * p + d] = bit;
                mask_buf[d] = bit;
            }
            repair_mask(mask_buf, min_active, state);
            // Sync the repaired bits back into X_pos so the velocity term
            // at iteration t+1 reads the actually-evaluated position, and
            // count inclusion on the post-repair mask (the one that
            // contributes to fitness).
            for (std::size_t d = 0; d < p; ++d) {
                X_pos[i * p + d] = mask_buf[d];
                if (mask_buf[d] != 0U) {
                    ++inc_count[d];
                }
            }
            mask_to_indices(mask_buf, idx_buf);
            double rmse = std::numeric_limits<double>::infinity();
            status = subset_rmse(ctx, cfg, X, Y, plan, idx_buf, rmse);
            if (status != P4A_OK) return status;
            sum_iter_rmse += rmse;
            if (rmse < pbest_rmse[i]) {
                pbest_rmse[i] = rmse;
                for (std::size_t d = 0; d < p; ++d) {
                    pbest_x[i * p + d] = mask_buf[d];
                }
            }
            if (rmse < gbest_rmse) {
                gbest_rmse = rmse;
                gbest_x = mask_buf;
            }
        }
        out.best_rmse_by_iteration[t] = gbest_rmse;
        out.mean_rmse_by_iteration[t] =
            sum_iter_rmse / static_cast<double>(n_swarm);
    }

    // --- Pack outputs ------------------------------------------------------
    const double denom = static_cast<double>(n_swarm * n_iter);
    out.inclusion_frequencies.assign(p, 0.0);
    for (std::size_t d = 0; d < p; ++d) {
        out.inclusion_frequencies[d] =
            denom > 0.0 ? static_cast<double>(inc_count[d]) / denom : 0.0;
    }
    mask_to_indices(gbest_x, out.selected_indices);

    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(Y.cols);
    out.n_components = cfg.n_components;
    out.n_swarm = n_swarm_in;
    out.n_iterations = n_iterations_in;
    out.seed = seed;
    out.w = w;
    out.c1 = c1;
    out.c2 = c2;
    out.v_max = v_max;
    out.best_rmse = gbest_rmse;
    return P4A_OK;
}

}  // namespace pls4all::core
