// SPDX-License-Identifier: CECILL-2.1

#include "core/vissa_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
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
        std::int32_t n_iterations,
        std::int32_t n_submodels,
        double ratio_kept,
        double threshold,
        double floor_probability) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) return status;
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) return status;
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("VISSA-PLS matrices must be non-empty");
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
        ctx.set_error("VISSA-PLS requires at least one validation fold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_iterations < 1) {
        ctx.set_errorf("n_iterations must be >= 1; got %d", n_iterations);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_submodels < 1) {
        ctx.set_errorf("n_submodels must be >= 1; got %d", n_submodels);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(ratio_kept) || ratio_kept <= 0.0 || ratio_kept > 1.0) {
        ctx.set_error("ratio_kept must be in (0, 1]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        ctx.set_error("threshold must be in [0, 1]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(floor_probability) ||
        floor_probability < 0.0 || floor_probability >= 0.5) {
        ctx.set_error("floor_probability must be in [0, 0.5)");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 ||
        static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("VISSA-PLS feature count exceeds int32 result storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_columns(::pls4all::core::Context& ctx,
                                         const p4a_matrix_view_t& X,
                                         const std::vector<std::int64_t>& columns,
                                         std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("VISSA-PLS column selection must not be empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    out.assign(rows * cols, 0.0);
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("VISSA-PLS column index out of range");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("VISSA-PLS X contains NaN or Inf");
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

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_vissa(Context& ctx,
                              const Config& cfg,
                              const p4a_matrix_view_t& X,
                              const p4a_matrix_view_t& Y,
                              const ValidationPlan& plan,
                              std::int32_t n_iterations_in,
                              std::int32_t n_submodels_in,
                              double ratio_kept,
                              double threshold,
                              double floor_probability,
                              std::uint64_t seed,
                              VissaSelectionResult& out) {
    out = VissaSelectionResult{};
    p4a_status_t status = validate_request(ctx, cfg, X, Y, plan,
                                            n_iterations_in, n_submodels_in,
                                            ratio_kept, threshold,
                                            floor_probability);
    if (status != P4A_OK) return status;

    const auto p = static_cast<std::size_t>(X.cols);
    const auto n_iter = static_cast<std::size_t>(n_iterations_in);
    const auto n_sub = static_cast<std::size_t>(n_submodels_in);
    const auto k_keep_target = static_cast<std::size_t>(
        std::max<std::int64_t>(1,
            static_cast<std::int64_t>(std::round(
                ratio_kept * static_cast<double>(n_sub)))));
    const auto min_active =
        static_cast<std::size_t>(std::max<std::int32_t>(1, cfg.n_components));

    std::vector<double> w(p, 0.5);
    std::uint64_t state = seed;
    double best_rmse_global = std::numeric_limits<double>::infinity();
    std::vector<std::int64_t> best_indices_global;

    out.best_rmse_by_iteration.assign(n_iter, 0.0);
    out.mean_rmse_by_iteration.assign(n_iter, 0.0);
    out.top_k_per_iteration.assign(n_iter * p, 0.0);

    // Reusable per-iteration scratch.
    std::vector<std::uint8_t> mask(p, 0U);
    std::vector<std::vector<std::uint8_t>> masks;
    std::vector<double> rmses;
    std::vector<std::int64_t> idx_buf;
    masks.reserve(n_sub);
    rmses.reserve(n_sub);
    idx_buf.reserve(p);

    for (std::size_t t = 0; t < n_iter; ++t) {
        masks.clear();
        rmses.clear();

        // --- Sample n_sub WBMS submodels --------------------------------
        for (std::size_t s = 0; s < n_sub; ++s) {
            std::size_t active = 0;
            for (std::size_t d = 0; d < p; ++d) {
                const bool bit = uniform01(state) < w[d];
                mask[d] = bit ? 1U : 0U;
                if (bit) ++active;
            }
            if (active < min_active) {
                // Degenerate sample — skip this submodel.
                continue;
            }
            idx_buf.clear();
            for (std::size_t d = 0; d < p; ++d) {
                if (mask[d] != 0U) {
                    idx_buf.push_back(static_cast<std::int64_t>(d));
                }
            }
            double rmse = std::numeric_limits<double>::infinity();
            status = subset_rmse(ctx, cfg, X, Y, plan, idx_buf, rmse);
            if (status != P4A_OK) return status;
            masks.push_back(mask);
            rmses.push_back(rmse);
        }

        if (rmses.empty()) {
            ctx.set_errorf(
                "VISSA-PLS: all submodels in iteration %llu had fewer "
                "than n_components active features. Try raising "
                "floor_probability or n_submodels.",
                static_cast<unsigned long long>(t));
            return P4A_ERR_NUMERICAL_FAILURE;
        }

        // --- Sort ascending by RMSE; keep first k_actual --------------
        const std::size_t n_valid = rmses.size();
        if (n_valid == 0) {
            // No surviving masks in this VISSA iteration. Leave the
            // iteration metric slots untouched (initialised to NaN by
            // the result constructor) and skip to the next iteration.
            // Avoids order[0] / rmses[order[0]] on an empty vector,
            // which the compiler flags as a potential null-dereference
            // under -Werror=null-dereference.
            continue;
        }
        const std::size_t k_actual = std::min(k_keep_target, n_valid);
        std::vector<std::size_t> order(n_valid);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) {
                      return rmses[a] < rmses[b];
                  });

        // --- Iteration metrics ----------------------------------------
        const double iter_best = rmses[order[0]];
        double iter_sum = 0.0;
        for (double r : rmses) iter_sum += r;
        out.best_rmse_by_iteration[t] = iter_best;
        out.mean_rmse_by_iteration[t] =
            iter_sum / static_cast<double>(n_valid);

        if (iter_best < best_rmse_global) {
            best_rmse_global = iter_best;
            best_indices_global.clear();
            const auto& m_best = masks[order[0]];
            for (std::size_t d = 0; d < p; ++d) {
                if (m_best[d] != 0U) {
                    best_indices_global.push_back(static_cast<std::int64_t>(d));
                }
            }
        }

        // --- Update w_j ← mean(mask[j]) over kept ------------------
        std::vector<double> kept_counts(p, 0.0);
        for (std::size_t k = 0; k < k_actual; ++k) {
            const auto& m = masks[order[k]];
            for (std::size_t d = 0; d < p; ++d) {
                if (m[d] != 0U) kept_counts[d] += 1.0;
            }
        }
        const double inv_k = 1.0 / static_cast<double>(k_actual);
        const double floor_lo = floor_probability;
        const double floor_hi = 1.0 - floor_probability;
        for (std::size_t d = 0; d < p; ++d) {
            const double freq = kept_counts[d] * inv_k;
            double w_new = freq;
            if (w_new < floor_lo) w_new = floor_lo;
            if (w_new > floor_hi) w_new = floor_hi;
            w[d] = w_new;
            // Record the top-K mean mask as a diagnostic row.
            out.top_k_per_iteration[t * p + d] = freq;
        }
    }

    // --- Pack outputs ---------------------------------------------------
    out.final_probabilities = w;
    out.inclusion_frequencies = w;
    out.selected_indices.clear();
    for (std::size_t d = 0; d < p; ++d) {
        if (w[d] > threshold) {
            out.selected_indices.push_back(static_cast<std::int64_t>(d));
        }
    }
    out.best_rmse = best_rmse_global;
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(Y.cols);
    out.n_components = cfg.n_components;
    out.n_iterations = n_iterations_in;
    out.n_submodels = n_submodels_in;
    out.ratio_kept = ratio_kept;
    out.threshold = threshold;
    out.floor_probability = floor_probability;
    out.seed = seed;

    (void)best_indices_global;  // exposed via final_probabilities + threshold
    return P4A_OK;
}

}  // namespace pls4all::core
