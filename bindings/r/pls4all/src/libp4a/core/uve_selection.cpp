// SPDX-License-Identifier: CECILL-2.1

#include "core/uve_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/stability_selection.hpp"
#include "core/status.hpp"

namespace {

constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;
constexpr double kDoubleUnit = 1.0 / 9007199254740992.0;  // 2^-53.

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] bool checked_matrix_size(std::int64_t rows,
                                       std::int64_t cols,
                                       std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const auto urows = static_cast<std::size_t>(rows);
    const auto ucols = static_cast<std::size_t>(cols);
    if (ucols != 0U &&
        urows > std::numeric_limits<std::size_t>::max() / ucols) {
        return false;
    }
    out = urows * ucols;
    return true;
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
                                            const p4a_matrix_view_t& X,
                                            const p4a_matrix_view_t& Y,
                                            const ::pls4all::core::ValidationPlan& plan,
                                            std::int32_t noise_features) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("UVE matrices must be non-empty");
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
    if (plan.folds.size() < 2U) {
        ctx.set_error("UVE requires at least two Monte-Carlo folds");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (noise_features <= 0) {
        ctx.set_errorf("noise_features must be >= 1; got %d",
                       static_cast<int>(noise_features));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("UVE matrix dimensions exceed int32 result storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > std::numeric_limits<std::int64_t>::max() -
                     static_cast<std::int64_t>(noise_features)) {
        ctx.set_error("UVE augmented feature count is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t total_features = X.cols + static_cast<std::int64_t>(noise_features);
    if (total_features > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("UVE augmented feature count exceeds int32 storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
}

[[nodiscard]] double uniform_signed(std::uint64_t& state) noexcept {
    const std::uint64_t bits = splitmix64_next(state);
    const double unit = static_cast<double>(bits >> 11U) * kDoubleUnit;
    return 2.0 * unit - 1.0;
}

void standardize_noise_columns(std::vector<double>& augmented,
                               std::size_t rows,
                               std::size_t total_cols,
                               std::size_t first_noise_col,
                               std::size_t noise_cols) noexcept {
    for (std::size_t local_col = 0; local_col < noise_cols; ++local_col) {
        const std::size_t col = first_noise_col + local_col;
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += augmented[idx(row, total_cols, col)];
        }
        const double mean = sum / static_cast<double>(rows);

        double ss = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double centered = augmented[idx(row, total_cols, col)] - mean;
            augmented[idx(row, total_cols, col)] = centered;
            ss += centered * centered;
        }

        double scale = 1.0;
        if (rows > 1U) {
            scale = std::sqrt(ss / static_cast<double>(rows - 1U));
            if (!std::isfinite(scale) || scale <= std::numeric_limits<double>::epsilon()) {
                scale = 1.0;
            }
        }
        for (std::size_t row = 0; row < rows; ++row) {
            augmented[idx(row, total_cols, col)] /= scale;
        }
    }
}

[[nodiscard]] p4a_status_t build_augmented_matrix(::pls4all::core::Context& ctx,
                                                  const p4a_matrix_view_t& X,
                                                  std::int32_t noise_features,
                                                  std::uint64_t noise_seed,
                                                  std::vector<double>& out) {
    const std::int64_t total_cols = X.cols + static_cast<std::int64_t>(noise_features);
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, total_cols, n_values)) {
        ctx.set_error("UVE augmented matrix shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto real_cols = static_cast<std::size_t>(X.cols);
    const auto total = static_cast<std::size_t>(total_cols);
    const auto noise_cols = static_cast<std::size_t>(noise_features);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < real_cols; ++col) {
            out[idx(row, total, col)] = read_value(X, row, col);
        }
    }

    std::uint64_t state = noise_seed;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < noise_cols; ++col) {
            out[idx(row, total, real_cols + col)] = uniform_signed(state);
        }
    }
    standardize_noise_columns(out, rows, total, real_cols, noise_cols);
    return P4A_OK;
}

[[nodiscard]] std::vector<std::int64_t> rank_descending(const std::vector<double>& scores) {
    std::vector<std::int64_t> order(scores.size(), 0);
    std::iota(order.begin(), order.end(), static_cast<std::int64_t>(0));
    std::stable_sort(order.begin(), order.end(), [&scores](std::int64_t lhs, std::int64_t rhs) {
        const double left = scores[static_cast<std::size_t>(lhs)];
        const double right = scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });
    return order;
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_uve(Context& ctx,
                           const Config& cfg,
                           const p4a_matrix_view_t& X,
                           const p4a_matrix_view_t& Y,
                           const ValidationPlan& plan,
                           std::int32_t noise_features,
                           std::uint64_t noise_seed,
                           UveSelectionResult& out) {
    try {
        out = UveSelectionResult{};
        p4a_status_t status = validate_request(ctx, X, Y, plan, noise_features);
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> augmented;
        status = build_augmented_matrix(ctx, X, noise_features, noise_seed, augmented);
        if (status != P4A_OK) {
            return status;
        }

        const std::int64_t total_features = X.cols + static_cast<std::int64_t>(noise_features);
        p4a_matrix_view_t augmented_view = rowmajor_f64_view(augmented, X.rows, total_features);
        StabilitySelectionResult stability;
        status = select_by_coefficient_stability(ctx,
                                                 cfg,
                                                 augmented_view,
                                                 Y,
                                                 plan,
                                                 static_cast<std::int32_t>(total_features),
                                                 stability);
        if (status != P4A_OK) {
            out = UveSelectionResult{};
            return status;
        }
        if (stability.stability_scores.size() != static_cast<std::size_t>(total_features)) {
            ctx.set_error("UVE stability score count does not match augmented feature count");
            out = UveSelectionResult{};
            return P4A_ERR_INTERNAL;
        }

        const auto real_cols = static_cast<std::size_t>(X.cols);
        const auto noise_cols = static_cast<std::size_t>(noise_features);
        out.real_stability_scores.assign(stability.stability_scores.begin(),
                                         stability.stability_scores.begin() +
                                             static_cast<std::ptrdiff_t>(real_cols));
        out.noise_stability_scores.assign(stability.stability_scores.begin() +
                                              static_cast<std::ptrdiff_t>(real_cols),
                                          stability.stability_scores.end());
        out.noise_threshold = *std::max_element(out.noise_stability_scores.begin(),
                                                out.noise_stability_scores.end());

        std::vector<std::int64_t> order = rank_descending(out.real_stability_scores);
        for (const std::int64_t feature : order) {
            if (out.real_stability_scores[static_cast<std::size_t>(feature)] >
                out.noise_threshold) {
                out.selected_indices.push_back(feature);
            }
        }

        out.n_features = static_cast<std::int32_t>(real_cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_repeats = static_cast<std::int32_t>(plan.folds.size());
        out.n_noise_features = static_cast<std::int32_t>(noise_cols);
        out.selected_count = static_cast<std::int32_t>(out.selected_indices.size());
        out.noise_seed = noise_seed;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running UVE selection");
        out = UveSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running UVE selection");
        out = UveSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
