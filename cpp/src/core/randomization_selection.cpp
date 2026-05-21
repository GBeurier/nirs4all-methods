// SPDX-License-Identifier: CECILL-2.1

#include "core/randomization_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/status.hpp"

namespace {

constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
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

[[nodiscard]] n4m_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

[[nodiscard]] n4m_status_t validate_float_view(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& view,
                                               const char* name) noexcept {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(view);
    if (status != N4M_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::n4m::core::status_to_string(status));
        return status;
    }
    if (view.dtype != N4M_DTYPE_F64 && view.dtype != N4M_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_request(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& cfg,
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            std::int32_t n_permutations,
                                            double alpha) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("randomization-selection matrices must be non-empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_permutations <= 0) {
        ctx.set_errorf("n_permutations must be >= 1; got %d",
                       static_cast<int>(n_permutations));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_permutations == std::numeric_limits<std::int32_t>::max()) {
        ctx.set_error("n_permutations is too large for p-value smoothing");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
        ctx.set_errorf("alpha must be finite and in [0, 1]; got %.17g", alpha);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("randomization-selection matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_matrix(::n4m::core::Context& ctx,
                                       const n4m_matrix_view_t& view,
                                       const char* name,
                                       std::vector<double>& out) {
    const auto rows = static_cast<std::size_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    if (cols != 0U && rows > std::numeric_limits<std::size_t>::max() / cols) {
        ctx.set_errorf("%s matrix shape is too large", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(view, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               static_cast<unsigned long long>(row),
                               static_cast<unsigned long long>(col));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
}

[[nodiscard]] std::vector<std::int64_t> permutation(std::size_t n_rows,
                                                    std::uint64_t seed) {
    std::vector<std::int64_t> order(n_rows, 0);
    std::iota(order.begin(), order.end(), static_cast<std::int64_t>(0));
    std::uint64_t state = seed;
    for (std::size_t upper = n_rows; upper > 1U; --upper) {
        const std::uint64_t bits = splitmix64_next(state);
        const std::size_t swap_with = static_cast<std::size_t>(bits % upper);
        std::swap(order[upper - 1U], order[swap_with]);
    }
    return order;
}

[[nodiscard]] n4m_status_t coefficient_scores(::n4m::core::Context& ctx,
                                              const ::n4m::core::Config& cfg,
                                              const n4m_matrix_view_t& X,
                                              const n4m_matrix_view_t& Y,
                                              std::vector<double>& out) {
    std::unique_ptr<::n4m::core::Model> model;
    n4m_status_t status = ::n4m::core::fit_model(ctx, cfg, X, Y, model);
    if (status != N4M_OK) {
        return status;
    }
    if (!model || model->n_features <= 0 || model->n_targets <= 0) {
        ctx.set_error("randomization fit returned invalid model dimensions");
        return N4M_ERR_INTERNAL;
    }

    const auto n_features = static_cast<std::size_t>(model->n_features);
    const auto n_targets = static_cast<std::size_t>(model->n_targets);
    if (model->coefficients.size() != n_features * n_targets) {
        ctx.set_error("randomization fit returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }

    out.assign(n_features, 0.0);
    for (std::size_t feature = 0; feature < n_features; ++feature) {
        double score = 0.0;
        for (std::size_t target = 0; target < n_targets; ++target) {
            score = std::max(score, std::fabs(model->coefficients[idx(feature, n_targets, target)]));
        }
        out[feature] = score;
    }
    return N4M_OK;
}

void permute_y(const std::vector<double>& original,
               std::size_t rows,
               std::size_t cols,
               const std::vector<std::int64_t>& order,
               std::vector<double>& out) {
    out.assign(original.size(), 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto src = static_cast<std::size_t>(order[row]);
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = original[idx(src, cols, col)];
        }
    }
}

[[nodiscard]] std::vector<std::int64_t> rank_selected(const std::vector<double>& p_values,
                                                      const std::vector<double>& scores,
                                                      double alpha) {
    std::vector<std::int64_t> selected;
    selected.reserve(p_values.size());
    for (std::size_t feature = 0; feature < p_values.size(); ++feature) {
        if (p_values[feature] <= alpha) {
            selected.push_back(static_cast<std::int64_t>(feature));
        }
    }
    std::stable_sort(selected.begin(),
                     selected.end(),
                     [&p_values, &scores](std::int64_t lhs, std::int64_t rhs) {
                         const auto left = static_cast<std::size_t>(lhs);
                         const auto right = static_cast<std::size_t>(rhs);
                         if (p_values[left] == p_values[right]) {
                             if (scores[left] == scores[right]) {
                                 return lhs < rhs;
                             }
                             return scores[left] > scores[right];
                         }
                         return p_values[left] < p_values[right];
                     });
    return selected;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_randomization_test(Context& ctx,
                                          const Config& cfg,
                                          const n4m_matrix_view_t& X,
                                          const n4m_matrix_view_t& Y,
                                          std::int32_t n_permutations,
                                          std::uint64_t randomization_seed,
                                          double alpha,
                                          RandomizationSelectionResult& out) {
    try {
        out = RandomizationSelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, X, Y, n_permutations, alpha);
        if (status != N4M_OK) {
            return status;
        }

        status = coefficient_scores(ctx, cfg, X, Y, out.observed_scores);
        if (status != N4M_OK) {
            out = RandomizationSelectionResult{};
            return status;
        }

        std::vector<double> original_y;
        status = copy_matrix(ctx, Y, "Y", original_y);
        if (status != N4M_OK) {
            out = RandomizationSelectionResult{};
            return status;
        }

        const auto rows = static_cast<std::size_t>(Y.rows);
        const auto y_cols = static_cast<std::size_t>(Y.cols);
        out.exceedance_counts.assign(out.observed_scores.size(), 0);
        std::vector<double> permuted_y;
        std::vector<double> permuted_scores;
        for (std::int32_t perm = 0; perm < n_permutations; ++perm) {
            const std::uint64_t member_seed =
                randomization_seed + static_cast<std::uint64_t>(perm + 1) * kSplitMixGolden;
            const std::vector<std::int64_t> order = permutation(rows, member_seed);
            permute_y(original_y, rows, y_cols, order, permuted_y);
            n4m_matrix_view_t permuted_y_view = rowmajor_f64_view(permuted_y, Y.rows, Y.cols);
            status = coefficient_scores(ctx, cfg, X, permuted_y_view, permuted_scores);
            if (status != N4M_OK) {
                out = RandomizationSelectionResult{};
                return status;
            }
            if (permuted_scores.size() != out.observed_scores.size()) {
                ctx.set_error("randomization score count changed across permutations");
                out = RandomizationSelectionResult{};
                return N4M_ERR_INTERNAL;
            }
            for (std::size_t feature = 0; feature < out.observed_scores.size(); ++feature) {
                if (permuted_scores[feature] >= out.observed_scores[feature]) {
                    ++out.exceedance_counts[feature];
                }
            }
        }

        out.p_values.assign(out.observed_scores.size(), 0.0);
        const double denominator = static_cast<double>(n_permutations + 1);
        for (std::size_t feature = 0; feature < out.p_values.size(); ++feature) {
            out.p_values[feature] =
                static_cast<double>(out.exceedance_counts[feature] + 1) / denominator;
        }

        out.selected_indices = rank_selected(out.p_values, out.observed_scores, alpha);
        out.n_features = static_cast<std::int32_t>(out.observed_scores.size());
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_permutations = n_permutations;
        out.selected_count = static_cast<std::int32_t>(out.selected_indices.size());
        out.randomization_seed = randomization_seed;
        out.alpha = alpha;
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running randomization selection");
        out = RandomizationSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running randomization selection");
        out = RandomizationSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
