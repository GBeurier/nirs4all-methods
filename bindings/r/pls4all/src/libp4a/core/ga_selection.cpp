// SPDX-License-Identifier: CECILL-2.1

#include "core/ga_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#include "core/cross_validation.hpp"
#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/status.hpp"

namespace {

constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;
constexpr double kDoubleUnit = 1.0 / 9007199254740992.0;  // 2^-53.

struct EvaluatedSubset {
    double rmse{0.0};
    std::int64_t size{0};
    std::vector<std::int64_t> indices;
};

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
                                            const ::n4m::core::ValidationPlan& plan,
                                            std::int32_t n_generations,
                                            std::int32_t population_size,
                                            std::int32_t min_features,
                                            std::int32_t max_features,
                                            double mutation_rate) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("GA-PLS matrices must be non-empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (plan.n_samples != X.rows) {
        ctx.set_errorf("validation plan n_samples (%lld) must match X rows (%lld)",
                       static_cast<long long>(plan.n_samples),
                       static_cast<long long>(X.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (plan.folds.empty()) {
        ctx.set_error("GA-PLS requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_generations <= 0) {
        ctx.set_errorf("n_generations must be >= 1; got %d", static_cast<int>(n_generations));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (population_size < 2) {
        ctx.set_errorf("population_size must be >= 2; got %d", static_cast<int>(population_size));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (min_features < cfg.n_components || min_features > X.cols) {
        ctx.set_errorf("min_features must be in [n_components, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_features));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (max_features < min_features || max_features > X.cols) {
        ctx.set_errorf("max_features must be in [min_features, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(max_features));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(mutation_rate) || mutation_rate < 0.0 || mutation_rate > 1.0) {
        ctx.set_error("mutation_rate must be finite and in [0, 1]");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("GA-PLS matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_columns(::n4m::core::Context& ctx,
                                        const n4m_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("GA-PLS column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("GA-PLS subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("GA-PLS column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("GA-PLS X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t compute_subset_scores(::n4m::core::Context& ctx,
                                                 const ::n4m::core::Config& cfg,
                                                 const n4m_matrix_view_t& subset_x,
                                                 const n4m_matrix_view_t& Y,
                                                 std::vector<double>& out) {
    std::unique_ptr<::n4m::core::Model> model;
    n4m_status_t status = ::n4m::core::fit_model(ctx, cfg, subset_x, Y, model);
    if (status != N4M_OK) {
        return status;
    }
    const auto p = static_cast<std::size_t>(subset_x.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("GA-PLS fitted model returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }

    out.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double best = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            best = std::max(best, std::fabs(model->coefficients[idx(feature, q, target)]));
        }
        out[feature] = best;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t subset_rmse(::n4m::core::Context& ctx,
                                       const ::n4m::core::Config& cfg,
                                       const n4m_matrix_view_t& X,
                                       const n4m_matrix_view_t& Y,
                                       const ::n4m::core::ValidationPlan& plan,
                                       const std::vector<std::int64_t>& columns,
                                       double& out) {
    std::vector<double> subset_x;
    n4m_status_t status = copy_columns(ctx, X, columns, subset_x);
    if (status != N4M_OK) {
        return status;
    }
    n4m_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows, static_cast<std::int64_t>(columns.size()));
    ::n4m::core::CrossValidationResult cv;
    status = ::n4m::core::cross_validate_regression(ctx, cfg, subset_view, Y, plan, cv);
    if (status != N4M_OK) {
        return status;
    }
    out = cv.metrics.rmse;
    return N4M_OK;
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
}

[[nodiscard]] std::size_t random_bounded(std::uint64_t& state, std::size_t bound) noexcept {
    return static_cast<std::size_t>(splitmix64_next(state) % static_cast<std::uint64_t>(bound));
}

[[nodiscard]] double uniform01(std::uint64_t& state) noexcept {
    const std::uint64_t bits = splitmix64_next(state);
    return static_cast<double>(bits >> 11U) * kDoubleUnit;
}

[[nodiscard]] std::vector<unsigned char> membership(std::size_t p,
                                                    const std::vector<std::int64_t>& selected) {
    std::vector<unsigned char> out(p, 0U);
    for (const std::int64_t feature : selected) {
        out[static_cast<std::size_t>(feature)] = 1U;
    }
    return out;
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

[[nodiscard]] std::vector<std::int64_t> choose_ranked_features(std::size_t p,
                                                               const std::vector<double>& global_scores,
                                                               std::int32_t target_size,
                                                               std::uint64_t& state) {
    std::vector<std::int64_t> available(p, 0);
    std::iota(available.begin(), available.end(), static_cast<std::int64_t>(0));
    std::stable_sort(available.begin(), available.end(), [&global_scores](std::int64_t lhs,
                                                                          std::int64_t rhs) {
        const double left = global_scores[static_cast<std::size_t>(lhs)];
        const double right = global_scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });
    std::vector<std::int64_t> selected;
    selected.reserve(static_cast<std::size_t>(target_size));
    for (std::int32_t remaining = target_size; remaining > 0; --remaining) {
        if (available.empty()) {
            break;
        }
        const std::size_t window = std::max(static_cast<std::size_t>(remaining),
                                            (available.size() + 1U) / 2U);
        const std::size_t pos = random_bounded(state, window);
        selected.push_back(available[pos]);
        available.erase(available.begin() + static_cast<std::ptrdiff_t>(pos));
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

[[nodiscard]] std::vector<std::int64_t> remove_feature(const std::vector<std::int64_t>& selected,
                                                       const std::vector<double>& global_scores,
                                                       std::uint64_t& state) {
    std::vector<std::int64_t> candidate = selected;
    std::vector<std::int64_t> order = candidate;
    std::stable_sort(order.begin(), order.end(), [&global_scores](std::int64_t lhs,
                                                                  std::int64_t rhs) {
        const double left = global_scores[static_cast<std::size_t>(lhs)];
        const double right = global_scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs > rhs;
        }
        return left < right;
    });
    const std::size_t window = std::max<std::size_t>(1U, (order.size() + 1U) / 2U);
    const std::int64_t feature = order[random_bounded(state, window)];
    const auto it = std::find(candidate.begin(), candidate.end(), feature);
    if (it != candidate.end()) {
        candidate.erase(it);
    }
    std::sort(candidate.begin(), candidate.end());
    return candidate;
}

[[nodiscard]] std::vector<std::int64_t> add_feature(const std::vector<std::int64_t>& selected,
                                                    std::size_t p,
                                                    const std::vector<double>& global_scores,
                                                    std::uint64_t& state) {
    std::vector<std::int64_t> candidate = selected;
    const std::vector<unsigned char> is_selected = membership(p, candidate);
    std::vector<std::int64_t> available;
    available.reserve(p - candidate.size());
    for (std::size_t feature = 0; feature < p; ++feature) {
        if (is_selected[feature] == 0U) {
            available.push_back(static_cast<std::int64_t>(feature));
        }
    }
    std::stable_sort(available.begin(), available.end(), [&global_scores](std::int64_t lhs,
                                                                          std::int64_t rhs) {
        const double left = global_scores[static_cast<std::size_t>(lhs)];
        const double right = global_scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });
    if (available.empty()) {
        return candidate;
    }
    const std::size_t window = std::max<std::size_t>(1U, (available.size() + 1U) / 2U);
    candidate.push_back(available[random_bounded(state, window)]);
    std::sort(candidate.begin(), candidate.end());
    return candidate;
}

[[nodiscard]] std::vector<std::int64_t> crossover(const std::vector<std::int64_t>& parent_a,
                                                  const std::vector<std::int64_t>& parent_b,
                                                  std::size_t p,
                                                  const std::vector<double>& global_scores,
                                                  std::int32_t min_features,
                                                  std::int32_t max_features,
                                                  std::uint64_t& state) {
    const std::int32_t move = static_cast<std::int32_t>(random_bounded(state, 3U)) - 1;
    std::int32_t target_size =
        static_cast<std::int32_t>((parent_a.size() + parent_b.size()) / 2U) + move;
    target_size = std::max(min_features, std::min(max_features, target_size));

    const std::vector<unsigned char> in_b = membership(p, parent_b);
    std::vector<std::int64_t> child;
    for (const std::int64_t feature : parent_a) {
        if (in_b[static_cast<std::size_t>(feature)] != 0U) {
            child.push_back(feature);
        }
    }
    if (child.size() > static_cast<std::size_t>(target_size)) {
        std::vector<std::int64_t> order = child;
        std::stable_sort(order.begin(), order.end(), [&global_scores](std::int64_t lhs,
                                                                      std::int64_t rhs) {
            const double left = global_scores[static_cast<std::size_t>(lhs)];
            const double right = global_scores[static_cast<std::size_t>(rhs)];
            if (left == right) {
                return lhs > rhs;
            }
            return left < right;
        });
        const std::size_t remove_count = child.size() - static_cast<std::size_t>(target_size);
        for (std::size_t i = 0; i < remove_count; ++i) {
            const auto it = std::find(child.begin(), child.end(), order[i]);
            if (it != child.end()) {
                child.erase(it);
            }
        }
        std::sort(child.begin(), child.end());
    }

    std::vector<std::int64_t> union_features = parent_a;
    std::vector<unsigned char> in_union = membership(p, union_features);
    for (const std::int64_t feature : parent_b) {
        if (in_union[static_cast<std::size_t>(feature)] == 0U) {
            union_features.push_back(feature);
            in_union[static_cast<std::size_t>(feature)] = 1U;
        }
    }
    std::stable_sort(union_features.begin(), union_features.end(), [&global_scores](std::int64_t lhs,
                                                                                    std::int64_t rhs) {
        const double left = global_scores[static_cast<std::size_t>(lhs)];
        const double right = global_scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });

    while (child.size() < static_cast<std::size_t>(target_size)) {
        const std::vector<unsigned char> in_child = membership(p, child);
        std::vector<std::int64_t> available;
        for (const std::int64_t feature : union_features) {
            if (in_child[static_cast<std::size_t>(feature)] == 0U) {
                available.push_back(feature);
            }
        }
        if (available.empty()) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                if (in_child[feature] == 0U) {
                    available.push_back(static_cast<std::int64_t>(feature));
                }
            }
            std::stable_sort(available.begin(), available.end(), [&global_scores](std::int64_t lhs,
                                                                                  std::int64_t rhs) {
                const double left = global_scores[static_cast<std::size_t>(lhs)];
                const double right = global_scores[static_cast<std::size_t>(rhs)];
                if (left == right) {
                    return lhs < rhs;
                }
                return left > right;
            });
        }
        if (available.empty()) {
            break;
        }
        const std::size_t window = std::max<std::size_t>(1U, (available.size() + 1U) / 2U);
        child.push_back(available[random_bounded(state, window)]);
        std::sort(child.begin(), child.end());
    }
    return child;
}

[[nodiscard]] std::vector<std::int64_t> mutate(const std::vector<std::int64_t>& selected,
                                               std::size_t p,
                                               const std::vector<double>& global_scores,
                                               std::int32_t min_features,
                                               std::int32_t max_features,
                                               double mutation_rate,
                                               std::uint64_t& state) {
    if (uniform01(state) >= mutation_rate) {
        return selected;
    }
    const std::size_t move = random_bounded(state, 3U);
    if (move == 0U && selected.size() > static_cast<std::size_t>(min_features)) {
        return remove_feature(selected, global_scores, state);
    }
    if (move == 1U && selected.size() < static_cast<std::size_t>(max_features)) {
        return add_feature(selected, p, global_scores, state);
    }
    if (selected.size() > static_cast<std::size_t>(min_features) && selected.size() < p) {
        std::vector<std::int64_t> reduced = remove_feature(selected, global_scores, state);
        return add_feature(reduced, p, global_scores, state);
    }
    return selected;
}

[[nodiscard]] bool evaluated_less(const EvaluatedSubset& lhs,
                                  const EvaluatedSubset& rhs) {
    if (lhs.rmse == rhs.rmse) {
        if (lhs.size == rhs.size) {
            return std::lexicographical_compare(lhs.indices.begin(),
                                                lhs.indices.end(),
                                                rhs.indices.begin(),
                                                rhs.indices.end());
        }
        return lhs.size < rhs.size;
    }
    return lhs.rmse < rhs.rmse;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_ga(Context& ctx,
                          const Config& cfg,
                          const n4m_matrix_view_t& X,
                          const n4m_matrix_view_t& Y,
                          const ValidationPlan& plan,
                          std::int32_t n_generations,
                          std::int32_t population_size,
                          std::int32_t min_features,
                          std::int32_t max_features,
                          double mutation_rate,
                          std::uint64_t seed,
                          GaSelectionResult& out) {
    try {
        out = GaSelectionResult{};
        n4m_status_t status = validate_request(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               n_generations,
                                               population_size,
                                               min_features,
                                               max_features,
                                               mutation_rate);
        if (status != N4M_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(X.cols);
        std::vector<std::int64_t> all_columns(p, 0);
        std::iota(all_columns.begin(), all_columns.end(), static_cast<std::int64_t>(0));
        std::vector<double> full_x;
        status = copy_columns(ctx, X, all_columns, full_x);
        if (status != N4M_OK) {
            out = GaSelectionResult{};
            return status;
        }
        n4m_matrix_view_t full_x_view = rowmajor_f64_view(full_x, X.rows, X.cols);
        status = compute_subset_scores(ctx, cfg, full_x_view, Y, out.global_scores);
        if (status != N4M_OK) {
            out = GaSelectionResult{};
            return status;
        }
        if (out.global_scores.size() != p) {
            ctx.set_error("GA-PLS global score count is inconsistent");
            out = GaSelectionResult{};
            return N4M_ERR_INTERNAL;
        }

        std::uint64_t state = seed;
        std::vector<std::int64_t> ranked = rank_descending(out.global_scores);
        ranked.resize(static_cast<std::size_t>(max_features));
        std::sort(ranked.begin(), ranked.end());
        std::vector<std::vector<std::int64_t>> population;
        population.reserve(static_cast<std::size_t>(population_size));
        population.push_back(ranked);
        const std::int32_t span = max_features - min_features + 1;
        for (std::int32_t individual = 1; individual < population_size; ++individual) {
            const std::int32_t target_size = min_features + (individual % span);
            population.push_back(choose_ranked_features(p, out.global_scores, target_size, state));
        }

        std::vector<double> inclusion_counts(p, 0.0);
        double best_rmse = std::numeric_limits<double>::infinity();
        std::vector<std::int64_t> best_indices;
        out.best_rmse_by_generation.reserve(static_cast<std::size_t>(n_generations));
        out.mean_rmse_by_generation.reserve(static_cast<std::size_t>(n_generations));
        out.best_subset_sizes.reserve(static_cast<std::size_t>(n_generations));

        for (std::int32_t generation = 0; generation < n_generations; ++generation) {
            std::vector<EvaluatedSubset> evaluated;
            evaluated.reserve(population.size());
            double rmse_sum = 0.0;
            for (const auto& individual : population) {
                double rmse = 0.0;
                status = subset_rmse(ctx, cfg, X, Y, plan, individual, rmse);
                if (status != N4M_OK) {
                    out = GaSelectionResult{};
                    return status;
                }
                rmse_sum += rmse;
                for (const std::int64_t feature : individual) {
                    inclusion_counts[static_cast<std::size_t>(feature)] += 1.0;
                }
                evaluated.push_back(EvaluatedSubset{
                    rmse,
                    static_cast<std::int64_t>(individual.size()),
                    individual,
                });
            }
            std::stable_sort(evaluated.begin(), evaluated.end(), evaluated_less);
            const EvaluatedSubset& generation_best = evaluated.front();
            out.best_rmse_by_generation.push_back(generation_best.rmse);
            out.mean_rmse_by_generation.push_back(rmse_sum / static_cast<double>(evaluated.size()));
            out.best_subset_sizes.push_back(generation_best.size);
            if (generation_best.rmse < best_rmse) {
                best_rmse = generation_best.rmse;
                best_indices = generation_best.indices;
            }

            if (generation == n_generations - 1) {
                break;
            }
            const auto elite_count = static_cast<std::size_t>(std::min(2, population_size));
            const auto parent_pool_size = static_cast<std::size_t>(
                std::max(static_cast<std::int32_t>(elite_count), population_size / 2));
            std::vector<std::vector<std::int64_t>> parent_pool;
            parent_pool.reserve(parent_pool_size);
            for (std::size_t i = 0; i < parent_pool_size; ++i) {
                parent_pool.push_back(evaluated[i].indices);
            }
            std::vector<std::vector<std::int64_t>> next_population;
            next_population.reserve(static_cast<std::size_t>(population_size));
            for (std::size_t i = 0; i < elite_count; ++i) {
                next_population.push_back(evaluated[i].indices);
            }
            while (next_population.size() < static_cast<std::size_t>(population_size)) {
                const std::size_t parent_a = random_bounded(state, parent_pool.size());
                const std::size_t parent_b = random_bounded(state, parent_pool.size());
                std::vector<std::int64_t> child = crossover(parent_pool[parent_a],
                                                            parent_pool[parent_b],
                                                            p,
                                                            out.global_scores,
                                                            min_features,
                                                            max_features,
                                                            state);
                child = mutate(child,
                               p,
                               out.global_scores,
                               min_features,
                               max_features,
                               mutation_rate,
                               state);
                next_population.push_back(std::move(child));
            }
            population = std::move(next_population);
        }

        out.inclusion_frequencies.assign(p, 0.0);
        const double denom = static_cast<double>(n_generations) * static_cast<double>(population_size);
        for (std::size_t feature = 0; feature < p; ++feature) {
            out.inclusion_frequencies[feature] = inclusion_counts[feature] / denom;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_generations = n_generations;
        out.population_size = population_size;
        out.min_features = min_features;
        out.max_features = max_features;
        out.mutation_rate = mutation_rate;
        out.seed = seed;
        out.best_rmse = best_rmse;
        out.selected_indices = std::move(best_indices);
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running GA-PLS selection");
        out = GaSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running GA-PLS selection");
        out = GaSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
