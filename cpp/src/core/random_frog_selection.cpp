// SPDX-License-Identifier: CECILL-2.1

#include "core/random_frog_selection.hpp"

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
                                            const ::pls4all::core::Config& cfg,
                                            const p4a_matrix_view_t& X,
                                            const p4a_matrix_view_t& Y,
                                            const ::pls4all::core::ValidationPlan& plan,
                                            std::int32_t n_iterations,
                                            std::int32_t initial_size,
                                            std::int32_t min_size,
                                            std::int32_t max_size,
                                            std::int32_t top_k) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("Random Frog matrices must be non-empty");
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
        ctx.set_error("Random Frog requires at least one validation fold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_iterations <= 0) {
        ctx.set_errorf("n_iterations must be >= 1; got %d", static_cast<int>(n_iterations));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (min_size < cfg.n_components || min_size > X.cols) {
        ctx.set_errorf("min_size must be in [n_components, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_size));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (max_size < min_size || max_size > X.cols) {
        ctx.set_errorf("max_size must be in [min_size, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(max_size));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (initial_size < min_size || initial_size > max_size) {
        ctx.set_error("initial_size must be within [min_size, max_size]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (top_k <= 0 || top_k > X.cols) {
        ctx.set_errorf("top_k must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(top_k));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("Random Frog matrix dimensions exceed int32 result storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_columns(::pls4all::core::Context& ctx,
                                        const p4a_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("Random Frog column selection must not be empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("Random Frog subset matrix shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("Random Frog column index out of range");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("Random Frog X contains NaN or Inf");
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t compute_subset_scores(::pls4all::core::Context& ctx,
                                                 const ::pls4all::core::Config& cfg,
                                                 const p4a_matrix_view_t& subset_x,
                                                 const p4a_matrix_view_t& Y,
                                                 std::vector<double>& out) {
    std::unique_ptr<::pls4all::core::Model> model;
    p4a_status_t status = ::pls4all::core::fit_model(ctx, cfg, subset_x, Y, model);
    if (status != P4A_OK) {
        return status;
    }
    const auto p = static_cast<std::size_t>(subset_x.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("Random Frog fitted model returned inconsistent coefficients");
        return P4A_ERR_INTERNAL;
    }

    out.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double best = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            best = std::max(best, std::fabs(model->coefficients[idx(feature, q, target)]));
        }
        out[feature] = best;
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
    if (status != P4A_OK) {
        return status;
    }
    p4a_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows, static_cast<std::int64_t>(columns.size()));
    ::pls4all::core::CrossValidationResult cv;
    status = ::pls4all::core::cross_validate_regression(ctx, cfg, subset_view, Y, plan, cv);
    if (status != P4A_OK) {
        return status;
    }
    out = cv.metrics.rmse;
    return P4A_OK;
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

[[nodiscard]] std::vector<std::int64_t> add_features(
    const std::vector<std::int64_t>& selected,
    std::size_t p,
    const std::vector<double>& global_scores,
    std::int32_t count,
    std::uint64_t& state) {
    std::vector<std::int64_t> candidate = selected;
    const std::vector<unsigned char> is_selected = membership(p, selected);
    std::vector<std::int64_t> available;
    available.reserve(p - selected.size());
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

    for (std::int32_t remaining = count; remaining > 0; --remaining) {
        if (available.empty()) {
            break;
        }
        const std::size_t window = std::max(static_cast<std::size_t>(remaining),
                                            (available.size() + 1U) / 2U);
        const std::size_t pos = random_bounded(state, window);
        candidate.push_back(available[pos]);
        available.erase(available.begin() + static_cast<std::ptrdiff_t>(pos));
    }
    std::sort(candidate.begin(), candidate.end());
    return candidate;
}

[[nodiscard]] std::vector<std::int64_t> remove_features(
    const std::vector<std::int64_t>& selected,
    const std::vector<double>& current_scores,
    std::int32_t count,
    std::uint64_t& state) {
    std::vector<std::int64_t> candidate = selected;
    std::vector<std::int64_t> order = candidate;
    std::stable_sort(order.begin(), order.end(), [&current_scores](std::int64_t lhs,
                                                                   std::int64_t rhs) {
        const double left = current_scores[static_cast<std::size_t>(lhs)];
        const double right = current_scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs > rhs;
        }
        return left < right;
    });

    for (std::int32_t remaining = count; remaining > 0; --remaining) {
        if (order.empty()) {
            break;
        }
        const std::size_t window = std::max(static_cast<std::size_t>(remaining),
                                            (order.size() + 1U) / 2U);
        const std::size_t pos = random_bounded(state, window);
        const std::int64_t feature = order[pos];
        order.erase(order.begin() + static_cast<std::ptrdiff_t>(pos));
        const auto it = std::find(candidate.begin(), candidate.end(), feature);
        if (it != candidate.end()) {
            candidate.erase(it);
        }
    }
    std::sort(candidate.begin(), candidate.end());
    return candidate;
}

[[nodiscard]] std::vector<std::int64_t> swap_feature(
    const std::vector<std::int64_t>& selected,
    std::size_t p,
    const std::vector<double>& global_scores,
    const std::vector<double>& current_scores,
    std::uint64_t& state) {
    if (selected.empty() || selected.size() == p) {
        return selected;
    }
    std::vector<std::int64_t> reduced = remove_features(selected, current_scores, 1, state);
    const std::vector<unsigned char> was_selected = membership(p, selected);
    std::vector<std::int64_t> available;
    available.reserve(p - selected.size());
    for (std::size_t feature = 0; feature < p; ++feature) {
        if (was_selected[feature] == 0U) {
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
        return selected;
    }
    const std::size_t window = std::max<std::size_t>(1U, (available.size() + 1U) / 2U);
    const std::size_t pos = random_bounded(state, window);
    reduced.push_back(available[pos]);
    std::sort(reduced.begin(), reduced.end());
    return reduced;
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_random_frog(Context& ctx,
                                   const Config& cfg,
                                   const p4a_matrix_view_t& X,
                                   const p4a_matrix_view_t& Y,
                                   const ValidationPlan& plan,
                                   std::int32_t n_iterations,
                                   std::int32_t initial_size,
                                   std::int32_t min_size,
                                   std::int32_t max_size,
                                   std::int32_t top_k,
                                   std::uint64_t seed,
                                   RandomFrogSelectionResult& out) {
    try {
        out = RandomFrogSelectionResult{};
        p4a_status_t status = validate_request(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               n_iterations,
                                               initial_size,
                                               min_size,
                                               max_size,
                                               top_k);
        if (status != P4A_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(X.cols);
        std::vector<std::int64_t> all_columns(p, 0);
        std::iota(all_columns.begin(), all_columns.end(), static_cast<std::int64_t>(0));
        std::vector<double> full_x;
        status = copy_columns(ctx, X, all_columns, full_x);
        if (status != P4A_OK) {
            out = RandomFrogSelectionResult{};
            return status;
        }
        p4a_matrix_view_t full_x_view = rowmajor_f64_view(full_x, X.rows, X.cols);
        status = compute_subset_scores(ctx, cfg, full_x_view, Y, out.global_scores);
        if (status != P4A_OK) {
            out = RandomFrogSelectionResult{};
            return status;
        }
        if (out.global_scores.size() != p) {
            ctx.set_error("Random Frog global score count is inconsistent");
            out = RandomFrogSelectionResult{};
            return P4A_ERR_INTERNAL;
        }

        std::vector<std::int64_t> ranked = rank_descending(out.global_scores);
        ranked.resize(static_cast<std::size_t>(initial_size));
        std::vector<std::int64_t> current = ranked;
        std::sort(current.begin(), current.end());

        double current_rmse = 0.0;
        status = subset_rmse(ctx, cfg, X, Y, plan, current, current_rmse);
        if (status != P4A_OK) {
            out = RandomFrogSelectionResult{};
            return status;
        }

        std::vector<double> inclusion_counts(p, 0.0);
        std::vector<std::int64_t> best_indices = current;
        double best_rmse = current_rmse;
        std::uint64_t state = seed;
        out.rmse_by_iteration.reserve(static_cast<std::size_t>(n_iterations));
        out.subset_sizes.reserve(static_cast<std::size_t>(n_iterations));

        for (std::int32_t iteration = 0; iteration < n_iterations; ++iteration) {
            std::vector<double> current_x;
            status = copy_columns(ctx, X, current, current_x);
            if (status != P4A_OK) {
                out = RandomFrogSelectionResult{};
                return status;
            }
            p4a_matrix_view_t current_x_view =
                rowmajor_f64_view(current_x,
                                  X.rows,
                                  static_cast<std::int64_t>(current.size()));
            std::vector<double> active_scores;
            status = compute_subset_scores(ctx, cfg, current_x_view, Y, active_scores);
            if (status != P4A_OK) {
                out = RandomFrogSelectionResult{};
                return status;
            }
            if (active_scores.size() != current.size()) {
                ctx.set_error("Random Frog active score count is inconsistent");
                out = RandomFrogSelectionResult{};
                return P4A_ERR_INTERNAL;
            }
            std::vector<double> current_scores(p, 0.0);
            for (std::size_t local = 0; local < current.size(); ++local) {
                current_scores[static_cast<std::size_t>(current[local])] = active_scores[local];
            }

            const std::int32_t move = static_cast<std::int32_t>(random_bounded(state, 3U)) - 1;
            std::int32_t proposed_size = static_cast<std::int32_t>(current.size()) + move;
            proposed_size = std::max(min_size, std::min(max_size, proposed_size));

            std::vector<std::int64_t> candidate;
            if (proposed_size > static_cast<std::int32_t>(current.size())) {
                candidate = add_features(current,
                                         p,
                                         out.global_scores,
                                         proposed_size - static_cast<std::int32_t>(current.size()),
                                         state);
            } else if (proposed_size < static_cast<std::int32_t>(current.size())) {
                candidate = remove_features(current,
                                            current_scores,
                                            static_cast<std::int32_t>(current.size()) - proposed_size,
                                            state);
            } else {
                candidate = swap_feature(current, p, out.global_scores, current_scores, state);
            }
            if (candidate.size() < static_cast<std::size_t>(min_size)) {
                candidate = current;
            }

            double candidate_rmse = 0.0;
            status = subset_rmse(ctx, cfg, X, Y, plan, candidate, candidate_rmse);
            if (status != P4A_OK) {
                out = RandomFrogSelectionResult{};
                return status;
            }

            bool accept = candidate_rmse <= current_rmse;
            if (!accept) {
                const double temperature = std::max(std::fabs(current_rmse) * 0.05, 1e-12);
                const double probability = std::exp((current_rmse - candidate_rmse) / temperature);
                accept = uniform01(state) < probability;
            }
            if (accept) {
                current = std::move(candidate);
                current_rmse = candidate_rmse;
            }

            for (const std::int64_t feature : current) {
                inclusion_counts[static_cast<std::size_t>(feature)] += 1.0;
            }
            out.rmse_by_iteration.push_back(current_rmse);
            out.subset_sizes.push_back(static_cast<std::int64_t>(current.size()));
            if (current_rmse < best_rmse) {
                best_rmse = current_rmse;
                best_indices = current;
            }
        }

        out.inclusion_frequencies.assign(p, 0.0);
        const double denom = static_cast<double>(n_iterations);
        for (std::size_t feature = 0; feature < p; ++feature) {
            out.inclusion_frequencies[feature] = inclusion_counts[feature] / denom;
        }

        std::vector<std::int64_t> selected(p, 0);
        std::iota(selected.begin(), selected.end(), static_cast<std::int64_t>(0));
        std::stable_sort(selected.begin(), selected.end(), [&out](std::int64_t lhs, std::int64_t rhs) {
            const auto l = static_cast<std::size_t>(lhs);
            const auto r = static_cast<std::size_t>(rhs);
            if (out.inclusion_frequencies[l] == out.inclusion_frequencies[r]) {
                if (out.global_scores[l] == out.global_scores[r]) {
                    return lhs < rhs;
                }
                return out.global_scores[l] > out.global_scores[r];
            }
            return out.inclusion_frequencies[l] > out.inclusion_frequencies[r];
        });
        selected.resize(static_cast<std::size_t>(top_k));
        std::sort(selected.begin(), selected.end());

        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_iterations = n_iterations;
        out.initial_size = initial_size;
        out.min_size = min_size;
        out.max_size = max_size;
        out.top_k = top_k;
        out.seed = seed;
        out.best_rmse = best_rmse;
        out.selected_indices = std::move(selected);
        out.best_indices = std::move(best_indices);
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running Random Frog selection");
        out = RandomFrogSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running Random Frog selection");
        out = RandomFrogSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
