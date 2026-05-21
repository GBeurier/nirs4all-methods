// SPDX-License-Identifier: CECILL-2.1

#include "core/scars_selection.hpp"

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
#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

namespace {

constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;

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

[[nodiscard]] std::int32_t sample_count_for(std::int64_t rows,
                                            std::int32_t n_components,
                                            double sample_fraction) noexcept {
    const auto requested = static_cast<std::int32_t>(
        std::ceil(static_cast<double>(rows) * sample_fraction));
    const std::int32_t lower = std::max(2, n_components);
    return std::min(static_cast<std::int32_t>(rows), std::max(lower, requested));
}

[[nodiscard]] n4m_status_t validate_request(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& cfg,
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            const ::n4m::core::ValidationPlan& plan,
                                            std::int32_t n_iterations,
                                            std::int32_t min_features,
                                            double sample_fraction) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("SCARS matrices must be non-empty");
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
        ctx.set_error("SCARS requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_iterations <= 0) {
        ctx.set_errorf("n_iterations must be >= 1; got %d", static_cast<int>(n_iterations));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows < std::max<std::int64_t>(2, cfg.n_components)) {
        ctx.set_error("SCARS requires enough rows for the requested component count");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (min_features < cfg.n_components || min_features > X.cols) {
        ctx.set_errorf("min_features must be in [n_components, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_features));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(sample_fraction) || sample_fraction <= 0.0 || sample_fraction > 1.0) {
        ctx.set_error("sample_fraction must be finite and in (0, 1]");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        X.rows > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("SCARS matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
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

[[nodiscard]] std::vector<std::int64_t> sample_rows(std::size_t n_samples,
                                                    std::size_t sample_count,
                                                    std::uint64_t& state) {
    std::vector<std::pair<std::uint64_t, std::int64_t>> keyed;
    keyed.reserve(n_samples);
    for (std::size_t sample = 0; sample < n_samples; ++sample) {
        keyed.emplace_back(splitmix64_next(state), static_cast<std::int64_t>(sample));
    }
    std::sort(keyed.begin(), keyed.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
    std::vector<std::int64_t> rows;
    rows.reserve(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        rows.push_back(keyed[i].second);
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

[[nodiscard]] n4m_status_t copy_rows(::n4m::core::Context& ctx,
                                     const n4m_matrix_view_t& view,
                                     const std::vector<std::int64_t>& rows,
                                     const char* name,
                                     std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_matrix_size(static_cast<std::int64_t>(rows.size()), view.cols, n_values)) {
        ctx.set_errorf("%s row selection shape is too large", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto cols = static_cast<std::size_t>(view.cols);
    for (std::size_t local_row = 0; local_row < rows.size(); ++local_row) {
        const std::int64_t original_row = rows[local_row];
        if (original_row < 0 || original_row >= view.rows) {
            ctx.set_errorf("%s row index out of range", name);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_row = static_cast<std::size_t>(original_row);
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(view, src_row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(local_row, cols, col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_columns(::n4m::core::Context& ctx,
                                        const n4m_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("SCARS column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("SCARS subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("SCARS column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("SCARS X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_rows_columns(::n4m::core::Context& ctx,
                                             const n4m_matrix_view_t& X,
                                             const std::vector<std::int64_t>& rows,
                                             const std::vector<std::int64_t>& columns,
                                             std::vector<double>& out) {
    if (rows.empty() || columns.empty()) {
        ctx.set_error("SCARS sampled matrix must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(static_cast<std::int64_t>(rows.size()),
                             static_cast<std::int64_t>(columns.size()),
                             n_values)) {
        ctx.set_error("SCARS sampled matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto cols = columns.size();
    for (std::size_t local_row = 0; local_row < rows.size(); ++local_row) {
        const std::int64_t original_row = rows[local_row];
        if (original_row < 0 || original_row >= X.rows) {
            ctx.set_error("SCARS sampled row index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_row = static_cast<std::size_t>(original_row);
        for (std::size_t local_col = 0; local_col < cols; ++local_col) {
            const std::int64_t original_col = columns[local_col];
            if (original_col < 0 || original_col >= X.cols) {
                ctx.set_error("SCARS sampled column index out of range");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            const double value = read_value(X, src_row, static_cast<std::size_t>(original_col));
            if (!std::isfinite(value)) {
                ctx.set_error("SCARS sampled X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(local_row, cols, local_col)] = value;
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
        ctx.set_error("SCARS fitted model returned inconsistent coefficients");
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

[[nodiscard]] std::int32_t retained_count(std::int32_t n_features,
                                          std::int32_t min_features,
                                          std::int32_t iteration,
                                          std::int32_t n_iterations) noexcept {
    if (n_iterations <= 1) {
        return min_features;
    }
    const double ratio = static_cast<double>(min_features) / static_cast<double>(n_features);
    const double fraction = static_cast<double>(iteration) / static_cast<double>(n_iterations - 1);
    const auto retained = static_cast<std::int32_t>(
        std::ceil(static_cast<double>(n_features) * std::pow(ratio, fraction)));
    return std::max(min_features, std::min(n_features, retained));
}

[[nodiscard]] std::vector<std::int64_t> rank_active(
    const std::vector<std::int64_t>& active,
    const std::vector<double>& stability_scores,
    const std::vector<double>& full_scores) {
    std::vector<std::int64_t> order = active;
    std::stable_sort(order.begin(), order.end(), [&stability_scores, &full_scores](std::int64_t lhs,
                                                                                   std::int64_t rhs) {
        const auto left = static_cast<std::size_t>(lhs);
        const auto right = static_cast<std::size_t>(rhs);
        if (stability_scores[left] == stability_scores[right]) {
            if (full_scores[left] == full_scores[right]) {
                return lhs < rhs;
            }
            return full_scores[left] > full_scores[right];
        }
        return stability_scores[left] > stability_scores[right];
    });
    return order;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_scars(Context& ctx,
                             const Config& cfg,
                             const n4m_matrix_view_t& X,
                             const n4m_matrix_view_t& Y,
                             const ValidationPlan& plan,
                             std::int32_t n_iterations,
                             std::int32_t min_features,
                             double sample_fraction,
                             std::uint64_t seed,
                             ScarsSelectionResult& out) {
    try {
        out = ScarsSelectionResult{};
        n4m_status_t status = validate_request(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               n_iterations,
                                               min_features,
                                               sample_fraction);
        if (status != N4M_OK) {
            return status;
        }

        const auto n = static_cast<std::size_t>(X.rows);
        const auto p = static_cast<std::size_t>(X.cols);
        const auto sample_count = static_cast<std::size_t>(
            sample_count_for(X.rows, cfg.n_components, sample_fraction));
        std::vector<std::int64_t> active(p, 0);
        std::iota(active.begin(), active.end(), static_cast<std::int64_t>(0));

        std::vector<double> stability_sums(p, 0.0);
        std::vector<std::vector<std::int64_t>> candidates;
        candidates.reserve(static_cast<std::size_t>(n_iterations));
        out.coefficient_scores.assign(static_cast<std::size_t>(n_iterations) * p, 0.0);
        out.rmse_by_iteration.reserve(static_cast<std::size_t>(n_iterations));
        out.retained_counts.reserve(static_cast<std::size_t>(n_iterations));

        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t best_iteration = -1;
        std::uint64_t state = seed;

        for (std::int32_t iteration = 0; iteration < n_iterations; ++iteration) {
            std::vector<std::int64_t> rows = sample_rows(n, sample_count, state);

            std::vector<double> active_x;
            status = copy_rows_columns(ctx, X, rows, active, active_x);
            if (status != N4M_OK) {
                out = ScarsSelectionResult{};
                return status;
            }
            std::vector<double> active_y;
            status = copy_rows(ctx, Y, rows, "SCARS sampled Y", active_y);
            if (status != N4M_OK) {
                out = ScarsSelectionResult{};
                return status;
            }
            n4m_matrix_view_t active_x_view =
                rowmajor_f64_view(active_x,
                                  static_cast<std::int64_t>(rows.size()),
                                  static_cast<std::int64_t>(active.size()));
            n4m_matrix_view_t active_y_view =
                rowmajor_f64_view(active_y,
                                  static_cast<std::int64_t>(rows.size()),
                                  Y.cols);

            std::vector<double> active_scores;
            status = compute_subset_scores(ctx, cfg, active_x_view, active_y_view, active_scores);
            if (status != N4M_OK) {
                out = ScarsSelectionResult{};
                return status;
            }
            if (active_scores.size() != active.size()) {
                ctx.set_error("SCARS active score count is inconsistent");
                out = ScarsSelectionResult{};
                return N4M_ERR_INTERNAL;
            }

            std::vector<double> full_scores(p, 0.0);
            double score_sum = 0.0;
            for (const double score : active_scores) {
                score_sum += score;
            }
            for (std::size_t local = 0; local < active.size(); ++local) {
                const auto feature = static_cast<std::size_t>(active[local]);
                const double score = active_scores[local];
                full_scores[feature] = score;
                out.coefficient_scores[static_cast<std::size_t>(iteration) * p + feature] = score;
                if (score_sum > std::numeric_limits<double>::epsilon()) {
                    stability_sums[feature] += score / score_sum;
                }
            }

            std::vector<double> running_stability(p, 0.0);
            const double denom = static_cast<double>(iteration + 1);
            for (std::size_t feature = 0; feature < p; ++feature) {
                running_stability[feature] = stability_sums[feature] / denom;
            }

            std::int32_t keep = retained_count(static_cast<std::int32_t>(p),
                                               min_features,
                                               iteration,
                                               n_iterations);
            keep = std::min(keep, static_cast<std::int32_t>(active.size()));
            std::vector<std::int64_t> ranked = rank_active(active, running_stability, full_scores);
            ranked.resize(static_cast<std::size_t>(keep));
            std::sort(ranked.begin(), ranked.end());

            double rmse = 0.0;
            status = subset_rmse(ctx, cfg, X, Y, plan, ranked, rmse);
            if (status != N4M_OK) {
                out = ScarsSelectionResult{};
                return status;
            }

            out.retained_counts.push_back(static_cast<std::int64_t>(keep));
            out.rmse_by_iteration.push_back(rmse);
            candidates.push_back(ranked);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_iteration = iteration;
            }
            active = std::move(ranked);
        }

        if (best_iteration < 0) {
            ctx.set_error("SCARS did not produce a candidate subset");
            out = ScarsSelectionResult{};
            return N4M_ERR_INTERNAL;
        }

        out.stability_scores.assign(p, 0.0);
        const double final_denom = static_cast<double>(n_iterations);
        for (std::size_t feature = 0; feature < p; ++feature) {
            out.stability_scores[feature] = stability_sums[feature] / final_denom;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_iterations = n_iterations;
        out.min_features = min_features;
        out.sample_count = static_cast<std::int32_t>(sample_count);
        out.best_iteration = best_iteration;
        out.sample_fraction = sample_fraction;
        out.seed = seed;
        out.best_rmse = best_rmse;
        out.selected_indices = candidates[static_cast<std::size_t>(best_iteration)];
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running SCARS selection");
        out = ScarsSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running SCARS selection");
        out = ScarsSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
