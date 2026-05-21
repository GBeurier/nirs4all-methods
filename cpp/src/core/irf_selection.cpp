// SPDX-License-Identifier: CECILL-2.1
//
// Interval Random Frog — Random Frog applied to fixed-width sliding
// intervals over the spectral axis. Mirrors libPLS 1.95 `irf.m`.

#include "core/irf_selection.hpp"

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
constexpr double kDoubleUnit = 1.0 / 9007199254740992.0;

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
}

[[nodiscard]] std::size_t random_bounded(std::uint64_t& state, std::size_t bound) noexcept {
    if (bound == 0) return 0;
    return static_cast<std::size_t>(splitmix64_next(state) %
                                     static_cast<std::uint64_t>(bound));
}

[[nodiscard]] double uniform01(std::uint64_t& state) noexcept {
    const std::uint64_t bits = splitmix64_next(state);
    return static_cast<double>(bits >> 11U) * kDoubleUnit;
}

// Box-Muller for standard normal.
double standard_normal(std::uint64_t& state) {
    double u1 = uniform01(state);
    if (u1 < 1e-12) u1 = 1e-12;
    const double u2 = uniform01(state);
    const double r = std::sqrt(-2.0 * std::log(u1));
    return r * std::cos(2.0 * 3.141592653589793 * u2);
}

[[nodiscard]] bool checked_matrix_size(std::int64_t rows,
                                       std::int64_t cols,
                                       std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) return false;
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
        return static_cast<const double*>(view.data)[uoff];
    }
    return static_cast<double>(static_cast<const float*>(view.data)[uoff]);
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

[[nodiscard]] n4m_status_t copy_columns(::n4m::core::Context& ctx,
                                        const n4m_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("IRF column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()),
                              n_values)) {
        ctx.set_error("IRF subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("IRF column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("IRF X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[row * cols + local_col] = value;
        }
    }
    return N4M_OK;
}

// Compute the union of feature indices selected by interval subset V.
std::vector<std::int64_t> union_features(
    const std::vector<std::int32_t>& V,
    const std::vector<std::vector<std::int64_t>>& intervals) {
    std::vector<unsigned char> hit;
    std::int64_t max_feature = -1;
    for (std::int32_t iv : V) {
        for (std::int64_t f : intervals[static_cast<std::size_t>(iv)]) {
            if (f > max_feature) max_feature = f;
        }
    }
    if (max_feature < 0) return {};
    hit.assign(static_cast<std::size_t>(max_feature + 1), 0U);
    for (std::int32_t iv : V) {
        for (std::int64_t f : intervals[static_cast<std::size_t>(iv)]) {
            hit[static_cast<std::size_t>(f)] = 1U;
        }
    }
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(max_feature + 1));
    for (std::int64_t f = 0; f <= max_feature; ++f) {
        if (hit[static_cast<std::size_t>(f)] != 0U) out.push_back(f);
    }
    return out;
}

[[nodiscard]] n4m_status_t fit_pls_coefs(::n4m::core::Context& ctx,
                                         ::n4m::core::Config cfg,
                                         const n4m_matrix_view_t& subset_x,
                                         const n4m_matrix_view_t& Y,
                                         std::vector<double>& abs_coefs_max) {
    const auto p = static_cast<std::size_t>(subset_x.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (cfg.n_components > static_cast<std::int32_t>(p)) {
        cfg.n_components = static_cast<std::int32_t>(p);
    }
    if (cfg.n_components < 1) cfg.n_components = 1;
    std::unique_ptr<::n4m::core::Model> model;
    n4m_status_t status = ::n4m::core::fit_model(ctx, cfg, subset_x, Y, model);
    if (status != N4M_OK) return status;
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("IRF fitted model returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }
    abs_coefs_max.assign(p, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        double best = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            best = std::max(best,
                            std::fabs(model->coefficients[f * q + target]));
        }
        abs_coefs_max[f] = best;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t subset_rmse(::n4m::core::Context& ctx,
                                       ::n4m::core::Config cfg,
                                       const n4m_matrix_view_t& X,
                                       const n4m_matrix_view_t& Y,
                                       const ::n4m::core::ValidationPlan& plan,
                                       const std::vector<std::int64_t>& columns,
                                       double& out) {
    std::vector<double> subset_x;
    n4m_status_t status = copy_columns(ctx, X, columns, subset_x);
    if (status != N4M_OK) return status;
    n4m_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows,
                          static_cast<std::int64_t>(columns.size()));
    if (cfg.n_components > static_cast<std::int32_t>(columns.size())) {
        cfg.n_components = static_cast<std::int32_t>(columns.size());
    }
    if (cfg.n_components < 1) cfg.n_components = 1;
    ::n4m::core::CrossValidationResult cv;
    status = ::n4m::core::cross_validate_regression(ctx, cfg, subset_view, Y,
                                                        plan, cv);
    if (status != N4M_OK) return status;
    out = cv.metrics.rmse;
    return N4M_OK;
}

// Build the "interval-score" vector: for each interval index in V, sum
// of |coef| over features that belong to that interval (intersected with
// the union U).
void score_intervals(
    const std::vector<std::int32_t>& V,
    const std::vector<std::vector<std::int64_t>>& intervals,
    const std::vector<std::int64_t>& U,
    const std::vector<double>& abs_coefs,
    std::vector<double>& out_scores) {
    out_scores.assign(V.size(), 0.0);
    // Build a feature → index-in-U lookup for O(p) intersection.
    std::int64_t max_feature = -1;
    for (std::int64_t f : U) {
        if (f > max_feature) max_feature = f;
    }
    if (max_feature < 0) return;
    std::vector<std::int64_t> lookup(
        static_cast<std::size_t>(max_feature + 1), -1);
    for (std::size_t idx_u = 0; idx_u < U.size(); ++idx_u) {
        lookup[static_cast<std::size_t>(U[idx_u])] =
            static_cast<std::int64_t>(idx_u);
    }
    for (std::size_t k = 0; k < V.size(); ++k) {
        double s = 0.0;
        for (std::int64_t f : intervals[static_cast<std::size_t>(V[k])]) {
            if (f >= 0 && f <= max_feature) {
                const std::int64_t pos =
                    lookup[static_cast<std::size_t>(f)];
                if (pos >= 0) s += abs_coefs[static_cast<std::size_t>(pos)];
            }
        }
        out_scores[k] = s;
    }
}

// Score intervals by MEAN of |coef| over the interval's intersection
// with the union U (libPLS's `Binterval_star(ii)=sum(B(IA))/length(B(IA))`).
void score_intervals_mean(
    const std::vector<std::int32_t>& V,
    const std::vector<std::vector<std::int64_t>>& intervals,
    const std::vector<std::int64_t>& U,
    const std::vector<double>& abs_coefs,
    std::vector<double>& out_scores) {
    out_scores.assign(V.size(), 0.0);
    std::int64_t max_feature = -1;
    for (std::int64_t f : U) {
        if (f > max_feature) max_feature = f;
    }
    if (max_feature < 0) return;
    std::vector<std::int64_t> lookup(
        static_cast<std::size_t>(max_feature + 1), -1);
    for (std::size_t idx_u = 0; idx_u < U.size(); ++idx_u) {
        lookup[static_cast<std::size_t>(U[idx_u])] =
            static_cast<std::int64_t>(idx_u);
    }
    for (std::size_t k = 0; k < V.size(); ++k) {
        double s = 0.0;
        std::int64_t count = 0;
        for (std::int64_t f : intervals[static_cast<std::size_t>(V[k])]) {
            if (f >= 0 && f <= max_feature) {
                const std::int64_t pos =
                    lookup[static_cast<std::size_t>(f)];
                if (pos >= 0) {
                    s += abs_coefs[static_cast<std::size_t>(pos)];
                    ++count;
                }
            }
        }
        out_scores[k] = count > 0 ? s / static_cast<double>(count) : 0.0;
    }
}

// Generate a candidate subset Vstar following the libPLS strategy
// faithfully:
//   d = nV1 - |V0|
//   d == 0 → return V0 unchanged.
//   d  < 0 → drop the |d| lowest-scoring V0 intervals (sort by V0 scores
//             descending, keep top nV1).
//   d  > 0 → pick min(3d, kvar) candidates UNIFORMLY AT RANDOM from the
//             unselected pool (libPLS: `randperm(kvar)(1:min(3d,kvar))`),
//             refit PLS on the union of V0 ∪ picks, then re-score every
//             interval in V0 ∪ picks by the **mean** of |coef| over its
//             intersection with the new union, and keep the top nV1 by
//             that score.
// Returns the new subset Vstar.
n4m_status_t generate_new_model(
    ::n4m::core::Context& ctx,
    const ::n4m::core::Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::vector<std::vector<std::int64_t>>& intervals,
    const std::vector<std::int32_t>& V0,
    std::int32_t nV1,
    const std::vector<double>& binterval_scores_v0,
    std::int32_t p,
    std::uint64_t& rng_state,
    std::vector<std::int32_t>& out_Vstar) {
    const std::int32_t nV0 = static_cast<std::int32_t>(V0.size());
    const std::int32_t d = nV1 - nV0;
    if (d == 0) {
        out_Vstar = V0;
        return N4M_OK;
    }

    if (d < 0) {
        // Drop lowest |d| scores.
        const std::size_t keep = static_cast<std::size_t>(nV1);
        std::vector<std::size_t> order(V0.size(), 0);
        std::iota(order.begin(), order.end(), static_cast<std::size_t>(0));
        std::stable_sort(order.begin(), order.end(),
                         [&binterval_scores_v0](std::size_t a, std::size_t b) {
                             const double la = binterval_scores_v0[a];
                             const double lb = binterval_scores_v0[b];
                             if (la == lb) return a < b;
                             return la > lb;
                         });
        out_Vstar.clear();
        out_Vstar.reserve(keep);
        for (std::size_t i = 0; i < keep && i < order.size(); ++i) {
            out_Vstar.push_back(V0[order[i]]);
        }
        std::sort(out_Vstar.begin(), out_Vstar.end());
        return N4M_OK;
    }

    // d > 0: random candidates via Fisher-Yates from the unselected
    // pool (matches libPLS `randperm(kvar)`).
    std::vector<unsigned char> in_v(static_cast<std::size_t>(p), 0U);
    for (std::int32_t iv : V0) in_v[static_cast<std::size_t>(iv)] = 1U;
    std::vector<std::int32_t> candidates;
    candidates.reserve(static_cast<std::size_t>(p));
    for (std::int32_t iv = 0; iv < p; ++iv) {
        if (in_v[static_cast<std::size_t>(iv)] == 0U) {
            candidates.push_back(iv);
        }
    }
    const std::size_t kvar = candidates.size();
    if (kvar == 0) {
        out_Vstar = V0;
        std::sort(out_Vstar.begin(), out_Vstar.end());
        return N4M_OK;
    }
    const std::size_t pick = std::min(static_cast<std::size_t>(3 * d), kvar);
    // Fisher-Yates partial shuffle on `candidates`.
    for (std::size_t i = 0; i < pick && i < kvar - 1; ++i) {
        const double r = uniform01(rng_state);
        const std::size_t j = i + static_cast<std::size_t>(
            r * static_cast<double>(kvar - i));
        const std::size_t jj = std::min(j, kvar - 1);
        std::swap(candidates[i], candidates[jj]);
    }
    std::vector<std::int32_t> combined = V0;
    combined.insert(combined.end(), candidates.begin(),
                    candidates.begin() +
                    static_cast<std::ptrdiff_t>(pick));

    // Refit PLS on the union of intervals in `combined`, then re-score
    // every entry of `combined` by the mean of |coef| over its
    // intersection with the new union.
    std::vector<std::int64_t> U_star = union_features(combined, intervals);
    std::vector<double> subset_x;
    n4m_status_t status = copy_columns(ctx, X, U_star, subset_x);
    if (status != N4M_OK) return status;
    n4m_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows,
                          static_cast<std::int64_t>(U_star.size()));
    std::vector<double> abs_coefs_star;
    status = fit_pls_coefs(ctx, cfg, subset_view, Y, abs_coefs_star);
    if (status != N4M_OK) return status;
    std::vector<double> combined_scores;
    score_intervals_mean(combined, intervals, U_star, abs_coefs_star,
                         combined_scores);

    std::vector<std::size_t> order(combined.size(), 0);
    std::iota(order.begin(), order.end(), static_cast<std::size_t>(0));
    std::stable_sort(order.begin(), order.end(),
                     [&combined_scores](std::size_t a, std::size_t b) {
                         const double la = combined_scores[a];
                         const double lb = combined_scores[b];
                         if (la == lb) return a < b;
                         return la > lb;
                     });
    const std::size_t keep_n =
        std::min(static_cast<std::size_t>(nV1), combined.size());
    out_Vstar.clear();
    out_Vstar.reserve(keep_n);
    for (std::size_t i = 0; i < keep_n; ++i) {
        out_Vstar.push_back(combined[order[i]]);
    }
    std::sort(out_Vstar.begin(), out_Vstar.end());
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_irf(Context& ctx,
                           const Config& cfg,
                           const n4m_matrix_view_t& X,
                           const n4m_matrix_view_t& Y,
                           const ValidationPlan& plan,
                           std::int32_t n_iterations,
                           std::int32_t window_size,
                           std::int32_t initial_intervals,
                           std::int32_t top_k,
                           std::uint64_t seed,
                           IrfSelectionResult& out) {
    try {
        out = IrfSelectionResult{};
        n4m_status_t status = validate_float_view(ctx, X, "X");
        if (status != N4M_OK) return status;
        status = validate_float_view(ctx, Y, "Y");
        if (status != N4M_OK) return status;
        if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
            ctx.set_error("IRF matrices must be non-empty");
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
            ctx.set_error("IRF requires at least one validation fold");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (n_iterations <= 0) {
            ctx.set_errorf("n_iterations must be >= 1; got %d",
                           static_cast<int>(n_iterations));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (window_size <= 0 || window_size > X.cols) {
            ctx.set_errorf("window_size must be in [1, %lld]; got %d",
                           static_cast<long long>(X.cols),
                           static_cast<int>(window_size));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("cfg.n_components must be >= 1");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        const std::int32_t n_intervals =
            static_cast<std::int32_t>(X.cols) - window_size + 1;
        if (initial_intervals < 1 || initial_intervals > n_intervals) {
            ctx.set_errorf("initial_intervals must be in [1, %d]; got %d",
                           static_cast<int>(n_intervals),
                           static_cast<int>(initial_intervals));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (top_k <= 0 || top_k > n_intervals) {
            ctx.set_errorf("top_k must be in [1, %d]; got %d",
                           static_cast<int>(n_intervals),
                           static_cast<int>(top_k));
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<std::vector<std::int64_t>> intervals(
            static_cast<std::size_t>(n_intervals));
        for (std::int32_t k = 0; k < n_intervals; ++k) {
            intervals[static_cast<std::size_t>(k)].reserve(
                static_cast<std::size_t>(window_size));
            for (std::int32_t off = 0; off < window_size; ++off) {
                intervals[static_cast<std::size_t>(k)].push_back(
                    static_cast<std::int64_t>(k + off));
            }
        }

        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_iterations = n_iterations;
        out.window_size = window_size;
        out.initial_intervals = initial_intervals;
        out.n_intervals = n_intervals;
        out.top_k = top_k;
        out.seed = seed;

        // Initial V0: take initial_intervals random intervals.
        std::uint64_t state =
            seed == 0 ? UINT64_C(0xBF58476D1CE4E5B9) : seed;
        std::vector<std::int32_t> perm(static_cast<std::size_t>(n_intervals));
        std::iota(perm.begin(), perm.end(), 0);
        for (std::int32_t i = n_intervals - 1; i > 0; --i) {
            const std::size_t k = random_bounded(state,
                                                  static_cast<std::size_t>(i + 1));
            std::swap(perm[static_cast<std::size_t>(i)], perm[k]);
        }
        std::vector<std::int32_t> V0(
            perm.begin(),
            perm.begin() + static_cast<std::ptrdiff_t>(initial_intervals));
        std::sort(V0.begin(), V0.end());

        std::vector<double> probability(
            static_cast<std::size_t>(n_intervals), 0.0);
        out.rmse_by_iteration.reserve(
            static_cast<std::size_t>(n_iterations));
        out.subset_sizes.reserve(
            static_cast<std::size_t>(n_iterations));

        // Pre-compute the initial union RMSE.
        std::vector<std::int64_t> U0 = union_features(V0, intervals);
        double rmse_current = 0.0;
        status = subset_rmse(ctx, cfg, X, Y, plan, U0, rmse_current);
        if (status != N4M_OK) {
            out = IrfSelectionResult{};
            return status;
        }
        std::int32_t current_size = static_cast<std::int32_t>(V0.size());

        double best_rmse = rmse_current;
        std::vector<std::int32_t> best_V = V0;

        for (std::int32_t it = 0; it < n_iterations; ++it) {
            // 1) Score the current intervals via a single PLS fit on the
            //    union U0.
            std::vector<double> subset_x;
            status = copy_columns(ctx, X, U0, subset_x);
            if (status != N4M_OK) {
                out = IrfSelectionResult{};
                return status;
            }
            n4m_matrix_view_t subset_view = rowmajor_f64_view(
                subset_x, X.rows, static_cast<std::int64_t>(U0.size()));
            std::vector<double> abs_coefs;
            status = fit_pls_coefs(ctx, cfg, subset_view, Y, abs_coefs);
            if (status != N4M_OK) {
                out = IrfSelectionResult{};
                return status;
            }
            std::vector<double> binterval_scores;
            score_intervals(V0, intervals, U0, abs_coefs, binterval_scores);

            // 2) Propose Vstar.
            // libPLS uses `nVstar = round(randn() * 0.3 * Q + Q)` where
            // `Q = length(V0)` at each iteration (dynamic, not the
            // initial size — codex catch).
            const double gauss = standard_normal(state);
            std::int32_t nVstar = static_cast<std::int32_t>(
                std::lround(gauss * 0.3 * static_cast<double>(current_size) +
                            static_cast<double>(current_size)));
            nVstar = std::max(2, nVstar);
            nVstar = std::min(n_intervals, nVstar);
            std::vector<std::int32_t> Vstar;
            status = generate_new_model(
                ctx, cfg, X, Y, intervals, V0, nVstar, binterval_scores,
                n_intervals, state, Vstar);
            if (status != N4M_OK) {
                out = IrfSelectionResult{};
                return status;
            }
            if (Vstar.empty()) Vstar = V0;

            // 3) Compute RMSEs for V0 and Vstar (V0 already computed).
            std::vector<std::int64_t> Ustar = union_features(Vstar, intervals);
            double rmse_star = 0.0;
            if (Ustar == U0) {
                rmse_star = rmse_current;
            } else {
                status = subset_rmse(ctx, cfg, X, Y, plan, Ustar, rmse_star);
                if (status != N4M_OK) {
                    out = IrfSelectionResult{};
                    return status;
                }
            }

            // 4) Accept/reject.
            double accept_prob = 0.0;
            if (rmse_star < rmse_current) {
                accept_prob = 1.0;
            } else if (rmse_star > 0.0) {
                accept_prob = 0.1 * rmse_current / rmse_star;
            }
            const double r = uniform01(state);
            if (r < accept_prob) {
                V0 = std::move(Vstar);
                U0 = std::move(Ustar);
                rmse_current = rmse_star;
                current_size = static_cast<std::int32_t>(V0.size());
            }

            // 5) Update statistics.
            for (std::int32_t iv : V0) {
                probability[static_cast<std::size_t>(iv)] += 1.0;
            }
            out.rmse_by_iteration.push_back(rmse_current);
            out.subset_sizes.push_back(static_cast<std::int64_t>(V0.size()));
            if (rmse_current < best_rmse) {
                best_rmse = rmse_current;
                best_V = V0;
            }
        }

        const double inv_n = 1.0 / static_cast<double>(n_iterations);
        out.probability.assign(static_cast<std::size_t>(n_intervals), 0.0);
        for (std::int32_t k = 0; k < n_intervals; ++k) {
            out.probability[static_cast<std::size_t>(k)] =
                probability[static_cast<std::size_t>(k)] * inv_n;
        }

        // Rank intervals by inclusion probability (descending), break
        // ties by index.
        std::vector<std::int64_t> ranking(
            static_cast<std::size_t>(n_intervals), 0);
        std::iota(ranking.begin(), ranking.end(),
                  static_cast<std::int64_t>(0));
        std::stable_sort(ranking.begin(), ranking.end(),
                         [&out](std::int64_t lhs, std::int64_t rhs) {
                             const double l = out.probability[
                                 static_cast<std::size_t>(lhs)];
                             const double r = out.probability[
                                 static_cast<std::size_t>(rhs)];
                             if (l == r) return lhs < rhs;
                             return l > r;
                         });
        out.top_k_intervals.assign(
            ranking.begin(),
            ranking.begin() + static_cast<std::ptrdiff_t>(top_k));

        // Build the union of the top-K intervals as the final selection.
        std::vector<std::int32_t> top_V(static_cast<std::size_t>(top_k));
        for (std::int32_t k = 0; k < top_k; ++k) {
            top_V[static_cast<std::size_t>(k)] =
                static_cast<std::int32_t>(ranking[static_cast<std::size_t>(k)]);
        }
        out.selected_indices = union_features(top_V, intervals);
        out.best_rmse = best_rmse;
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running IRF selection");
        out = IrfSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running IRF selection");
        out = IrfSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
