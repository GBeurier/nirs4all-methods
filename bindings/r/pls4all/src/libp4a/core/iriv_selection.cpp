// SPDX-License-Identifier: CECILL-2.1
//
// IRIV — Iteratively Retains Informative Variables. Mirrors libPLS 1.95
// `iriv.m` minus the trailing backward-elimination pass (BVE is already
// available as a standalone selector via `select_by_bve`).

#include "core/iriv_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "core/cross_validation.hpp"
#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

constexpr std::uint64_t kSplitMixGolden = 0x9E3779B97F4A7C15ull;

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += kSplitMixGolden;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
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
        ctx.set_error("IRIV column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()),
                              n_values)) {
        ctx.set_error("IRIV subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("IRIV column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("IRIV X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[row * cols + local_col] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t mask_rmse(::n4m::core::Context& ctx,
                                     ::n4m::core::Config cfg,
                                     const n4m_matrix_view_t& X,
                                     const n4m_matrix_view_t& Y,
                                     const ::n4m::core::ValidationPlan& plan,
                                     const std::vector<unsigned char>& mask,
                                     double& out) {
    std::vector<std::int64_t> cols;
    cols.reserve(mask.size());
    for (std::size_t j = 0; j < mask.size(); ++j) {
        if (mask[j] != 0U) cols.push_back(static_cast<std::int64_t>(j));
    }
    if (cols.empty()) {
        out = std::numeric_limits<double>::infinity();
        return N4M_OK;
    }
    std::vector<double> subset_x;
    n4m_status_t status = copy_columns(ctx, X, cols, subset_x);
    if (status != N4M_OK) return status;
    n4m_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows, static_cast<std::int64_t>(cols.size()));
    // Cap n_components to the mask size.
    if (cfg.n_components > static_cast<std::int32_t>(cols.size())) {
        cfg.n_components = static_cast<std::int32_t>(cols.size());
    }
    if (cfg.n_components < 1) cfg.n_components = 1;
    ::n4m::core::CrossValidationResult cv;
    status = ::n4m::core::cross_validate_regression(ctx, cfg, subset_view, Y,
                                                        plan, cv);
    if (status != N4M_OK) return status;
    out = cv.metrics.rmse;
    return N4M_OK;
}

// Choose the row count for the binary matrix the way libPLS does.
[[nodiscard]] std::int32_t binary_matrix_rows(std::int32_t n_features) noexcept {
    if (n_features >= 500) return 500;
    if (n_features >= 300) return 300;
    if (n_features >= 100) return 200;
    if (n_features >= 50)  return 100;
    if (n_features >= 10)  return 50;
    return 20;
}

// Generate a (rows × cols) binary matrix where each column has ~rows/2
// ones (Fisher-Yates partition) AND each row has at least 2 ones (so we
// can always fit a 2-component PLS). Uses splitmix64.
void generate_binary_matrix(std::int32_t rows, std::int32_t cols,
                            std::uint64_t& state,
                            std::vector<unsigned char>& out) {
    out.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
               0U);
    bool ok = false;
    constexpr std::int32_t kMaxRetries = 8;
    const auto urows = static_cast<std::size_t>(rows);
    const auto ucols = static_cast<std::size_t>(cols);
    const std::int32_t half = rows / 2;
    std::vector<std::int32_t> perm(urows, 0);
    for (std::int32_t attempt = 0; attempt < kMaxRetries && !ok; ++attempt) {
        for (std::size_t j = 0; j < ucols; ++j) {
            for (std::size_t r = 0; r < urows; ++r) {
                perm[r] = static_cast<std::int32_t>(r);
            }
            // Fisher-Yates shuffle.
            for (std::size_t r = urows; r > 1; --r) {
                const std::uint64_t bits = splitmix64_next(state);
                const std::size_t k =
                    static_cast<std::size_t>(bits %
                        static_cast<std::uint64_t>(r));
                std::swap(perm[r - 1U], perm[k]);
            }
            // First `half` entries get 0, the rest get 1 (matches the
            // libPLS rule `rand_row<=length/2 -> 0`).
            for (std::size_t r = 0; r < urows; ++r) {
                const std::int32_t rank = perm[r];
                out[r * ucols + j] = (rank < half) ? 0U : 1U;
            }
        }
        // Validate: each row must have >= 2 ones.
        ok = true;
        for (std::size_t r = 0; r < urows && ok; ++r) {
            std::int32_t row_sum = 0;
            for (std::size_t j = 0; j < ucols; ++j) {
                row_sum += out[r * ucols + j];
            }
            if (row_sum < 2) ok = false;
        }
    }
    if (!ok) {
        // Last-resort: force the first two columns of every row to 1.
        const std::size_t fill_cols =
            std::min<std::size_t>(static_cast<std::size_t>(2), ucols);
        for (std::size_t r = 0; r < urows; ++r) {
            for (std::size_t j = 0; j < fill_cols; ++j) {
                out[r * ucols + j] = 1U;
            }
        }
    }
}

// Mann-Whitney U test (two-tailed). Returns the asymptotic p-value via
// normal approximation. The libPLS code calls MATLAB's `ranksum(...,
// 'alpha', 0.05)`; the asymptotic formula matches MATLAB's default for
// n + m >= 20 and stays a perfectly defensible approximation for smaller
// samples (the only thing the IRIV loop reads is the p < 0.05 decision).
double mann_whitney_u_pvalue(const std::vector<double>& a,
                              const std::vector<double>& b) {
    const std::size_t na = a.size();
    const std::size_t nb = b.size();
    if (na == 0 || nb == 0) return 1.0;
    std::vector<std::pair<double, int>> all;
    all.reserve(na + nb);
    for (double v : a) all.emplace_back(v, 0);
    for (double v : b) all.emplace_back(v, 1);
    std::stable_sort(all.begin(), all.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                     });

    // Assign average ranks for ties.
    std::vector<double> ranks(all.size(), 0.0);
    std::size_t i = 0;
    double tie_correction = 0.0;
    while (i < all.size()) {
        std::size_t j = i + 1;
        while (j < all.size() && all[j].first == all[i].first) ++j;
        const std::size_t tie_n = j - i;
        const double avg_rank =
            (static_cast<double>(i + 1) + static_cast<double>(j)) * 0.5;
        for (std::size_t k = i; k < j; ++k) ranks[k] = avg_rank;
        if (tie_n > 1) {
            const double t = static_cast<double>(tie_n);
            tie_correction += t * t * t - t;
        }
        i = j;
    }
    double r1 = 0.0;
    for (std::size_t k = 0; k < all.size(); ++k) {
        if (all[k].second == 0) r1 += ranks[k];
    }
    const double n1 = static_cast<double>(na);
    const double n2 = static_cast<double>(nb);
    const double u1 = r1 - n1 * (n1 + 1.0) * 0.5;
    const double u2 = n1 * n2 - u1;
    const double u = std::min(u1, u2);

    const double n = n1 + n2;
    if (n < 2.0) return 1.0;
    const double mean_u = n1 * n2 * 0.5;
    const double tie_term = tie_correction / (n * (n - 1.0));
    const double var_u = n1 * n2 / 12.0 * ((n + 1.0) - tie_term);
    if (var_u <= 0.0) return 1.0;
    const double z = (u - mean_u + 0.5) / std::sqrt(var_u);  // continuity correction
    // Two-tailed normal CDF.
    const double abs_z = std::fabs(z);
    // erfc(x / sqrt(2)) gives 2 * (1 - Phi(x)).
    const double p = std::erfc(abs_z / std::sqrt(2.0));
    return std::max(0.0, std::min(1.0, p));
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_iriv(Context& ctx,
                            const Config& cfg,
                            const n4m_matrix_view_t& X,
                            const n4m_matrix_view_t& Y,
                            const ValidationPlan& plan,
                            std::int32_t max_rounds,
                            std::uint64_t seed,
                            IrivSelectionResult& out) {
    try {
        out = IrivSelectionResult{};
        n4m_status_t status = validate_float_view(ctx, X, "X");
        if (status != N4M_OK) return status;
        status = validate_float_view(ctx, Y, "Y");
        if (status != N4M_OK) return status;
        if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
            ctx.set_error("IRIV matrices must be non-empty");
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
            ctx.set_error("IRIV requires at least one validation fold");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1 ||
            static_cast<std::int64_t>(cfg.n_components) > X.cols) {
            ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                           static_cast<long long>(X.cols),
                           static_cast<int>(cfg.n_components));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (max_rounds <= 0) {
            ctx.set_errorf("max_rounds must be >= 1; got %d",
                           static_cast<int>(max_rounds));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("IRIV matrix dimensions exceed int32 result storage");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        const auto p = static_cast<std::int32_t>(X.cols);
        std::vector<std::int64_t> active(static_cast<std::size_t>(p), 0);
        for (std::int32_t j = 0; j < p; ++j) {
            active[static_cast<std::size_t>(j)] = j;
        }

        out.n_features = p;
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.seed = seed;
        out.binary_matrix_rows = binary_matrix_rows(p);
        out.remaining_per_round.push_back(static_cast<std::int64_t>(p));

        std::uint64_t state = seed == 0 ? UINT64_C(0xC2B2AE3D27D4EB4F) : seed;

        for (std::int32_t round = 0; round < max_rounds; ++round) {
            const auto nx = static_cast<std::int32_t>(active.size());
            if (nx < 2) break;
            const std::int32_t rows = binary_matrix_rows(nx);
            std::vector<unsigned char> bmat;
            generate_binary_matrix(rows, nx, state, bmat);

            // RMSECV_origin: one CV run per row.
            std::vector<double> rmse_origin(static_cast<std::size_t>(rows), 0.0);
            for (std::int32_t r = 0; r < rows; ++r) {
                // Build the full (mask over the original-X column space).
                std::vector<unsigned char> mask(static_cast<std::size_t>(p), 0U);
                for (std::int32_t local = 0; local < nx; ++local) {
                    if (bmat[static_cast<std::size_t>(r) *
                              static_cast<std::size_t>(nx) +
                              static_cast<std::size_t>(local)] != 0U) {
                        mask[static_cast<std::size_t>(active[
                            static_cast<std::size_t>(local)])] = 1U;
                    }
                }
                double rmse = 0.0;
                n4m_status_t s = mask_rmse(ctx, cfg, X, Y, plan, mask, rmse);
                if (s != N4M_OK) {
                    out = IrivSelectionResult{};
                    return s;
                }
                rmse_origin[static_cast<std::size_t>(r)] = rmse;
            }

            // For each active variable j: flip its bit on every row, run
            // CV. Cost is O(nx × rows × CV) — heavy but matches libPLS.
            std::vector<double> rmse_replace(
                static_cast<std::size_t>(rows) *
                static_cast<std::size_t>(nx), 0.0);
            for (std::int32_t j = 0; j < nx; ++j) {
                const auto orig_col = active[static_cast<std::size_t>(j)];
                for (std::int32_t r = 0; r < rows; ++r) {
                    std::vector<unsigned char> mask(static_cast<std::size_t>(p), 0U);
                    for (std::int32_t local = 0; local < nx; ++local) {
                        if (bmat[static_cast<std::size_t>(r) *
                                  static_cast<std::size_t>(nx) +
                                  static_cast<std::size_t>(local)] != 0U) {
                            mask[static_cast<std::size_t>(active[
                                static_cast<std::size_t>(local)])] = 1U;
                        }
                    }
                    // Flip j on this row.
                    const auto orig_bit =
                        bmat[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(nx) +
                             static_cast<std::size_t>(j)];
                    if (orig_bit != 0U) {
                        mask[static_cast<std::size_t>(orig_col)] = 0U;
                    } else {
                        mask[static_cast<std::size_t>(orig_col)] = 1U;
                    }
                    double rmse = 0.0;
                    n4m_status_t s = mask_rmse(ctx, cfg, X, Y, plan, mask, rmse);
                    if (s != N4M_OK) {
                        out = IrivSelectionResult{};
                        return s;
                    }
                    rmse_replace[static_cast<std::size_t>(r) *
                                  static_cast<std::size_t>(nx) +
                                  static_cast<std::size_t>(j)] = rmse;
                }
            }

            // Per-variable Mann-Whitney U + DMEAN-based P-value adjustment.
            std::vector<std::int64_t> to_remove;
            for (std::int32_t j = 0; j < nx; ++j) {
                std::vector<double> exclude;
                std::vector<double> include;
                exclude.reserve(static_cast<std::size_t>(rows));
                include.reserve(static_cast<std::size_t>(rows));
                // libPLS:
                //   RMSECV_exclude = RMSECV_replace; exclude[bmat==0] = origin;
                //   RMSECV_include = RMSECV_replace; include[bmat==1] = origin;
                for (std::int32_t r = 0; r < rows; ++r) {
                    const auto bit =
                        bmat[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(nx) +
                             static_cast<std::size_t>(j)];
                    const double rep = rmse_replace[
                        static_cast<std::size_t>(r) *
                        static_cast<std::size_t>(nx) +
                        static_cast<std::size_t>(j)];
                    const double origin = rmse_origin[
                        static_cast<std::size_t>(r)];
                    if (bit == 0U) {
                        // origin: variable NOT included; replace: it IS.
                        exclude.push_back(origin);
                        include.push_back(rep);
                    } else {
                        // origin: variable IS included; replace: it is NOT.
                        exclude.push_back(rep);
                        include.push_back(origin);
                    }
                }
                double p_value = mann_whitney_u_pvalue(exclude, include);
                double mean_excl = 0.0;
                double mean_incl = 0.0;
                for (double v : exclude) mean_excl += v;
                for (double v : include) mean_incl += v;
                mean_excl /= static_cast<double>(exclude.size());
                mean_incl /= static_cast<double>(include.size());
                const double dmean = mean_excl - mean_incl;
                if (dmean < 0.0) p_value += 1.0;
                if (p_value >= 1.0) {
                    to_remove.push_back(active[static_cast<std::size_t>(j)]);
                }
            }

            out.removed_per_round.push_back(
                static_cast<std::int64_t>(to_remove.size()));
            if (to_remove.empty()) {
                out.n_rounds = round + 1;
                out.remaining_per_round.push_back(
                    static_cast<std::int64_t>(active.size()));
                break;
            }
            // Drop the to-remove variables from active (sorted union diff).
            std::vector<unsigned char> drop_mask(static_cast<std::size_t>(p), 0U);
            for (std::int64_t v : to_remove) {
                drop_mask[static_cast<std::size_t>(v)] = 1U;
            }
            std::vector<std::int64_t> next_active;
            next_active.reserve(active.size() - to_remove.size());
            for (std::int64_t v : active) {
                if (drop_mask[static_cast<std::size_t>(v)] == 0U) {
                    next_active.push_back(v);
                }
            }
            active = std::move(next_active);
            out.remaining_per_round.push_back(
                static_cast<std::int64_t>(active.size()));
            out.n_rounds = round + 1;
            if (active.size() < 2) break;
        }

        out.selected_indices = active;
        std::sort(out.selected_indices.begin(), out.selected_indices.end());
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running IRIV selection");
        out = IrivSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running IRIV selection");
        out = IrivSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
