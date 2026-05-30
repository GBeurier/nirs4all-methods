// SPDX-License-Identifier: CECILL-2.1

#include "core/uve_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/stability_selection.hpp"
#include "core/common/status.hpp"
#include "core/model.hpp"
#include "core/common/rng_mt_r.h"

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
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            const ::n4m::core::ValidationPlan& plan,
                                            std::int32_t noise_features) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("UVE matrices must be non-empty");
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
    if (plan.folds.size() < 2U) {
        ctx.set_error("UVE requires at least two Monte-Carlo folds");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (noise_features <= 0) {
        ctx.set_errorf("noise_features must be >= 1; got %d",
                       static_cast<int>(noise_features));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("UVE matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > std::numeric_limits<std::int64_t>::max() -
                     static_cast<std::int64_t>(noise_features)) {
        ctx.set_error("UVE augmented feature count is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t total_features = X.cols + static_cast<std::int64_t>(noise_features);
    if (total_features > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("UVE augmented feature count exceeds int32 storage");
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

[[nodiscard]] n4m_status_t build_augmented_matrix(::n4m::core::Context& ctx,
                                                  const n4m_matrix_view_t& X,
                                                  std::int32_t noise_features,
                                                  std::uint64_t noise_seed,
                                                  std::vector<double>& out) {
    const std::int64_t total_cols = X.cols + static_cast<std::int64_t>(noise_features);
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, total_cols, n_values)) {
        ctx.set_error("UVE augmented matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
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
    return N4M_OK;
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

// ---------------------------------------------------------------------------
// R-exact MCUVE path (opt-in via cfg.rng_kind == N4M_RNG_MT_R).
//
// Reproduces R `plsVarSel::mcuve_pls` for-the-selected-set:
//   W   <- matrix(runif(Nx*Mx,0,1), Mx, Nx)   (R-MT, column-major, NOT scaled)
//   Z   <- cbind(X, W);  K <- floor(Mx*ratio)  (== first MC fold's train size)
//   for it in 1..N: calk <- sample(Mx)[1:K]
//                   opt  <- which.min(LOO-PRESS over 1..ncomp comps)
//                   C[it,] <- coef(plsr(scale=FALSE), opt)
//   RI  <- colMeans(C)/colSd(C);  thr <- max|RI[noise]|
//   sel <- which(|RI[real]| > thr); too-few fallback -> top-ncomp |RI[real]|.
// Every primitive is bit-exact / parity-validated; proven Jaccard 1.0 vs R in
// parity/scripts/uve_r_exact_reference.py and the doctest. The default
// rng_kind (SPLITMIX64) keeps the historical path untouched — this is additive.
// ---------------------------------------------------------------------------

[[nodiscard]] ::n4m::core::Config r_exact_pls_cfg(const ::n4m::core::Config& base,
                                                  std::int32_t k) noexcept {
    ::n4m::core::Config cfg = base;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_NIPALS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = k;
    cfg.center_x = 1;
    cfg.center_y = 1;
    cfg.scale_x = 0;   // R plsr(scale = FALSE)
    cfg.scale_y = 0;
    return cfg;
}

[[nodiscard]] n4m_status_t r_exact_fit_coef(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& base,
                                            const std::vector<double>& Zrows,
                                            std::size_t n_rows,
                                            std::size_t n_cols,
                                            const std::vector<double>& yrows,
                                            std::int32_t k,
                                            std::vector<double>& out_coef) {
    ::n4m::core::Config cfg = r_exact_pls_cfg(base, k);
    auto Zv = const_cast<std::vector<double>&>(Zrows);
    auto Yv = const_cast<std::vector<double>&>(yrows);
    n4m_matrix_view_t Xview = rowmajor_f64_view(Zv, static_cast<std::int64_t>(n_rows),
                                                static_cast<std::int64_t>(n_cols));
    n4m_matrix_view_t Yview = rowmajor_f64_view(Yv, static_cast<std::int64_t>(n_rows), 1);
    std::unique_ptr<::n4m::core::Model> model;
    const n4m_status_t st = ::n4m::core::fit_model(ctx, cfg, Xview, Yview, model);
    if (st != N4M_OK) return st;
    if (!model || model->coefficients.size() != n_cols) return N4M_ERR_INTERNAL;
    out_coef = model->coefficients;   // n_cols x 1
    return N4M_OK;
}

[[nodiscard]] double r_exact_loo_predict(::n4m::core::Context& ctx,
                                         const ::n4m::core::Config& base,
                                         const std::vector<double>& Ztrain,
                                         std::size_t train_rows,
                                         std::size_t n_cols,
                                         const std::vector<double>& ytrain,
                                         std::int32_t k,
                                         const double* xrow,
                                         bool& ok) {
    ::n4m::core::Config cfg = r_exact_pls_cfg(base, k);
    auto Zv = const_cast<std::vector<double>&>(Ztrain);
    auto Yv = const_cast<std::vector<double>&>(ytrain);
    n4m_matrix_view_t Xview = rowmajor_f64_view(Zv, static_cast<std::int64_t>(train_rows),
                                                static_cast<std::int64_t>(n_cols));
    n4m_matrix_view_t Yview = rowmajor_f64_view(Yv, static_cast<std::int64_t>(train_rows), 1);
    std::unique_ptr<::n4m::core::Model> model;
    if (::n4m::core::fit_model(ctx, cfg, Xview, Yview, model) != N4M_OK || !model) {
        ok = false;
        return 0.0;
    }
    std::vector<double> xbuf(xrow, xrow + n_cols);
    std::vector<double> pred(1, 0.0);
    n4m_matrix_view_t xv = rowmajor_f64_view(xbuf, 1, static_cast<std::int64_t>(n_cols));
    n4m_matrix_view_t pv = rowmajor_f64_view(pred, 1, 1);
    if (::n4m::core::predict_into(ctx, *model, xv, pv) != N4M_OK) {
        ok = false;
        return 0.0;
    }
    ok = true;
    return pred[0];
}

[[nodiscard]] n4m_status_t select_by_uve_r_exact(::n4m::core::Context& ctx,
                                                 const ::n4m::core::Config& cfg,
                                                 const n4m_matrix_view_t& X,
                                                 const n4m_matrix_view_t& Y,
                                                 const ::n4m::core::ValidationPlan& plan,
                                                 std::uint64_t seed,
                                                 ::n4m::core::UveSelectionResult& out) {
    const auto Mx = static_cast<std::size_t>(X.rows);
    const auto Nx = static_cast<std::size_t>(X.cols);
    const std::size_t total = 2U * Nx;                       // [X | W]
    const std::size_t N = plan.folds.size();                // MC iterations (R's N)
    const std::size_t K = std::min(plan.folds[0].train_indices.size(), Mx);  // floor(Mx*ratio)

    std::int32_t ncomp = cfg.n_components;                   // R clamps to min(Mx, Nx, ncomp)
    {
        const std::int32_t cap = static_cast<std::int32_t>(std::min(Mx, Nx));
        if (ncomp > cap) ncomp = cap;
        if (ncomp < 1) ncomp = 1;
    }

    std::vector<double> y(Mx, 0.0);
    for (std::size_t i = 0; i < Mx; ++i) y[i] = read_value(Y, i, 0);

    // Z = [X | W]; W = Nx unscaled runif columns, R-MT, column-major fill.
    std::vector<double> Z(Mx * total, 0.0);
    for (std::size_t i = 0; i < Mx; ++i)
        for (std::size_t j = 0; j < Nx; ++j)
            Z[idx(i, total, j)] = read_value(X, i, j);
    n4m_rng_mt_r rng;
    n4m_rng_mt_r_set_seed(&rng, static_cast<std::uint32_t>(seed));
    for (std::size_t j = 0; j < Nx; ++j)                     // column-major (R matrix())
        for (std::size_t i = 0; i < Mx; ++i)
            Z[idx(i, total, Nx + j)] = n4m_rng_mt_r_unif(&rng);

    std::vector<double> C(N * total, 0.0);
    std::vector<int> perm(Mx, 0);
    for (std::size_t it = 0; it < N; ++it) {
        n4m_rng_mt_r_sample(&rng, static_cast<int>(Mx), perm.data());
        const std::size_t k_rows = K;
        std::vector<double> Zcal(k_rows * total, 0.0);
        std::vector<double> ycal(k_rows, 0.0);
        for (std::size_t r = 0; r < k_rows; ++r) {
            const auto src = static_cast<std::size_t>(perm[r]);
            for (std::size_t c = 0; c < total; ++c) Zcal[idx(r, total, c)] = Z[idx(src, total, c)];
            ycal[r] = y[src];
        }
        std::int32_t maxk = ncomp;
        if (maxk > static_cast<std::int32_t>(total) - 1) maxk = static_cast<std::int32_t>(total) - 1;
        if (maxk > static_cast<std::int32_t>(k_rows) - 1) maxk = static_cast<std::int32_t>(k_rows) - 1;
        if (maxk < 1) maxk = 1;

        std::int32_t opt = 1;
        double best_press = std::numeric_limits<double>::infinity();
        for (std::int32_t k = 1; k <= maxk; ++k) {
            double press = 0.0;
            bool ok = true;
            for (std::size_t hold = 0; hold < k_rows && ok; ++hold) {
                std::vector<double> Ztr((k_rows - 1) * total, 0.0);
                std::vector<double> ytr(k_rows - 1, 0.0);
                std::size_t w = 0;
                for (std::size_t r = 0; r < k_rows; ++r) {
                    if (r == hold) continue;
                    for (std::size_t c = 0; c < total; ++c)
                        Ztr[idx(w, total, c)] = Zcal[idx(r, total, c)];
                    ytr[w] = ycal[r];
                    ++w;
                }
                const double pred = r_exact_loo_predict(ctx, cfg, Ztr, k_rows - 1, total,
                                                        ytr, k, &Zcal[idx(hold, total, 0)], ok);
                const double e = ycal[hold] - pred;
                press += e * e;
            }
            if (ok && press < best_press) {
                best_press = press;
                opt = k;
            }
        }

        std::vector<double> coef;
        const n4m_status_t st = r_exact_fit_coef(ctx, cfg, Zcal, k_rows, total, ycal, opt, coef);
        if (st != N4M_OK) {
            out = ::n4m::core::UveSelectionResult{};
            return st;
        }
        for (std::size_t c = 0; c < total; ++c) C[it * total + c] = coef[c];
    }

    // RI = colMeans(C) / colSd(C, ddof = 1).
    std::vector<double> RI(total, 0.0);
    for (std::size_t c = 0; c < total; ++c) {
        double mean = 0.0;
        for (std::size_t it = 0; it < N; ++it) mean += C[it * total + c];
        mean /= static_cast<double>(N);
        double ss = 0.0;
        for (std::size_t it = 0; it < N; ++it) {
            const double d = C[it * total + c] - mean;
            ss += d * d;
        }
        const double sd = (N > 1U) ? std::sqrt(ss / static_cast<double>(N - 1U)) : 0.0;
        RI[c] = (sd > 0.0) ? mean / sd : 0.0;
    }

    double thr = 0.0;
    for (std::size_t j = Nx; j < total; ++j) thr = std::max(thr, std::fabs(RI[j]));

    out = ::n4m::core::UveSelectionResult{};
    out.real_stability_scores.assign(Nx, 0.0);
    out.noise_stability_scores.assign(Nx, 0.0);
    for (std::size_t j = 0; j < Nx; ++j) out.real_stability_scores[j] = std::fabs(RI[j]);
    for (std::size_t j = 0; j < Nx; ++j) out.noise_stability_scores[j] = std::fabs(RI[Nx + j]);
    out.noise_threshold = thr;

    std::vector<std::int64_t> sel;
    for (std::size_t j = 0; j < Nx; ++j)
        if (std::fabs(RI[j]) > thr) sel.push_back(static_cast<std::int64_t>(j));
    // R too-few fallback: take the top-ncomp |RI[real]| when the threshold rule
    // keeps no more than ncomp+1 features.
    if (sel.size() <= static_cast<std::size_t>(ncomp + 1)) {
        std::vector<std::int64_t> order = rank_descending(out.real_stability_scores);
        order.resize(static_cast<std::size_t>(ncomp));
        std::sort(order.begin(), order.end());
        sel = order;
    } else {
        std::sort(sel.begin(), sel.end());
    }

    out.selected_indices = sel;
    out.n_features = static_cast<std::int32_t>(Nx);
    out.n_targets = static_cast<std::int32_t>(Y.cols);
    out.n_repeats = static_cast<std::int32_t>(N);
    out.n_noise_features = static_cast<std::int32_t>(Nx);
    out.selected_count = static_cast<std::int32_t>(sel.size());
    out.noise_seed = seed;
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_uve(Context& ctx,
                           const Config& cfg,
                           const n4m_matrix_view_t& X,
                           const n4m_matrix_view_t& Y,
                           const ValidationPlan& plan,
                           std::int32_t noise_features,
                           std::uint64_t noise_seed,
                           UveSelectionResult& out) {
    try {
        out = UveSelectionResult{};
        n4m_status_t status = validate_request(ctx, X, Y, plan, noise_features);
        if (status != N4M_OK) {
            return status;
        }

        // Opt-in R-exact MCUVE path (additive): reproduces R
        // plsVarSel::mcuve_pls bit-for-set. The default rng_kind
        // (SPLITMIX64) falls through to the historical path below, unchanged.
        if (cfg.rng_kind == static_cast<std::int32_t>(N4M_RNG_MT_R)) {
            return select_by_uve_r_exact(ctx, cfg, X, Y, plan, noise_seed, out);
        }

        std::vector<double> augmented;
        status = build_augmented_matrix(ctx, X, noise_features, noise_seed, augmented);
        if (status != N4M_OK) {
            return status;
        }

        const std::int64_t total_features = X.cols + static_cast<std::int64_t>(noise_features);
        n4m_matrix_view_t augmented_view = rowmajor_f64_view(augmented, X.rows, total_features);
        StabilitySelectionResult stability;
        status = select_by_coefficient_stability(ctx,
                                                 cfg,
                                                 augmented_view,
                                                 Y,
                                                 plan,
                                                 static_cast<std::int32_t>(total_features),
                                                 stability);
        if (status != N4M_OK) {
            out = UveSelectionResult{};
            return status;
        }
        if (stability.stability_scores.size() != static_cast<std::size_t>(total_features)) {
            ctx.set_error("UVE stability score count does not match augmented feature count");
            out = UveSelectionResult{};
            return N4M_ERR_INTERNAL;
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
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running UVE selection");
        out = UveSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running UVE selection");
        out = UveSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
