// SPDX-License-Identifier: CECILL-2.1
//
// Native AOM robust-HPO screening over strict-linear preprocessing chains.

#include "core/aom_robust_hpo.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/aom_operators.hpp"
#include "core/model.hpp"
#include "core/operator_entry.hpp"
#include "core/ridge.hpp"

namespace n4m::core {

namespace {

struct ChainSpec {
    std::int32_t id{0};
    const char* name{nullptr};
    std::vector<OperatorEntry> ops;
};

struct Fold {
    std::vector<std::int64_t> train;
    std::vector<std::int64_t> valid;
};

struct FitResult {
    std::vector<double> coefficients;
    double intercept{0.0};
    std::vector<double> predictions;
};

struct Candidate {
    std::int32_t chain_id{0};
    AomRobustHpoHead head{AomRobustHpoHead::kRidge};
    double param{0.0};
};

[[nodiscard]] n4m_matrix_view_t rowmajor_view(double* data,
                                              std::int64_t rows,
                                              std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = data;
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    view.reserved0 = 0;
    return view;
}

[[nodiscard]] n4m_matrix_view_t rowmajor_view(std::vector<double>& data,
                                              std::int64_t rows,
                                              std::int64_t cols) noexcept {
    return rowmajor_view(data.empty() ? nullptr : data.data(), rows, cols);
}

OperatorEntry op(n4m_operator_kind_t kind,
                 std::initializer_list<double> params = {}) {
    const std::vector<double> owned(params);
    return OperatorEntry(kind,
                         owned.empty() ? nullptr : owned.data(),
                         static_cast<std::int32_t>(owned.size()));
}

void add_chain(std::vector<ChainSpec>& chains,
               const char* name,
               std::vector<OperatorEntry> ops) {
    ChainSpec chain{};
    chain.id = static_cast<std::int32_t>(chains.size());
    chain.name = name;
    chain.ops = std::move(ops);
    chains.push_back(std::move(chain));
}

std::vector<ChainSpec> build_bank(std::int32_t profile) {
    std::vector<ChainSpec> chains;
    add_chain(chains, "raw", {op(N4M_OP_IDENTITY)});
    add_chain(chains, "detrend1", {op(N4M_OP_DETREND_POLY, {1})});
    add_chain(chains, "detrend2", {op(N4M_OP_DETREND_POLY, {2})});
    add_chain(chains, "savgol_w5_p2_d0",
              {op(N4M_OP_SAVGOL_SMOOTH, {5, 2})});
    add_chain(chains, "savgol_w7_p2_d0",
              {op(N4M_OP_SAVGOL_SMOOTH, {7, 2})});
    add_chain(chains, "savgol_w7_p2_d1",
              {op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    add_chain(chains, "savgol_w11_p2_d2",
              {op(N4M_OP_SAVGOL_DERIVATIVE, {11, 2, 2})});
    add_chain(chains, "nw_s5_g5_d1",
              {op(N4M_OP_NORRIS_WILLIAMS, {5, 5, 1})});
    add_chain(chains, "finite_diff1",
              {op(N4M_OP_FINITE_DIFFERENCE, {1})});
    add_chain(chains, "detrend1_savgol_w7_p2_d1",
              {op(N4M_OP_DETREND_POLY, {1}),
               op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    add_chain(chains, "detrend1_nw_s5_g5_d1",
              {op(N4M_OP_DETREND_POLY, {1}),
               op(N4M_OP_NORRIS_WILLIAMS, {5, 5, 1})});
    add_chain(chains, "savgol_w5_p2_d0_finite_diff1",
              {op(N4M_OP_SAVGOL_SMOOTH, {5, 2}),
               op(N4M_OP_FINITE_DIFFERENCE, {1})});

    if (profile == static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        add_chain(chains, "savgol_w9_p2_d0",
                  {op(N4M_OP_SAVGOL_SMOOTH, {9, 2})});
        add_chain(chains, "savgol_w11_p3_d0",
                  {op(N4M_OP_SAVGOL_SMOOTH, {11, 3})});
        add_chain(chains, "savgol_w15_p3_d0",
                  {op(N4M_OP_SAVGOL_SMOOTH, {15, 3})});
        add_chain(chains, "savgol_w9_p2_d1",
                  {op(N4M_OP_SAVGOL_DERIVATIVE, {9, 2, 1})});
        add_chain(chains, "savgol_w15_p3_d1",
                  {op(N4M_OP_SAVGOL_DERIVATIVE, {15, 3, 1})});
        add_chain(chains, "savgol_w21_p3_d1",
                  {op(N4M_OP_SAVGOL_DERIVATIVE, {21, 3, 1})});
        add_chain(chains, "savgol_w15_p3_d2",
                  {op(N4M_OP_SAVGOL_DERIVATIVE, {15, 3, 2})});
        add_chain(chains, "nw_s7_g7_d1",
                  {op(N4M_OP_NORRIS_WILLIAMS, {7, 7, 1})});
        add_chain(chains, "nw_s11_g11_d1",
                  {op(N4M_OP_NORRIS_WILLIAMS, {11, 11, 1})});
        add_chain(chains, "nw_s7_g7_d2",
                  {op(N4M_OP_NORRIS_WILLIAMS, {7, 7, 2})});
        add_chain(chains, "finite_diff2",
                  {op(N4M_OP_FINITE_DIFFERENCE, {2})});
        add_chain(chains, "gaussian_s1",
                  {op(N4M_OP_GAUSSIAN, {1.0})});
        add_chain(chains, "gaussian_s2",
                  {op(N4M_OP_GAUSSIAN, {2.0})});
        add_chain(chains, "fck_alpha0",
                  {op(N4M_OP_FCK, {0.0})});
        add_chain(chains, "fck_alpha1",
                  {op(N4M_OP_FCK, {1.0})});
        add_chain(chains, "whittaker_100",
                  {op(N4M_OP_WHITTAKER, {100.0})});
        add_chain(chains, "whittaker_1000",
                  {op(N4M_OP_WHITTAKER, {1000.0})});
        add_chain(chains, "detrend2_savgol_w11_p2_d1",
                  {op(N4M_OP_DETREND_POLY, {2}),
                   op(N4M_OP_SAVGOL_DERIVATIVE, {11, 2, 1})});
        add_chain(chains, "whittaker_100_savgol_w7_p2_d1",
                  {op(N4M_OP_WHITTAKER, {100.0}),
                   op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    }
    return chains;
}

[[nodiscard]] n4m_status_t transform_chain_simple(Context& ctx,
                                                  const ChainSpec& chain,
                                                  const n4m_matrix_view_t& X,
                                                  std::vector<double>& out,
                                                  std::int64_t& rows,
                                                  std::int64_t& cols) {
    rows = X.rows;
    cols = X.cols;
    std::vector<double> current;
    n4m_matrix_view_t current_view = X;
    for (const OperatorEntry& entry : chain.ops) {
        std::vector<double> next;
        const n4m_status_t status =
            transform_aom_strict_operator(ctx, entry, current_view, next);
        if (status != N4M_OK) {
            return status;
        }
        current = std::move(next);
        current_view = rowmajor_view(current, rows, cols);
    }
    if (chain.ops.empty()) {
        ctx.set_error("AOM robust-HPO chain must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out = std::move(current);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t build_chain_operator_matrix(Context& ctx,
                                                       const ChainSpec& chain,
                                                       std::int64_t n_features,
                                                       std::vector<double>& out) {
    const auto p = static_cast<std::size_t>(n_features);
    if (p == 0 ||
        p > (std::numeric_limits<std::size_t>::max() / p)) {
        ctx.set_error("AOM robust-HPO operator matrix dimensions overflow addressable memory");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> basis(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        basis[i * p + i] = 1.0;
    }
    n4m_matrix_view_t basis_view =
        rowmajor_view(basis, n_features, n_features);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    return transform_chain_simple(ctx, chain, basis_view, out, rows, cols);
}

[[nodiscard]] n4m_status_t fold_chain_coefficients_to_input_space(
    Context& ctx,
    const ChainSpec& chain,
    std::int64_t n_features,
    const std::vector<double>& transformed_coefficients,
    std::vector<double>& input_coefficients) {
    const auto p = static_cast<std::size_t>(n_features);
    if (p == 0 || transformed_coefficients.size() != p) {
        ctx.set_error("AOM robust-HPO coefficient folding received inconsistent dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> op_matrix;
    n4m_status_t st = build_chain_operator_matrix(ctx, chain, n_features,
                                                  op_matrix);
    if (st != N4M_OK) return st;
    if (op_matrix.size() != p * p) {
        ctx.set_error("AOM robust-HPO coefficient folding operator has invalid shape");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    input_coefficients.assign(p, 0.0);
    for (std::size_t src = 0; src < p; ++src) {
        double acc = 0.0;
        for (std::size_t mid = 0; mid < p; ++mid) {
            acc += op_matrix[src * p + mid] * transformed_coefficients[mid];
        }
        input_coefficients[src] = acc;
    }
    return N4M_OK;
}

std::vector<Fold> make_contiguous_folds(std::int64_t n_samples,
                                        std::int32_t requested_cv) {
    const std::int32_t n_folds =
        std::max<std::int32_t>(2, std::min<std::int32_t>(
                                      requested_cv,
                                      static_cast<std::int32_t>(n_samples)));
    std::vector<Fold> folds;
    folds.reserve(static_cast<std::size_t>(n_folds));
    std::int64_t start = 0;
    for (std::int32_t fold = 0; fold < n_folds; ++fold) {
        const std::int64_t base = n_samples / n_folds;
        const std::int64_t extra = fold < (n_samples % n_folds) ? 1 : 0;
        const std::int64_t count = base + extra;
        const std::int64_t end = start + count;
        Fold f;
        for (std::int64_t i = 0; i < n_samples; ++i) {
            if (i >= start && i < end) {
                f.valid.push_back(i);
            } else {
                f.train.push_back(i);
            }
        }
        if (!f.valid.empty() && !f.train.empty()) {
            folds.push_back(std::move(f));
        }
        start = end;
    }
    return folds;
}

void subset_rows(const std::vector<double>& X,
                 std::int64_t cols,
                 const std::vector<std::int64_t>& rows,
                 std::vector<double>& out) {
    out.assign(rows.size() * static_cast<std::size_t>(cols), 0.0);
    const auto p = static_cast<std::size_t>(cols);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto src_row = static_cast<std::size_t>(rows[i]);
        for (std::size_t j = 0; j < p; ++j) {
            out[i * p + j] = X[src_row * p + j];
        }
    }
}

void subset_y(const n4m_matrix_view_t& Y,
              const std::vector<std::int64_t>& rows,
              std::vector<double>& out) {
    const auto* y = static_cast<const double*>(Y.data);
    const auto rs = static_cast<std::size_t>(Y.row_stride);
    const auto cs = static_cast<std::size_t>(Y.col_stride);
    out.assign(rows.size(), 0.0);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        out[i] = y[static_cast<std::size_t>(rows[i]) * rs + 0U * cs];
    }
}

void predict_linear(const std::vector<double>& X,
                    std::int64_t rows,
                    std::int64_t cols,
                    const std::vector<double>& coef,
                    double intercept,
                    std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(rows), 0.0);
    const auto n = static_cast<std::size_t>(rows);
    const auto p = static_cast<std::size_t>(cols);
    for (std::size_t i = 0; i < n; ++i) {
        double acc = intercept;
        for (std::size_t j = 0; j < p; ++j) {
            acc += X[i * p + j] * coef[j];
        }
        out[i] = acc;
    }
}

double rmse(const std::vector<double>& truth,
            const std::vector<double>& pred) {
    if (truth.empty() || truth.size() != pred.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double sumsq = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double d = truth[i] - pred[i];
        sumsq += d * d;
    }
    return std::sqrt(sumsq / static_cast<double>(truth.size()));
}

Config base_head_config(const Config& cfg) {
    Config local = cfg;
    local.algorithm = N4M_ALGO_PLS_REGRESSION;
    local.solver = N4M_SOLVER_SIMPLS;
    local.deflation = N4M_DEFLATION_REGRESSION;
    local.center_x = 1;
    local.center_y = 1;
    local.scale_x = 0;
    local.scale_y = 0;
    local.store_scores = 0;
    local.store_diagnostics = 0;
    return local;
}

[[nodiscard]] n4m_status_t fit_ridge_head(Context& ctx,
                                          const Config& cfg,
                                          const std::vector<double>& X,
                                          std::int64_t rows,
                                          std::int64_t cols,
                                          const std::vector<double>& y,
                                          double alpha,
                                          FitResult& out) {
    Config local = base_head_config(cfg);
    local.ridge_fit_intercept = 1;
    std::vector<double> X_copy = X;
    std::vector<double> y_copy = y;
    n4m_matrix_view_t Xv = rowmajor_view(X_copy, rows, cols);
    n4m_matrix_view_t Yv = rowmajor_view(y_copy, rows, 1);
    RidgeResult ridge;
    const n4m_status_t status =
        fit_ridge(ctx, local, Xv, Yv, alpha, RidgeSolver::kAuto,
                  /*fit_intercept=*/true, ridge);
    if (status != N4M_OK) {
        return status;
    }
    out.coefficients = std::move(ridge.coefficients);
    out.intercept = ridge.intercept.empty() ? 0.0 : ridge.intercept[0];
    out.predictions = std::move(ridge.predictions);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_pls_head(Context& ctx,
                                        const Config& cfg,
                                        const std::vector<double>& X,
                                        std::int64_t rows,
                                        std::int64_t cols,
                                        const std::vector<double>& y,
                                        std::int32_t n_components,
                                        FitResult& out) {
    Config local = base_head_config(cfg);
    local.n_components = std::max<std::int32_t>(
        1,
        std::min<std::int32_t>(
            n_components,
            std::min<std::int32_t>(static_cast<std::int32_t>(cols),
                                   std::max<std::int32_t>(
                                       1,
                                       static_cast<std::int32_t>(rows) - 1))));
    std::vector<double> X_copy = X;
    std::vector<double> y_copy = y;
    n4m_matrix_view_t Xv = rowmajor_view(X_copy, rows, cols);
    n4m_matrix_view_t Yv = rowmajor_view(y_copy, rows, 1);
    std::unique_ptr<Model> model;
    const n4m_status_t status = fit_model(ctx, local, Xv, Yv, model);
    if (status != N4M_OK) {
        return status;
    }
    out.coefficients = model->coefficients;
    out.intercept = model->y_mean.empty() ? 0.0 : model->y_mean[0];
    for (std::int64_t j = 0; j < cols; ++j) {
        out.intercept -= model->x_mean[static_cast<std::size_t>(j)] *
                         out.coefficients[static_cast<std::size_t>(j)];
    }
    out.predictions.assign(static_cast<std::size_t>(rows), 0.0);
    n4m_matrix_view_t Pv = rowmajor_view(out.predictions, rows, 1);
    return predict_into(ctx, *model, Xv, Pv);
}

[[nodiscard]] n4m_status_t fit_candidate(Context& ctx,
                                         const Config& cfg,
                                         const Candidate& candidate,
                                         const std::vector<double>& X,
                                         std::int64_t rows,
                                         std::int64_t cols,
                                         const std::vector<double>& y,
                                         FitResult& out) {
    if (candidate.head == AomRobustHpoHead::kRidge) {
        return fit_ridge_head(ctx, cfg, X, rows, cols, y, candidate.param, out);
    }
    return fit_pls_head(ctx, cfg, X, rows, cols, y,
                        static_cast<std::int32_t>(std::llround(candidate.param)),
                        out);
}

std::vector<double> ridge_params() {
    return {1e-4, 1e-2, 1.0, 100.0};
}

std::vector<double> pls_params(std::int32_t profile) {
    if (profile == static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        return {2.0, 4.0, 8.0, 12.0};
    }
    return {2.0, 4.0, 8.0};
}

}  // namespace

n4m_status_t fit_aom_robust_hpo(Context& ctx,
                                const Config& cfg,
                                const n4m_matrix_view_t& X,
                                const n4m_matrix_view_t& Y,
                                std::int32_t profile,
                                std::int32_t cv,
                                std::int32_t heads_mask,
                                AomRobustHpoResult& out) {
    out = AomRobustHpoResult{};
    if (profile != static_cast<std::int32_t>(AomRobustHpoProfile::kCompact) &&
        profile != static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        ctx.set_error("AOM robust-HPO profile must be 0=compact or 1=wide");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cv < 2) {
        ctx.set_error("AOM robust-HPO cv must be >= 2");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int32_t allowed_heads =
        kAomRobustHpoHeadMaskRidge | kAomRobustHpoHeadMaskPls;
    if ((heads_mask & allowed_heads) == 0 || (heads_mask & ~allowed_heads) != 0) {
        ctx.set_error("AOM robust-HPO heads_mask must include ridge=1 and/or pls=2");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.data == nullptr || Y.data == nullptr || X.rows < 2 || X.cols < 1 ||
        Y.rows != X.rows || Y.cols != 1 || X.dtype != N4M_DTYPE_F64 ||
        Y.dtype != N4M_DTYPE_F64) {
        ctx.set_error("AOM robust-HPO requires finite F64 X (n x p) and single-output F64 Y (n x 1)");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const std::vector<ChainSpec> chains = build_bank(profile);
    const std::vector<Fold> folds = make_contiguous_folds(X.rows, cv);
    if (folds.size() < 2U) {
        ctx.set_error("AOM robust-HPO could not build at least two non-empty folds");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<Candidate> candidates;
    for (const ChainSpec& chain : chains) {
        if ((heads_mask & kAomRobustHpoHeadMaskRidge) != 0) {
            for (double alpha : ridge_params()) {
                candidates.push_back({chain.id, AomRobustHpoHead::kRidge, alpha});
            }
        }
        if ((heads_mask & kAomRobustHpoHeadMaskPls) != 0) {
            for (double n_components : pls_params(profile)) {
                candidates.push_back({chain.id, AomRobustHpoHead::kPls,
                                      n_components});
            }
        }
    }

    out.candidate_scores.assign(candidates.size() * 4U, 0.0);
    double best_score = std::numeric_limits<double>::infinity();
    std::size_t best_index = candidates.size();

    std::vector<std::vector<double>> transformed_by_chain(chains.size());
    std::int64_t transformed_rows = X.rows;
    std::int64_t transformed_cols = X.cols;
    for (const ChainSpec& chain : chains) {
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        n4m_status_t status = transform_chain_simple(
            ctx, chain, X, transformed_by_chain[static_cast<std::size_t>(chain.id)],
            rows, cols);
        if (status != N4M_OK) {
            return status;
        }
        transformed_rows = rows;
        transformed_cols = cols;
    }

    for (std::size_t c = 0; c < candidates.size(); ++c) {
        const Candidate& candidate = candidates[c];
        const std::vector<double>& Xt =
            transformed_by_chain[static_cast<std::size_t>(candidate.chain_id)];
        double fold_sum = 0.0;
        std::int32_t n_scored = 0;
        for (const Fold& fold : folds) {
            std::vector<double> X_train;
            std::vector<double> X_valid;
            std::vector<double> y_train;
            std::vector<double> y_valid;
            subset_rows(Xt, transformed_cols, fold.train, X_train);
            subset_rows(Xt, transformed_cols, fold.valid, X_valid);
            subset_y(Y, fold.train, y_train);
            subset_y(Y, fold.valid, y_valid);
            FitResult fitted;
            n4m_status_t status =
                fit_candidate(ctx, cfg, candidate, X_train,
                              static_cast<std::int64_t>(fold.train.size()),
                              transformed_cols, y_train, fitted);
            if (status != N4M_OK) {
                fold_sum = std::numeric_limits<double>::infinity();
                n_scored = static_cast<std::int32_t>(folds.size());
                ctx.clear_error();
                break;
            }
            std::vector<double> pred_valid;
            predict_linear(X_valid,
                           static_cast<std::int64_t>(fold.valid.size()),
                           transformed_cols,
                           fitted.coefficients,
                           fitted.intercept,
                           pred_valid);
            const double fold_rmse = rmse(y_valid, pred_valid);
            fold_sum += fold_rmse;
            ++n_scored;
        }
        const double score =
            (n_scored > 0) ? (fold_sum / static_cast<double>(n_scored))
                           : std::numeric_limits<double>::infinity();
        out.candidate_scores[c * 4U + 0U] =
            static_cast<double>(candidate.chain_id);
        out.candidate_scores[c * 4U + 1U] =
            static_cast<double>(static_cast<std::int32_t>(candidate.head));
        out.candidate_scores[c * 4U + 2U] = candidate.param;
        out.candidate_scores[c * 4U + 3U] = score;
        if (std::isfinite(score) && score < best_score) {
            best_score = score;
            best_index = c;
        }
    }

    if (best_index >= candidates.size()) {
        ctx.set_error("AOM robust-HPO all candidates failed during CV");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    const Candidate& selected = candidates[best_index];
    const std::vector<double>& selected_Xt =
        transformed_by_chain[static_cast<std::size_t>(selected.chain_id)];
    std::vector<double> y_full;
    std::vector<std::int64_t> all_rows(static_cast<std::size_t>(X.rows));
    for (std::int64_t i = 0; i < X.rows; ++i) {
        all_rows[static_cast<std::size_t>(i)] = i;
    }
    subset_y(Y, all_rows, y_full);
    FitResult final_fit;
    const n4m_status_t final_status =
        fit_candidate(ctx, cfg, selected, selected_Xt, transformed_rows,
                      transformed_cols, y_full, final_fit);
    if (final_status != N4M_OK) {
        return final_status;
    }

    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_features_transformed = static_cast<std::int32_t>(transformed_cols);
    out.profile = profile;
    out.cv = static_cast<std::int32_t>(folds.size());
    out.heads_mask = heads_mask;
    out.selected_chain_id = selected.chain_id;
    out.selected_head_id = static_cast<std::int32_t>(selected.head);
    out.selected_param = selected.param;
    out.selected_cv_rmse = best_score;
    out.n_chains = static_cast<std::int32_t>(chains.size());
    out.n_candidates = static_cast<std::int32_t>(candidates.size());
    out.predictions = std::move(final_fit.predictions);
    out.coefficients_transformed = final_fit.coefficients;
    const n4m_status_t fold_status = fold_chain_coefficients_to_input_space(
        ctx, chains[static_cast<std::size_t>(selected.chain_id)], X.cols,
        final_fit.coefficients, out.input_coefficients);
    if (fold_status != N4M_OK) {
        return fold_status;
    }
    out.intercept = {final_fit.intercept};
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
