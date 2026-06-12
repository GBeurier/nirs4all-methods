// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_ridge_blender.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "core/aom_operators.hpp"
#include "core/aom_robust_hpo.hpp"
#include "core/common/matrix_view.hpp"
#include "core/operator_entry.hpp"
#include "core/ridge.hpp"

namespace n4m::core {

namespace {

struct ChainSpec {
    std::int32_t id{0};
    std::vector<OperatorEntry> ops;
};

[[nodiscard]] n4m_matrix_view_t rowmajor_view(std::vector<double>& data,
                                              std::int64_t rows,
                                              std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = data.empty() ? nullptr : data.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    view.reserved0 = 0;
    return view;
}

OperatorEntry op(n4m_operator_kind_t kind,
                 std::initializer_list<double> params = {}) {
    const std::vector<double> owned(params);
    return OperatorEntry(kind,
                         owned.empty() ? nullptr : owned.data(),
                         static_cast<std::int32_t>(owned.size()));
}

void add_chain(std::vector<ChainSpec>& chains,
               std::vector<OperatorEntry> ops) {
    ChainSpec chain{};
    chain.id = static_cast<std::int32_t>(chains.size());
    chain.ops = std::move(ops);
    chains.push_back(std::move(chain));
}

std::vector<ChainSpec> build_bank(std::int32_t profile) {
    std::vector<ChainSpec> chains;
    add_chain(chains, {op(N4M_OP_IDENTITY)});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {1})});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {2})});
    add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {5, 2})});
    add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {7, 2})});
    add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {11, 2, 2})});
    add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {5, 5, 1})});
    add_chain(chains, {op(N4M_OP_FINITE_DIFFERENCE, {1})});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {1}),
                       op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {1}),
                       op(N4M_OP_NORRIS_WILLIAMS, {5, 5, 1})});
    add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {5, 2}),
                       op(N4M_OP_FINITE_DIFFERENCE, {1})});

    if (profile == static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {9, 2})});
        add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {11, 3})});
        add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {15, 3})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {9, 2, 1})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {15, 3, 1})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {21, 3, 1})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {15, 3, 2})});
        add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {7, 7, 1})});
        add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {11, 11, 1})});
        add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {7, 7, 2})});
        add_chain(chains, {op(N4M_OP_FINITE_DIFFERENCE, {2})});
        add_chain(chains, {op(N4M_OP_GAUSSIAN, {1.0})});
        add_chain(chains, {op(N4M_OP_GAUSSIAN, {2.0})});
        add_chain(chains, {op(N4M_OP_FCK, {0.0})});
        add_chain(chains, {op(N4M_OP_FCK, {1.0})});
        add_chain(chains, {op(N4M_OP_WHITTAKER, {100.0})});
        add_chain(chains, {op(N4M_OP_WHITTAKER, {1000.0})});
        add_chain(chains, {op(N4M_OP_DETREND_POLY, {2}),
                           op(N4M_OP_SAVGOL_DERIVATIVE, {11, 2, 1})});
        add_chain(chains, {op(N4M_OP_WHITTAKER, {100.0}),
                           op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    }
    return chains;
}

[[nodiscard]] n4m_status_t copy_matrix(Context& ctx,
                                       const n4m_matrix_view_t& V,
                                       const char* name,
                                       std::vector<double>& out) {
    if (validate_nonnull_view(V) != N4M_OK ||
        (V.dtype != N4M_DTYPE_F64 && V.dtype != N4M_DTYPE_F32)) {
        ctx.set_errorf("%s must be a finite F64/F32 matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (V.rows <= 0 || V.cols <= 0) {
        ctx.set_errorf("%s must be non-empty", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<std::size_t>(V.rows);
    const auto cols = static_cast<std::size_t>(V.cols);
    const auto rs = static_cast<std::size_t>(V.row_stride);
    const auto cs = static_cast<std::size_t>(V.col_stride);
    out.assign(rows * cols, 0.0);
    if (V.dtype == N4M_DTYPE_F64) {
        const auto* data = static_cast<const double*>(V.data);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double value = data[r * rs + c * cs];
                if (!std::isfinite(value)) {
                    ctx.set_errorf("%s contains NaN or Inf", name);
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                out[r * cols + c] = value;
            }
        }
    } else {
        const auto* data = static_cast<const float*>(V.data);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double value = static_cast<double>(data[r * rs + c * cs]);
                if (!std::isfinite(value)) {
                    ctx.set_errorf("%s contains NaN or Inf", name);
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                out[r * cols + c] = value;
            }
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t make_fold_ids(Context& ctx,
                                         std::int64_t n,
                                         std::int32_t cv,
                                         const std::int32_t* fold_ids,
                                         std::int64_t n_fold_ids,
                                         std::vector<std::int32_t>& ids,
                                         std::int32_t& out_cv) {
    if (n <= 1) {
        ctx.set_error("AOM Ridge blender requires at least two rows");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    ids.assign(static_cast<std::size_t>(n), 0);
    if (fold_ids == nullptr || n_fold_ids == 0) {
        if (cv < 2 || cv > n) {
            ctx.set_error("AOM Ridge blender cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        out_cv = cv;
        for (std::int64_t i = 0; i < n; ++i) {
            ids[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>((i * cv) / n);
        }
    } else {
        if (n_fold_ids != n) {
            ctx.set_error("AOM Ridge blender fold_ids length must match n_samples");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::int32_t max_id = -1;
        for (std::int64_t i = 0; i < n; ++i) {
            const auto id = fold_ids[i];
            if (id < 0) {
                ctx.set_error("AOM Ridge blender fold_ids must be non-negative");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            ids[static_cast<std::size_t>(i)] = id;
            max_id = std::max(max_id, id);
        }
        out_cv = cv > 0 ? cv : max_id + 1;
        if (out_cv < 2) {
            ctx.set_error("AOM Ridge blender requires at least two folds");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (const auto id : ids) {
            if (id >= out_cv) {
                ctx.set_error("AOM Ridge blender fold id is outside cv range");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }

    std::vector<std::int64_t> counts(static_cast<std::size_t>(out_cv), 0);
    for (const auto id : ids) {
        ++counts[static_cast<std::size_t>(id)];
    }
    for (std::int32_t f = 0; f < out_cv; ++f) {
        if (counts[static_cast<std::size_t>(f)] == 0) {
            ctx.set_error("AOM Ridge blender requires every fold to contain rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (counts[static_cast<std::size_t>(f)] == n) {
            ctx.set_error("AOM Ridge blender fold would leave an empty train set");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

std::vector<std::vector<std::int64_t>> fold_rows_from_ids(
    const std::vector<std::int32_t>& ids,
    std::int32_t cv) {
    std::vector<std::vector<std::int64_t>> rows(static_cast<std::size_t>(cv));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        rows[static_cast<std::size_t>(ids[i])].push_back(
            static_cast<std::int64_t>(i));
    }
    return rows;
}

std::vector<std::vector<std::int64_t>> train_rows_from_folds(
    std::int64_t n,
    const std::vector<std::vector<std::int64_t>>& fold_rows) {
    std::vector<std::vector<std::int64_t>> out(fold_rows.size());
    std::vector<char> held(static_cast<std::size_t>(n), static_cast<char>(0));
    for (std::size_t fold = 0; fold < fold_rows.size(); ++fold) {
        std::fill(held.begin(), held.end(), static_cast<char>(0));
        for (const auto row : fold_rows[fold]) {
            held[static_cast<std::size_t>(row)] = static_cast<char>(1);
        }
        auto& dst = out[fold];
        dst.reserve(static_cast<std::size_t>(n) - fold_rows[fold].size());
        for (std::int64_t row = 0; row < n; ++row) {
            if (!held[static_cast<std::size_t>(row)]) dst.push_back(row);
        }
    }
    return out;
}

void extract_rows(const std::vector<double>& src,
                  std::int64_t cols,
                  const std::vector<std::int64_t>& rows,
                  std::vector<double>& out) {
    const auto c = static_cast<std::size_t>(cols);
    out.assign(rows.size() * c, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto src_row = static_cast<std::size_t>(rows[r]);
        std::copy_n(src.data() + src_row * c, c, out.data() + r * c);
    }
}

[[nodiscard]] n4m_status_t transform_chain(Context& ctx,
                                           const ChainSpec& chain,
                                           const n4m_matrix_view_t& X,
                                           std::vector<double>& out) {
    if (chain.ops.empty()) {
        ctx.set_error("AOM Ridge blender chain must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> current;
    n4m_matrix_view_t current_view = X;
    for (std::size_t i = 0; i < chain.ops.size(); ++i) {
        std::vector<double> next;
        const n4m_status_t status =
            transform_aom_strict_operator(ctx, chain.ops[i], current_view, next);
        if (status != N4M_OK) return status;
        current = std::move(next);
        current_view = rowmajor_view(current, X.rows, X.cols);
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
        ctx.set_error("AOM Ridge blender operator matrix dimensions overflow addressable memory");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> basis(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        basis[i * p + i] = 1.0;
    }
    n4m_matrix_view_t basis_view =
        rowmajor_view(basis, n_features, n_features);
    return transform_chain(ctx, chain, basis_view, out);
}

[[nodiscard]] n4m_status_t fold_chain_coefficients_to_input_space(
    Context& ctx,
    const ChainSpec& chain,
    std::int64_t n_features,
    std::int64_t n_targets,
    const std::vector<double>& transformed_coefficients,
    std::vector<double>& input_coefficients) {
    const auto p = static_cast<std::size_t>(n_features);
    const auto q = static_cast<std::size_t>(n_targets);
    if (p == 0 || q == 0 || transformed_coefficients.size() != p * q) {
        ctx.set_error("AOM Ridge blender coefficient folding received inconsistent dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> op_matrix;
    n4m_status_t st = build_chain_operator_matrix(ctx, chain, n_features,
                                                  op_matrix);
    if (st != N4M_OK) return st;
    if (op_matrix.size() != p * p) {
        ctx.set_error("AOM Ridge blender coefficient folding operator has invalid shape");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    input_coefficients.assign(p * q, 0.0);
    for (std::size_t src = 0; src < p; ++src) {
        for (std::size_t mid = 0; mid < p; ++mid) {
            const double a = op_matrix[src * p + mid];
            if (a == 0.0) continue;
            for (std::size_t target = 0; target < q; ++target) {
                input_coefficients[src * q + target] +=
                    a * transformed_coefficients[mid * q + target];
            }
        }
    }
    return N4M_OK;
}

void predict_rows(const RidgeResult& fit,
                  const std::vector<double>& X,
                  std::int64_t p,
                  std::int64_t q,
                  const std::vector<std::int64_t>& rows,
                  std::vector<double>& oof,
                  std::int64_t candidate,
                  std::int64_t n_candidates) {
    const auto cols = static_cast<std::size_t>(p);
    const auto targets = static_cast<std::size_t>(q);
    const auto nc = static_cast<std::size_t>(n_candidates);
    const auto cand = static_cast<std::size_t>(candidate);
    for (const auto row_i64 : rows) {
        const auto row = static_cast<std::size_t>(row_i64);
        for (std::size_t t = 0; t < targets; ++t) {
            double acc = fit.intercept[t];
            for (std::size_t c = 0; c < cols; ++c) {
                acc += X[row * cols + c] * fit.coefficients[c * targets + t];
            }
            oof[(row * targets + t) * nc + cand] = acc;
        }
    }
}

double rmse_from_candidate(const std::vector<double>& Z,
                           const std::vector<double>& y,
                           std::int64_t n_rows,
                           std::int64_t n_candidates,
                           std::int64_t candidate) {
    double sse = 0.0;
    const auto rows = static_cast<std::size_t>(n_rows);
    const auto nc = static_cast<std::size_t>(n_candidates);
    const auto cand = static_cast<std::size_t>(candidate);
    for (std::size_t r = 0; r < rows; ++r) {
        const double diff = Z[r * nc + cand] - y[r];
        sse += diff * diff;
    }
    return std::sqrt(sse / static_cast<double>(rows));
}

std::vector<double> project_simplex(std::vector<double> values) {
    const std::size_t k = values.size();
    if (k == 0) return values;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    double cumsum = 0.0;
    std::size_t rho = 0;
    bool have = false;
    for (std::size_t i = 0; i < k; ++i) {
        cumsum += sorted[i];
        const double theta = (cumsum - 1.0) / static_cast<double>(i + 1U);
        if (sorted[i] - theta > 0.0) {
            rho = i;
            have = true;
        }
    }
    if (!have) {
        return std::vector<double>(k, 1.0 / static_cast<double>(k));
    }
    double active_sum = 0.0;
    for (std::size_t i = 0; i <= rho; ++i) active_sum += sorted[i];
    const double theta = (active_sum - 1.0) / static_cast<double>(rho + 1U);
    double total = 0.0;
    for (double& v : values) {
        v = std::max(v - theta, 0.0);
        total += v;
    }
    if (total <= 0.0 || !std::isfinite(total)) {
        return std::vector<double>(k, 1.0 / static_cast<double>(k));
    }
    for (double& v : values) v /= total;
    return values;
}

std::vector<double> solve_simplex_qp(const std::vector<double>& Z,
                                     const std::vector<double>& y,
                                     std::int64_t n_rows,
                                     std::int64_t n_candidates,
                                     double regularizer) {
    const auto rows = static_cast<std::size_t>(n_rows);
    const auto k = static_cast<std::size_t>(n_candidates);
    const double lam = std::max(regularizer, 0.0);
    std::vector<double> uniform(k, 1.0 / static_cast<double>(k));
    std::vector<double> ztz(k * k, 0.0);
    std::vector<double> zty(k, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < k; ++i) {
            const double zi = Z[r * k + i];
            zty[i] += zi * y[r];
            for (std::size_t j = 0; j < k; ++j) {
                ztz[i * k + j] += zi * Z[r * k + j];
            }
        }
    }
    double frob = 0.0;
    for (const double v : ztz) frob += v * v;
    const double step = 1.0 / std::max(std::sqrt(frob) + lam, 1e-12);
    std::vector<double> weights = uniform;
    std::vector<double> grad(k, 0.0);
    for (std::int32_t iter = 0; iter < 2000; ++iter) {
        std::fill(grad.begin(), grad.end(), 0.0);
        for (std::size_t i = 0; i < k; ++i) {
            double g = -zty[i] + lam * (weights[i] - uniform[i]);
            for (std::size_t j = 0; j < k; ++j) {
                g += ztz[i * k + j] * weights[j];
            }
            grad[i] = g;
        }
        std::vector<double> updated(k, 0.0);
        for (std::size_t i = 0; i < k; ++i) {
            updated[i] = weights[i] - step * grad[i];
        }
        updated = project_simplex(std::move(updated));
        double l1 = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            l1 += std::fabs(updated[i] - weights[i]);
        }
        weights = std::move(updated);
        if (l1 <= 1e-12) break;
    }
    return weights;
}

void blend_predictions(const std::vector<double>& candidates,
                       const std::vector<double>& weights,
                       std::int64_t n_rows,
                       std::int64_t n_candidates,
                       std::vector<double>& out) {
    const auto rows = static_cast<std::size_t>(n_rows);
    const auto k = static_cast<std::size_t>(n_candidates);
    out.assign(rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            acc += candidates[r * k + c] * weights[c];
        }
        out[r] = acc;
    }
}

}  // namespace

n4m_status_t fit_aom_ridge_blender(Context& ctx,
                                   const Config& cfg,
                                   const n4m_matrix_view_t& X,
                                   const n4m_matrix_view_t& Y,
                                   std::int32_t profile,
                                   std::int32_t cv,
                                   const std::int32_t* fold_ids,
                                   std::int64_t n_fold_ids,
                                   const double* ridge_lambdas,
                                   std::int64_t n_ridge_lambdas,
                                   double regularizer,
                                   AomRidgeBlenderResult& out) {
    if (profile != static_cast<std::int32_t>(AomRobustHpoProfile::kCompact) &&
        profile != static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        ctx.set_error("AOM Ridge blender profile must be 0=compact or 1=wide");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (ridge_lambdas == nullptr || n_ridge_lambdas <= 0) {
        ctx.set_error("AOM Ridge blender requires at least one Ridge lambda");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::int64_t i = 0; i < n_ridge_lambdas; ++i) {
        if (!std::isfinite(ridge_lambdas[i]) || ridge_lambdas[i] <= 0.0) {
            ctx.set_error("AOM Ridge blender lambdas must be finite and > 0");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    if (!std::isfinite(regularizer) || regularizer < 0.0) {
        ctx.set_error("AOM Ridge blender regularizer must be finite and >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> Xall;
    std::vector<double> Yall;
    n4m_status_t st = copy_matrix(ctx, X, "X", Xall);
    if (st != N4M_OK) return st;
    st = copy_matrix(ctx, Y, "Y", Yall);
    if (st != N4M_OK) return st;
    if (X.rows != Y.rows) {
        ctx.set_error("AOM Ridge blender: X and Y must have the same rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }

    std::vector<std::int32_t> ids;
    std::int32_t actual_cv = 0;
    st = make_fold_ids(ctx, X.rows, cv, fold_ids, n_fold_ids, ids, actual_cv);
    if (st != N4M_OK) return st;
    const auto valid_rows = fold_rows_from_ids(ids, actual_cv);
    const auto train_rows = train_rows_from_folds(X.rows, valid_rows);
    const auto chains = build_bank(profile);

    const auto n = X.rows;
    const auto p = X.cols;
    const auto q = Y.cols;
    const auto n_flat = n * q;
    const auto n_candidates =
        static_cast<std::int64_t>(chains.size()) * n_ridge_lambdas;
    out = AomRidgeBlenderResult{};
    out.n_samples = n;
    out.n_features = p;
    out.n_targets = q;
    out.profile = profile;
    out.cv = actual_cv;
    out.n_chains = static_cast<std::int64_t>(chains.size());
    out.n_candidates = n_candidates;
    out.regularizer = regularizer;
    out.fold_ids = ids;
    out.oof_candidate_predictions.assign(
        static_cast<std::size_t>(n_flat * n_candidates), 0.0);
    out.candidate_predictions.assign(
        static_cast<std::size_t>(n_flat * n_candidates), 0.0);
    out.candidate_scores.assign(static_cast<std::size_t>(n_candidates * 5), 0.0);
    std::vector<double> candidate_input_coefficients(
        static_cast<std::size_t>(n_candidates * p * q), 0.0);
    std::vector<double> candidate_intercepts(
        static_cast<std::size_t>(n_candidates * q), 0.0);

    std::int64_t candidate = 0;
    for (const auto& chain : chains) {
        std::vector<double> Xt;
        st = transform_chain(ctx, chain, X, Xt);
        if (st != N4M_OK) return st;

        for (std::int64_t lambda_ix = 0; lambda_ix < n_ridge_lambdas; ++lambda_ix) {
            const double lambda = ridge_lambdas[lambda_ix];
            const auto score_offset = static_cast<std::size_t>(candidate * 5);
            out.candidate_scores[score_offset + 0] = static_cast<double>(candidate);
            out.candidate_scores[score_offset + 1] = static_cast<double>(chain.id);
            out.candidate_scores[score_offset + 2] = lambda;

            for (std::size_t fold = 0; fold < valid_rows.size(); ++fold) {
                std::vector<double> Xtrain;
                std::vector<double> Ytrain;
                extract_rows(Xt, p, train_rows[fold], Xtrain);
                extract_rows(Yall, q, train_rows[fold], Ytrain);
                n4m_matrix_view_t Xtv = rowmajor_view(
                    Xtrain, static_cast<std::int64_t>(train_rows[fold].size()), p);
                n4m_matrix_view_t Ytv = rowmajor_view(
                    Ytrain, static_cast<std::int64_t>(train_rows[fold].size()), q);
                RidgeResult fit;
                st = fit_ridge(ctx, cfg, Xtv, Ytv, lambda, RidgeSolver::kAuto,
                               true, fit);
                if (st != N4M_OK) return st;
                predict_rows(fit, Xt, p, q, valid_rows[fold],
                             out.oof_candidate_predictions, candidate,
                             n_candidates);
            }

            out.candidate_scores[score_offset + 3] =
                rmse_from_candidate(out.oof_candidate_predictions, Yall, n_flat,
                                    n_candidates, candidate);

            n4m_matrix_view_t Xfull = rowmajor_view(Xt, n, p);
            n4m_matrix_view_t Yfull = rowmajor_view(Yall, n, q);
            RidgeResult full_fit;
            st = fit_ridge(ctx, cfg, Xfull, Yfull, lambda, RidgeSolver::kAuto,
                           true, full_fit);
            if (st != N4M_OK) return st;
            std::vector<double> folded;
            st = fold_chain_coefficients_to_input_space(
                ctx, chain, p, q, full_fit.coefficients, folded);
            if (st != N4M_OK) return st;
            std::copy(folded.begin(), folded.end(),
                      candidate_input_coefficients.begin() +
                          static_cast<std::ptrdiff_t>(candidate * p * q));
            std::copy(full_fit.intercept.begin(), full_fit.intercept.end(),
                      candidate_intercepts.begin() +
                          static_cast<std::ptrdiff_t>(candidate * q));
            for (std::int64_t row = 0; row < n_flat; ++row) {
                out.candidate_predictions[
                    static_cast<std::size_t>(row * n_candidates + candidate)] =
                    full_fit.predictions[static_cast<std::size_t>(row)];
            }
            ++candidate;
        }
    }

    out.weights = solve_simplex_qp(out.oof_candidate_predictions, Yall, n_flat,
                                   n_candidates, regularizer);
    out.input_coefficients.assign(static_cast<std::size_t>(p * q), 0.0);
    out.intercept.assign(static_cast<std::size_t>(q), 0.0);
    for (std::int64_t c = 0; c < n_candidates; ++c) {
        const double weight = out.weights[static_cast<std::size_t>(c)];
        const auto coef_offset = static_cast<std::size_t>(c * p * q);
        for (std::int64_t ix = 0; ix < p * q; ++ix) {
            out.input_coefficients[static_cast<std::size_t>(ix)] +=
                weight * candidate_input_coefficients[
                    coef_offset + static_cast<std::size_t>(ix)];
        }
        const auto intercept_offset = static_cast<std::size_t>(c * q);
        for (std::int64_t target = 0; target < q; ++target) {
            out.intercept[static_cast<std::size_t>(target)] +=
                weight * candidate_intercepts[
                    intercept_offset + static_cast<std::size_t>(target)];
        }
    }
    for (std::int64_t c = 0; c < n_candidates; ++c) {
        out.candidate_scores[static_cast<std::size_t>(c * 5 + 4)] =
            out.weights[static_cast<std::size_t>(c)];
    }
    auto best_it = std::max_element(out.weights.begin(), out.weights.end());
    out.selected_candidate_id =
        static_cast<std::int64_t>(std::distance(out.weights.begin(), best_it));
    const auto selected_offset =
        static_cast<std::size_t>(out.selected_candidate_id * 5);
    out.selected_chain_id =
        static_cast<std::int64_t>(out.candidate_scores[selected_offset + 1]);
    out.selected_param = out.candidate_scores[selected_offset + 2];
    out.selected_cv_rmse = out.candidate_scores[selected_offset + 3];

    std::vector<double> blended_oof_flat;
    std::vector<double> blended_pred_flat;
    blend_predictions(out.oof_candidate_predictions, out.weights, n_flat,
                      n_candidates, blended_oof_flat);
    blend_predictions(out.candidate_predictions, out.weights, n_flat,
                      n_candidates, blended_pred_flat);
    out.oof_predictions = std::move(blended_oof_flat);
    out.predictions = std::move(blended_pred_flat);

    double blend_sse = 0.0;
    for (std::int64_t row = 0; row < n_flat; ++row) {
        const double diff =
            out.oof_predictions[static_cast<std::size_t>(row)] -
            Yall[static_cast<std::size_t>(row)];
        blend_sse += diff * diff;
    }
    out.blend_oof_rmse = std::sqrt(blend_sse / static_cast<double>(n_flat));
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
