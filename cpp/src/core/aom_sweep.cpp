// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "core/aom_operators.hpp"
#include "core/aom_robust_hpo.hpp"
#include "core/common/banded_solver.h"
#include "core/common/linalg.hpp"
#include "core/common/matrix_view.hpp"
#include "core/moments.hpp"
#include "core/operator_entry.hpp"
#include "core/sweep.hpp"

namespace n4m::core {

namespace {

constexpr std::size_t kMaxWideOperatorMomentFeatures = 48;
constexpr std::size_t kMaxBandedOperatorMomentFeatures = 1024;
constexpr std::size_t kMaxBandedRidgeMomentFeatures = 256;
constexpr std::size_t kMaxMomentPrefixCacheFeatures = 256;
constexpr std::size_t kMaxMomentPrefixCacheEntries = 64;

enum class OperatorMomentRoute {
    kNone,
    kDense,
    kBanded,
    kStructured,
};

[[nodiscard]] std::int32_t candidate_route_code(
    OperatorMomentRoute route) noexcept {
    switch (route) {
    case OperatorMomentRoute::kNone:
        return 0;
    case OperatorMomentRoute::kDense:
        return 1;
    case OperatorMomentRoute::kBanded:
        return 2;
    case OperatorMomentRoute::kStructured:
        return 3;
    }
    return 0;
}

struct ChainSpec {
    std::int32_t id{0};
    std::vector<OperatorEntry> ops;
};

struct MomentPrefixKey {
    std::vector<std::int32_t> op_kinds;
    std::vector<std::int32_t> param_offsets;
    std::vector<double> params;

    [[nodiscard]] bool operator<(const MomentPrefixKey& other) const {
        if (op_kinds != other.op_kinds) return op_kinds < other.op_kinds;
        if (param_offsets != other.param_offsets) {
            return param_offsets < other.param_offsets;
        }
        return params < other.params;
    }
};

struct CachedMomentSet {
    MomentStats all;
    std::vector<MomentStats> heldout;
    OperatorMomentRoute route{OperatorMomentRoute::kNone};
};

struct MomentPrefixCache {
    bool enabled{false};
    std::map<MomentPrefixKey, CachedMomentSet> entries;
    std::int64_t hits{0};
    std::int64_t misses{0};
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

void export_chain_descriptor(const std::vector<ChainSpec>& chains,
                             AomSweepResult& out) {
    out.chain_offsets.clear();
    out.op_kinds.clear();
    out.param_offsets.clear();
    out.chain_params.clear();
    out.chain_offsets.reserve(chains.size() + 1);
    out.param_offsets.reserve(chains.size() + 1);
    out.chain_offsets.push_back(0);
    out.param_offsets.push_back(0);

    for (const auto& chain : chains) {
        for (const auto& entry : chain.ops) {
            out.op_kinds.push_back(static_cast<std::int32_t>(entry.kind));
            out.chain_params.insert(out.chain_params.end(),
                                    entry.params.begin(),
                                    entry.params.end());
            out.param_offsets.push_back(
                static_cast<std::int32_t>(out.chain_params.size()));
        }
        out.chain_offsets.push_back(
            static_cast<std::int32_t>(out.op_kinds.size()));
    }
}

[[nodiscard]] bool is_supported_strict_operator(n4m_operator_kind_t kind) noexcept {
    switch (kind) {
        case N4M_OP_IDENTITY:
        case N4M_OP_DETREND_POLY:
        case N4M_OP_SAVGOL_SMOOTH:
        case N4M_OP_SAVGOL_DERIVATIVE:
        case N4M_OP_NORRIS_WILLIAMS:
        case N4M_OP_FINITE_DIFFERENCE:
        case N4M_OP_WHITTAKER:
        case N4M_OP_FCK:
        case N4M_OP_GAUSSIAN:
            return true;
        default:
            return false;
    }
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

[[nodiscard]] n4m_status_t validate_offsets(Context& ctx,
                                            const std::int32_t* offsets,
                                            std::int64_t n_offsets,
                                            std::int64_t expected_last,
                                            const char* name) {
    if (offsets == nullptr || n_offsets < 2) {
        ctx.set_errorf("AOM chain sweep %s must contain at least two offsets", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (offsets[0] != 0) {
        ctx.set_errorf("AOM chain sweep %s must start at zero", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::int64_t i = 1; i < n_offsets; ++i) {
        if (offsets[i] < offsets[i - 1]) {
            ctx.set_errorf("AOM chain sweep %s must be monotonic", name);
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    if (static_cast<std::int64_t>(offsets[n_offsets - 1]) != expected_last) {
        ctx.set_errorf("AOM chain sweep %s final offset does not match payload size", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t build_chains_from_descriptor(
    Context& ctx,
    const std::int32_t* chain_offsets,
    std::int64_t n_chain_offsets,
    const std::int32_t* op_kinds,
    std::int64_t n_op_kinds,
    const std::int32_t* param_offsets,
    std::int64_t n_param_offsets,
    const double* params,
    std::int64_t n_params,
    std::vector<ChainSpec>& chains) {
    chains.clear();
    if (n_op_kinds <= 0 || op_kinds == nullptr) {
        ctx.set_error("AOM chain sweep requires at least one operator");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_params < 0) {
        ctx.set_error("AOM chain sweep parameter count must be non-negative");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_params > 0 && params == nullptr) {
        ctx.set_error("AOM chain sweep parameters pointer is null");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_param_offsets != n_op_kinds + 1) {
        ctx.set_error("AOM chain sweep param_offsets length must be n_ops + 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_status_t st = validate_offsets(ctx, chain_offsets, n_chain_offsets,
                                       n_op_kinds, "chain_offsets");
    if (st != N4M_OK) return st;
    st = validate_offsets(ctx, param_offsets, n_param_offsets,
                          n_params, "param_offsets");
    if (st != N4M_OK) return st;

    const std::int64_t n_chains = n_chain_offsets - 1;
    chains.reserve(static_cast<std::size_t>(n_chains));
    for (std::int64_t chain_i = 0; chain_i < n_chains; ++chain_i) {
        const std::int32_t begin = chain_offsets[chain_i];
        const std::int32_t end = chain_offsets[chain_i + 1];
        if (begin == end) {
            ctx.set_error("AOM chain sweep does not allow empty chains; use identity");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<OperatorEntry> ops;
        ops.reserve(static_cast<std::size_t>(end - begin));
        for (std::int32_t op_i = begin; op_i < end; ++op_i) {
            const auto kind = static_cast<n4m_operator_kind_t>(op_kinds[op_i]);
            if (!is_supported_strict_operator(kind)) {
                ctx.set_errorf("AOM chain sweep operator %d is not a supported strict-linear operator",
                               static_cast<int>(op_kinds[op_i]));
                return N4M_ERR_UNSUPPORTED;
            }
            const std::int32_t p_begin = param_offsets[op_i];
            const std::int32_t p_end = param_offsets[op_i + 1];
            const double* p = (p_end > p_begin) ? params + p_begin : nullptr;
            ops.emplace_back(kind, p, p_end - p_begin);
        }
        add_chain(chains, std::move(ops));
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_xy(Context& ctx,
                                       const n4m_matrix_view_t& X,
                                       const n4m_matrix_view_t& Y) {
    n4m_status_t st = validate_nonnull_view(X);
    if (st != N4M_OK) {
        ctx.set_error("AOM sweep X matrix view is invalid");
        return st;
    }
    st = validate_nonnull_view(Y);
    if (st != N4M_OK) {
        ctx.set_error("AOM sweep Y matrix view is invalid");
        return st;
    }
    if (X.dtype != N4M_DTYPE_F64 || Y.dtype != N4M_DTYPE_F64) {
        ctx.set_error("AOM sweep requires F64 X and Y matrices");
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (X.rows <= 1 || X.cols <= 0 || Y.rows != X.rows || Y.cols <= 0) {
        ctx.set_error("AOM sweep requires X (n>1,p>0) and Y with matching rows");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t transform_chain(Context& ctx,
                                           const ChainSpec& chain,
                                           const n4m_matrix_view_t& X,
                                           std::vector<double>& out) {
    std::vector<double> current;
    n4m_matrix_view_t current_view = X;
    for (const auto& entry : chain.ops) {
        std::vector<double> next;
        const n4m_status_t st =
            transform_aom_strict_operator(ctx, entry, current_view, next);
        if (st != N4M_OK) {
            out.clear();
            return st;
        }
        current = std::move(next);
        current_view = rowmajor_view(current, X.rows, X.cols);
    }
    if (chain.ops.empty()) {
        ctx.set_error("AOM sweep chain must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out = std::move(current);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t build_aom_fold_ids(Context& ctx,
                                              std::int64_t n,
                                              std::int32_t requested_cv,
                                              const std::int32_t* given,
                                              std::int64_t n_given,
                                              std::vector<std::int32_t>& ids,
                                              std::int32_t& out_cv) {
    if (given == nullptr || n_given == 0) {
        if (requested_cv < 2 || requested_cv > n) {
            ctx.set_error("AOM moment sweep cv must be in [2, n_samples] when fold_ids are not provided");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        ids.assign(static_cast<std::size_t>(n), 0);
        for (std::int64_t i = 0; i < n; ++i) {
            ids[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>((i * requested_cv) / n);
        }
        out_cv = requested_cv;
    } else {
        if (n_given != n) {
            ctx.set_error("AOM moment sweep fold_ids length must equal n_samples");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        ids.assign(given, given + static_cast<std::size_t>(n_given));
        std::int32_t max_id = -1;
        for (const auto id : ids) {
            if (id < 0) {
                ctx.set_error("AOM moment sweep fold_ids must be non-negative");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            max_id = std::max(max_id, id);
        }
        out_cv = (requested_cv > 0) ? requested_cv : (max_id + 1);
        if (out_cv < 2 || out_cv > n) {
            ctx.set_error("AOM moment sweep inferred/provided cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (const auto id : ids) {
            if (id >= out_cv) {
                ctx.set_error("AOM moment sweep fold_ids contain id >= cv");
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
            ctx.set_error("AOM moment sweep requires every fold to contain at least one row");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (counts[static_cast<std::size_t>(f)] == n) {
            ctx.set_error("AOM moment sweep fold would leave an empty train set");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

std::vector<std::vector<std::int64_t>> aom_fold_rows_from_ids(
    const std::vector<std::int32_t>& ids,
    std::int32_t cv) {
    std::vector<std::vector<std::int64_t>> rows(static_cast<std::size_t>(cv));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        rows[static_cast<std::size_t>(ids[i])].push_back(
            static_cast<std::int64_t>(i));
    }
    return rows;
}

[[nodiscard]] std::size_t min_train_rows(std::size_t n,
                                         const std::vector<std::vector<std::int64_t>>& fold_rows) {
    std::size_t out = n;
    for (const auto& rows : fold_rows) {
        out = std::min(out, n - rows.size());
    }
    return out;
}

[[nodiscard]] bool ridge_grid_is_strictly_positive(
    const Config& cfg,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas) noexcept {
    if (n_ridge_lambdas < 0) return false;
    if (n_ridge_lambdas == 0) {
        return std::isfinite(cfg.ridge_lambda) && cfg.ridge_lambda > 0.0;
    }
    if (ridge_lambdas == nullptr) return false;
    for (std::int64_t i = 0; i < n_ridge_lambdas; ++i) {
        if (!std::isfinite(ridge_lambdas[i]) || ridge_lambdas[i] <= 0.0) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool can_use_operator_moment_ridge_features(
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas) {
    const auto p = static_cast<std::size_t>(X.cols);
    const auto n = static_cast<std::size_t>(X.rows);
    if (p <= min_train_rows(n, fold_rows)) return true;
    if (p > kMaxWideOperatorMomentFeatures) return false;
    return ridge_grid_is_strictly_positive(cfg, ridge_lambdas, n_ridge_lambdas);
}

[[nodiscard]] bool can_use_banded_operator_moment_ridge_features(
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas) {
    const auto p = static_cast<std::size_t>(X.cols);
    const auto n = static_cast<std::size_t>(X.rows);
    if (p <= min_train_rows(n, fold_rows)) return true;
    if (p > kMaxBandedRidgeMomentFeatures) return false;
    return ridge_grid_is_strictly_positive(cfg, ridge_lambdas, n_ridge_lambdas);
}

[[nodiscard]] bool can_use_operator_moment_ridge_path(
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    std::int32_t heads_mask,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas) {
    if (heads_mask != kSweepHeadRidge) return false;
    return can_use_operator_moment_ridge_features(
               cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas) ||
           can_use_banded_operator_moment_ridge_features(
               cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas);
}

[[nodiscard]] bool can_use_operator_moment_ridge_part(
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    std::int32_t heads_mask,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas) {
    if ((heads_mask & kSweepHeadRidge) == 0) return false;
    return can_use_operator_moment_ridge_features(
               cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas) ||
           can_use_banded_operator_moment_ridge_features(
               cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas);
}

[[nodiscard]] bool can_use_operator_moment_pls_part(
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    std::int32_t heads_mask) {
    if ((heads_mask & kSweepHeadPls) == 0) return false;
    if (Y.cols != 1) return false;
    if (cfg.solver != N4M_SOLVER_NIPALS ||
        cfg.deflation != N4M_DEFLATION_REGRESSION) {
        return false;
    }
    const auto p = static_cast<std::size_t>(X.cols);
    const auto n = static_cast<std::size_t>(X.rows);
    return p <= min_train_rows(n, fold_rows) ||
           p <= kMaxWideOperatorMomentFeatures ||
           p <= kMaxBandedOperatorMomentFeatures;
}

[[nodiscard]] bool uses_cuda_linalg_build() noexcept {
#if defined(N4M_USE_CUDA)
    return true;
#else
    return false;
#endif
}

[[nodiscard]] bool force_operator_moments_policy(const Config& cfg) noexcept {
    return cfg.aom_moment_policy ==
           static_cast<std::int32_t>(N4M_AOM_MOMENT_FORCE_MOMENTS);
}

[[nodiscard]] bool aom_score_only_policy(const Config& cfg) noexcept {
    return cfg.aom_score_only != 0;
}

[[nodiscard]] bool aom_pls_gcv_proxy_mode(const Config& cfg) noexcept {
    return cfg.aom_pls_score_mode ==
           static_cast<std::int32_t>(N4M_AOM_PLS_SCORE_GCV_PROXY);
}

[[nodiscard]] n4m_status_t reject_forced_moment_fallback(Context& ctx,
                                                         const char* reason) {
    ctx.set_errorf("AOM force_moments policy rejected materialized screen fallback: %s",
                   reason);
    return N4M_ERR_UNSUPPORTED;
}

[[nodiscard]] bool should_materialize_cpu_wide_ridge(
    const n4m_matrix_view_t& X,
    const std::vector<std::vector<std::int64_t>>& fold_rows) {
    if (uses_cuda_linalg_build()) {
        return false;
    }
    const auto p = static_cast<std::size_t>(X.cols);
    const auto n = static_cast<std::size_t>(X.rows);
    return p > min_train_rows(n, fold_rows);
}

[[nodiscard]] bool should_materialize_cpu_wide_pls(
    const n4m_matrix_view_t& X,
    const std::vector<std::vector<std::int64_t>>& fold_rows) {
    if (uses_cuda_linalg_build()) {
        return false;
    }
    const auto p = static_cast<std::size_t>(X.cols);
    const auto n = static_cast<std::size_t>(X.rows);
    const auto min_train = min_train_rows(n, fold_rows);
    return min_train < (4U * p);
}

[[nodiscard]] n4m_status_t build_chain_operator_matrix(Context& ctx,
                                                       const ChainSpec& chain,
                                                       std::int64_t n_features,
                                                       std::vector<double>& out) {
    const auto p = static_cast<std::size_t>(n_features);
    if (p == 0 ||
        p > (std::numeric_limits<std::size_t>::max() / p)) {
        ctx.set_error("AOM operator matrix dimensions overflow addressable memory");
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
        ctx.set_error("AOM coefficient folding received inconsistent dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> op_matrix;
    n4m_status_t st = build_chain_operator_matrix(ctx, chain, n_features,
                                                  op_matrix);
    if (st != N4M_OK) return st;
    if (op_matrix.size() != p * p) {
        ctx.set_error("AOM coefficient folding operator has invalid shape");
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

[[nodiscard]] n4m_status_t transform_moments_by_operator(
    Context& ctx,
    const MomentStats& stats,
    const std::vector<double>& op_matrix,
    MomentStats& out) {
    if (stats.n_samples <= 0 || stats.n_features <= 0 ||
        stats.n_targets <= 0) {
        ctx.set_error("AOM moment transform requires positive moment dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    if (op_matrix.size() != p * p ||
        stats.x_sum.size() != p ||
        stats.y_sum.size() != q ||
        stats.xtx.size() != p * p ||
        stats.xty.size() != p * q ||
        stats.yty.size() != q * q) {
        ctx.set_error("AOM moment transform received inconsistent buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out = MomentStats{};
    out.n_samples = stats.n_samples;
    out.n_features = stats.n_features;
    out.n_targets = stats.n_targets;
    out.y_sum = stats.y_sum;
    out.yty = stats.yty;
    out.x_sum.assign(p, 0.0);
    out.xtx.assign(p * p, 0.0);
    out.xty.assign(p * q, 0.0);

    for (std::size_t col = 0; col < p; ++col) {
        double sum = 0.0;
        for (std::size_t src = 0; src < p; ++src) {
            sum += stats.x_sum[src] * op_matrix[src * p + col];
        }
        out.x_sum[col] = sum;
    }

    linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                 p, q, p, 1.0,
                 op_matrix.data(), p, stats.xty.data(), q,
                 0.0, out.xty.data(), q);

    std::vector<double> temp(p * p, 0.0);
    linalg::gemm(linalg::Trans_No, linalg::Trans_No,
                 p, p, p, 1.0,
                 stats.xtx.data(), p, op_matrix.data(), p,
                 0.0, temp.data(), p);
    linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                 p, p, p, 1.0,
                 op_matrix.data(), p, temp.data(), p,
                 0.0, out.xtx.data(), p);

    return recompute_centered_moments(ctx, out);
}

[[nodiscard]] bool invert_small_square(std::vector<double> matrix,
                                       std::size_t n,
                                       std::vector<double>& inverse) {
    inverse.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i * n + i] = 1.0;
    }
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(matrix[col * n + col]);
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double candidate = std::fabs(matrix[row * n + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(matrix[col * n + j], matrix[pivot * n + j]);
                std::swap(inverse[col * n + j], inverse[pivot * n + j]);
            }
        }
        const double diag = matrix[col * n + col];
        for (std::size_t j = 0; j < n; ++j) {
            matrix[col * n + j] /= diag;
            inverse[col * n + j] /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = matrix[row * n + col];
            for (std::size_t j = 0; j < n; ++j) {
                matrix[row * n + j] -= factor * matrix[col * n + j];
                inverse[row * n + j] -= factor * inverse[col * n + j];
            }
        }
    }
    return true;
}

[[nodiscard]] n4m_status_t parse_aom_detrend_degree(Context& ctx,
                                                    const OperatorEntry& entry,
                                                    std::int32_t& degree) {
    degree = 1;
    if (entry.params.empty()) {
        return N4M_OK;
    }
    if (entry.params.size() != 1U) {
        ctx.set_error("AOM detrend expects zero params or one degree param");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const double raw = entry.params[0];
    const double rounded = std::round(raw);
    if (!std::isfinite(raw) || std::fabs(raw - rounded) > 1e-12 ||
        rounded < 0.0 || rounded > 5.0) {
        ctx.set_error("AOM detrend degree must be an integer in [0, 5]");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    degree = static_cast<std::int32_t>(rounded);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t build_detrend_projection_basis(
    Context& ctx,
    const OperatorEntry& entry,
    std::size_t p,
    std::vector<double>& basis,
    std::vector<double>& gram_inverse) {
    std::int32_t degree = 1;
    n4m_status_t st = parse_aom_detrend_degree(ctx, entry, degree);
    if (st != N4M_OK) {
        return st;
    }
    if (static_cast<std::size_t>(degree) >= p) {
        ctx.set_error("AOM detrend degree must be smaller than the column count");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto terms = static_cast<std::size_t>(degree) + 1U;
    basis.assign(p * terms, 1.0);
    for (std::size_t col = 0; col < p; ++col) {
        const double x = p > 1U
            ? -1.0 + 2.0 * static_cast<double>(col) /
                         static_cast<double>(p - 1U)
            : 0.0;
        for (std::size_t term = 1; term < terms; ++term) {
            basis[col * terms + term] =
                basis[col * terms + term - 1U] * x;
        }
    }

    std::vector<double> gram(terms * terms, 0.0);
    for (std::size_t row_term = 0; row_term < terms; ++row_term) {
        for (std::size_t col_term = 0; col_term < terms; ++col_term) {
            double sum = 0.0;
            for (std::size_t col = 0; col < p; ++col) {
                sum += basis[col * terms + row_term] *
                       basis[col * terms + col_term];
            }
            gram[row_term * terms + col_term] = sum;
        }
    }
    if (!invert_small_square(std::move(gram), terms, gram_inverse)) {
        ctx.set_error("failed to invert AOM detrend normal equations");
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    return N4M_OK;
}

void apply_detrend_projection_right(const std::vector<double>& matrix,
                                    std::size_t rows,
                                    std::size_t p,
                                    const std::vector<double>& basis,
                                    const std::vector<double>& gram_inverse,
                                    std::size_t terms,
                                    std::vector<double>& out) {
    out.assign(rows * p, 0.0);
    std::vector<double> projected(rows * terms, 0.0);
    std::vector<double> projected_inv(rows * terms, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t term = 0; term < terms; ++term) {
            double sum = 0.0;
            for (std::size_t col = 0; col < p; ++col) {
                sum += matrix[row * p + col] * basis[col * terms + term];
            }
            projected[row * terms + term] = sum;
        }
        for (std::size_t term = 0; term < terms; ++term) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < terms; ++inner) {
                sum += projected[row * terms + inner] *
                       gram_inverse[inner * terms + term];
            }
            projected_inv[row * terms + term] = sum;
        }
        for (std::size_t col = 0; col < p; ++col) {
            double fitted = 0.0;
            for (std::size_t term = 0; term < terms; ++term) {
                fitted += projected_inv[row * terms + term] *
                          basis[col * terms + term];
            }
            out[row * p + col] = matrix[row * p + col] - fitted;
        }
    }
}

void apply_detrend_projection_left(const std::vector<double>& matrix,
                                   std::size_t p,
                                   std::size_t cols,
                                   const std::vector<double>& basis,
                                   const std::vector<double>& gram_inverse,
                                   std::size_t terms,
                                   std::vector<double>& out) {
    out.assign(p * cols, 0.0);
    std::vector<double> projected(terms * cols, 0.0);
    std::vector<double> projected_inv(terms * cols, 0.0);
    for (std::size_t term = 0; term < terms; ++term) {
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t row = 0; row < p; ++row) {
                sum += basis[row * terms + term] * matrix[row * cols + col];
            }
            projected[term * cols + col] = sum;
        }
    }
    for (std::size_t term = 0; term < terms; ++term) {
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < terms; ++inner) {
                sum += gram_inverse[term * terms + inner] *
                       projected[inner * cols + col];
            }
            projected_inv[term * cols + col] = sum;
        }
    }
    for (std::size_t row = 0; row < p; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            double fitted = 0.0;
            for (std::size_t term = 0; term < terms; ++term) {
                fitted += basis[row * terms + term] *
                          projected_inv[term * cols + col];
            }
            out[row * cols + col] = matrix[row * cols + col] - fitted;
        }
    }
}

[[nodiscard]] n4m_status_t transform_moments_by_detrend_operator(
    Context& ctx,
    const MomentStats& stats,
    const OperatorEntry& entry,
    MomentStats& out) {
    if (stats.n_samples <= 0 || stats.n_features <= 0 ||
        stats.n_targets <= 0) {
        ctx.set_error("AOM detrend moment transform requires positive moment dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    if (stats.x_sum.size() != p ||
        stats.y_sum.size() != q ||
        stats.xtx.size() != p * p ||
        stats.xty.size() != p * q ||
        stats.yty.size() != q * q) {
        ctx.set_error("AOM detrend moment transform received inconsistent buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> basis;
    std::vector<double> gram_inverse;
    n4m_status_t st = build_detrend_projection_basis(
        ctx, entry, p, basis, gram_inverse);
    if (st != N4M_OK) {
        return st;
    }
    const std::size_t terms = basis.size() / p;

    out = MomentStats{};
    out.n_samples = stats.n_samples;
    out.n_features = stats.n_features;
    out.n_targets = stats.n_targets;
    out.y_sum = stats.y_sum;
    out.yty = stats.yty;

    apply_detrend_projection_right(stats.x_sum, 1U, p, basis, gram_inverse,
                                   terms, out.x_sum);
    apply_detrend_projection_left(stats.xty, p, q, basis, gram_inverse,
                                  terms, out.xty);
    std::vector<double> left_xtx;
    apply_detrend_projection_left(stats.xtx, p, p, basis, gram_inverse,
                                  terms, left_xtx);
    apply_detrend_projection_right(left_xtx, p, p, basis, gram_inverse,
                                   terms, out.xtx);

    return recompute_centered_moments(ctx, out);
}

[[nodiscard]] n4m_status_t parse_aom_whittaker_lambda(Context& ctx,
                                                      const OperatorEntry& entry,
                                                      double& lambda) {
    lambda = 1000.0;
    if (!entry.params.empty()) {
        if (entry.params.size() != 1U) {
            ctx.set_error("AOM Whittaker expects zero params or one lambda param");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        lambda = entry.params[0];
    }
    if (!std::isfinite(lambda) || lambda <= 0.0) {
        ctx.set_error("AOM Whittaker lambda must be finite and positive");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t build_whittaker_factor(
    Context& ctx,
    std::size_t p,
    double lambda,
    std::vector<double>& lower,
    std::vector<double>& diag) {
    if (p == 0U ||
        p > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        ctx.set_error("AOM Whittaker moment transform feature count is invalid");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (p < 3U) {
        lower.clear();
        diag.clear();
        return N4M_OK;
    }

    const auto n = static_cast<std::int64_t>(p);
    std::vector<double> main_diag(p, 0.0);
    std::vector<double> super1(p, 0.0);
    std::vector<double> super2(p, 0.0);
    n4m_second_diff_penalty_pent5(n, lambda,
                                  main_diag.data(),
                                  super1.data(),
                                  super2.data());
    for (double& value : main_diag) {
        value += 1.0;
    }

    lower.assign(p * 2U, 0.0);
    diag.assign(p, 0.0);
    const n4m_status_t st = n4m_banded5_factor_into(
        lower.data(), diag.data(),
        main_diag.data(), super1.data(), super2.data(), n);
    if (st != N4M_OK) {
        ctx.set_error("failed to factor AOM Whittaker moment system");
        lower.clear();
        diag.clear();
        return st;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t solve_whittaker_right(
    Context& ctx,
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& matrix,
    std::size_t rows,
    std::size_t p,
    std::vector<double>& out) {
    if (matrix.size() != rows * p) {
        ctx.set_error("AOM Whittaker right solve received inconsistent buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (p < 3U) {
        out = matrix;
        return N4M_OK;
    }
    const auto n = static_cast<std::int64_t>(p);
    std::vector<double> rhs(p, 0.0);
    std::vector<double> solution(p, 0.0);
    out.assign(rows * p, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < p; ++col) {
            rhs[col] = matrix[row * p + col];
        }
        const n4m_status_t st = n4m_banded5_solve_into(
            lower.data(), diag.data(), n, rhs.data(), solution.data());
        if (st != N4M_OK) {
            ctx.set_error("failed to solve AOM Whittaker right moment system");
            out.clear();
            return st;
        }
        for (std::size_t col = 0; col < p; ++col) {
            out[row * p + col] = solution[col];
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t solve_whittaker_left(
    Context& ctx,
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& matrix,
    std::size_t p,
    std::size_t cols,
    std::vector<double>& out) {
    if (matrix.size() != p * cols) {
        ctx.set_error("AOM Whittaker left solve received inconsistent buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (p < 3U) {
        out = matrix;
        return N4M_OK;
    }
    const auto n = static_cast<std::int64_t>(p);
    std::vector<double> rhs(p, 0.0);
    std::vector<double> solution(p, 0.0);
    out.assign(p * cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = 0; row < p; ++row) {
            rhs[row] = matrix[row * cols + col];
        }
        const n4m_status_t st = n4m_banded5_solve_into(
            lower.data(), diag.data(), n, rhs.data(), solution.data());
        if (st != N4M_OK) {
            ctx.set_error("failed to solve AOM Whittaker left moment system");
            out.clear();
            return st;
        }
        for (std::size_t row = 0; row < p; ++row) {
            out[row * cols + col] = solution[row];
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t transform_moments_by_whittaker_operator(
    Context& ctx,
    const MomentStats& stats,
    const OperatorEntry& entry,
    MomentStats& out) {
    if (stats.n_samples <= 0 || stats.n_features <= 0 ||
        stats.n_targets <= 0) {
        ctx.set_error("AOM Whittaker moment transform requires positive moment dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    if (stats.x_sum.size() != p ||
        stats.y_sum.size() != q ||
        stats.xtx.size() != p * p ||
        stats.xty.size() != p * q ||
        stats.yty.size() != q * q) {
        ctx.set_error("AOM Whittaker moment transform received inconsistent buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    double lambda = 1000.0;
    n4m_status_t st = parse_aom_whittaker_lambda(ctx, entry, lambda);
    if (st != N4M_OK) {
        return st;
    }
    if (p < 3U) {
        out = stats;
        return N4M_OK;
    }

    std::vector<double> lower;
    std::vector<double> diag;
    st = build_whittaker_factor(ctx, p, lambda, lower, diag);
    if (st != N4M_OK) {
        return st;
    }

    out = MomentStats{};
    out.n_samples = stats.n_samples;
    out.n_features = stats.n_features;
    out.n_targets = stats.n_targets;
    out.y_sum = stats.y_sum;
    out.yty = stats.yty;

    st = solve_whittaker_right(ctx, lower, diag, stats.x_sum, 1U, p,
                               out.x_sum);
    if (st != N4M_OK) {
        return st;
    }
    st = solve_whittaker_left(ctx, lower, diag, stats.xty, p, q, out.xty);
    if (st != N4M_OK) {
        return st;
    }
    std::vector<double> left_xtx;
    st = solve_whittaker_left(ctx, lower, diag, stats.xtx, p, p, left_xtx);
    if (st != N4M_OK) {
        return st;
    }
    st = solve_whittaker_right(ctx, lower, diag, left_xtx, p, p, out.xtx);
    if (st != N4M_OK) {
        return st;
    }

    return recompute_centered_moments(ctx, out);
}

[[nodiscard]] n4m_status_t validate_banded_operator(
    Context& ctx,
    const BandedLinearOperator& op,
    std::size_t p) {
    if (op.n_features != static_cast<std::int64_t>(p) ||
        op.offsets.size() != p + 1U ||
        op.src_indices.size() != op.coeffs.size()) {
        ctx.set_error("AOM banded moment transform received inconsistent operator buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (op.offsets.empty() || op.offsets.front() != 0 ||
        op.offsets.back() != static_cast<std::int32_t>(op.src_indices.size())) {
        ctx.set_error("AOM banded moment transform received invalid offsets");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::size_t col = 0; col < p; ++col) {
        const auto begin = op.offsets[col];
        const auto end = op.offsets[col + 1U];
        if (begin < 0 || end < begin ||
            end > static_cast<std::int32_t>(op.src_indices.size())) {
            ctx.set_error("AOM banded moment transform offsets are not monotonic");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (std::int32_t e = begin; e < end; ++e) {
            const auto src = op.src_indices[static_cast<std::size_t>(e)];
            const double coeff = op.coeffs[static_cast<std::size_t>(e)];
            if (src < 0 || src >= static_cast<std::int32_t>(p) ||
                !std::isfinite(coeff)) {
                ctx.set_error("AOM banded moment transform entry is invalid");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t transform_moments_by_banded_operator(
    Context& ctx,
    const MomentStats& stats,
    const BandedLinearOperator& op,
    MomentStats& out) {
    if (stats.n_samples <= 0 || stats.n_features <= 0 ||
        stats.n_targets <= 0) {
        ctx.set_error("AOM banded moment transform requires positive moment dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    if (stats.x_sum.size() != p ||
        stats.y_sum.size() != q ||
        stats.xtx.size() != p * p ||
        stats.xty.size() != p * q ||
        stats.yty.size() != q * q) {
        ctx.set_error("AOM banded moment transform received inconsistent moment buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_status_t st = validate_banded_operator(ctx, op, p);
    if (st != N4M_OK) {
        return st;
    }

    out = MomentStats{};
    out.n_samples = stats.n_samples;
    out.n_features = stats.n_features;
    out.n_targets = stats.n_targets;
    out.y_sum = stats.y_sum;
    out.yty = stats.yty;
    out.x_sum.assign(p, 0.0);
    out.xtx.assign(p * p, 0.0);
    out.xty.assign(p * q, 0.0);

    for (std::size_t col = 0; col < p; ++col) {
        const auto begin = op.offsets[col];
        const auto end = op.offsets[col + 1U];
        for (std::int32_t e = begin; e < end; ++e) {
            const auto src =
                static_cast<std::size_t>(op.src_indices[static_cast<std::size_t>(e)]);
            const double coeff = op.coeffs[static_cast<std::size_t>(e)];
            out.x_sum[col] += stats.x_sum[src] * coeff;
            for (std::size_t target = 0; target < q; ++target) {
                out.xty[col * q + target] +=
                    stats.xty[src * q + target] * coeff;
            }
        }
    }

    std::vector<double> temp(p * p, 0.0);
    for (std::size_t col = 0; col < p; ++col) {
        const auto begin = op.offsets[col];
        const auto end = op.offsets[col + 1U];
        for (std::int32_t e = begin; e < end; ++e) {
            const auto src =
                static_cast<std::size_t>(op.src_indices[static_cast<std::size_t>(e)]);
            const double coeff = op.coeffs[static_cast<std::size_t>(e)];
            for (std::size_t row = 0; row < p; ++row) {
                temp[row * p + col] += stats.xtx[row * p + src] * coeff;
            }
        }
    }
    for (std::size_t row_out = 0; row_out < p; ++row_out) {
        const auto begin = op.offsets[row_out];
        const auto end = op.offsets[row_out + 1U];
        for (std::int32_t e = begin; e < end; ++e) {
            const auto src =
                static_cast<std::size_t>(op.src_indices[static_cast<std::size_t>(e)]);
            const double coeff = op.coeffs[static_cast<std::size_t>(e)];
            for (std::size_t col_out = 0; col_out < p; ++col_out) {
                out.xtx[row_out * p + col_out] +=
                    coeff * temp[src * p + col_out];
            }
        }
    }

    return recompute_centered_moments(ctx, out);
}

[[nodiscard]] n4m_status_t transform_moments_by_structured_chain(
    Context& ctx,
    const MomentStats& stats,
    const ChainSpec& chain,
    MomentStats& out,
    OperatorMomentRoute& route) {
    if (chain.ops.empty()) {
        ctx.set_error("AOM sweep chain must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    route = OperatorMomentRoute::kBanded;
    MomentStats current = stats;
    for (const auto& entry : chain.ops) {
        BandedLinearOperator op;
        n4m_status_t st = build_aom_banded_operator(
            ctx, entry, current.n_features, op);
        if (st == N4M_OK) {
            MomentStats next;
            st = transform_moments_by_banded_operator(ctx, current, op, next);
            if (st != N4M_OK) {
                return st;
            }
            current = std::move(next);
            continue;
        }
        if (st != N4M_ERR_UNSUPPORTED) {
            return st;
        }
        ctx.clear_error();

        if (entry.kind == N4M_OP_DETREND_POLY) {
            MomentStats next;
            st = transform_moments_by_detrend_operator(ctx, current, entry, next);
            if (st != N4M_OK) {
                return st;
            }
            current = std::move(next);
            route = OperatorMomentRoute::kStructured;
            continue;
        }
        if (entry.kind == N4M_OP_WHITTAKER) {
            MomentStats next;
            st = transform_moments_by_whittaker_operator(ctx, current, entry, next);
            if (st != N4M_OK) {
                return st;
            }
            current = std::move(next);
            route = OperatorMomentRoute::kStructured;
            continue;
        }
        {
            ctx.set_error("AOM operator is not supported by the structured moment route");
            return N4M_ERR_UNSUPPORTED;
        }
    }
    out = std::move(current);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t transform_moment_set_by_chain(
    Context& ctx,
    const MomentStats& all_base,
    const std::vector<MomentStats>& heldout_base,
    const ChainSpec& chain,
    bool allow_dense,
    bool allow_structured,
    MomentStats& all_transformed,
    std::vector<MomentStats>& heldout_transformed,
    OperatorMomentRoute& route) {
    route = OperatorMomentRoute::kNone;
    if (allow_structured) {
        n4m_status_t st = transform_moments_by_structured_chain(
            ctx, all_base, chain, all_transformed, route);
        if (st == N4M_OK) {
            const OperatorMomentRoute all_route = route;
            heldout_transformed.resize(heldout_base.size());
            for (std::size_t fold = 0; fold < heldout_base.size(); ++fold) {
                OperatorMomentRoute fold_route = OperatorMomentRoute::kNone;
                st = transform_moments_by_structured_chain(
                    ctx, heldout_base[fold], chain,
                    heldout_transformed[fold], fold_route);
                if (st != N4M_OK) {
                    return st;
                }
                if (fold_route != all_route) {
                    ctx.set_error("AOM structured moment route changed across folds");
                    return N4M_ERR_NUMERICAL_FAILURE;
                }
            }
            route = all_route;
            ctx.clear_error();
            return N4M_OK;
        }
        if (st != N4M_ERR_UNSUPPORTED) {
            return st;
        }
        ctx.clear_error();
    }

    if (!allow_dense) {
        return N4M_ERR_UNSUPPORTED;
    }

    std::vector<double> op_matrix;
    n4m_status_t st = build_chain_operator_matrix(
        ctx, chain, all_base.n_features, op_matrix);
    if (st != N4M_OK) {
        return st;
    }
    st = transform_moments_by_operator(
        ctx, all_base, op_matrix, all_transformed);
    if (st != N4M_OK) {
        return st;
    }

    heldout_transformed.resize(heldout_base.size());
    for (std::size_t fold = 0; fold < heldout_base.size(); ++fold) {
        st = transform_moments_by_operator(
            ctx, heldout_base[fold], op_matrix,
            heldout_transformed[fold]);
        if (st != N4M_OK) {
            return st;
        }
    }
    route = OperatorMomentRoute::kDense;
    return N4M_OK;
}

[[nodiscard]] MomentPrefixKey make_moment_prefix_key(
    const ChainSpec& chain,
    std::size_t n_ops) {
    MomentPrefixKey key;
    key.param_offsets.push_back(0);
    for (std::size_t op_ix = 0; op_ix < n_ops; ++op_ix) {
        const auto& entry = chain.ops[op_ix];
        key.op_kinds.push_back(static_cast<std::int32_t>(entry.kind));
        key.params.insert(key.params.end(), entry.params.begin(),
                          entry.params.end());
        key.param_offsets.push_back(
            static_cast<std::int32_t>(key.params.size()));
    }
    return key;
}

[[nodiscard]] bool moment_prefix_cache_can_run(const MomentStats& all_base,
                                               const MomentPrefixCache& cache) {
    return cache.enabled &&
           all_base.n_features > 0 &&
           static_cast<std::size_t>(all_base.n_features) <=
               kMaxMomentPrefixCacheFeatures;
}

[[nodiscard]] bool moment_prefix_cache_can_store(
    const MomentPrefixCache& cache) {
    return cache.entries.size() < kMaxMomentPrefixCacheEntries;
}

void copy_moment_prefix_cache_counters(const MomentPrefixCache& cache,
                                       AomSweepResult& out) {
    out.n_moment_prefix_cache_hits = cache.hits;
    out.n_moment_prefix_cache_misses = cache.misses;
}

[[nodiscard]] n4m_status_t transform_one_operator_moment_set(
    Context& ctx,
    const MomentStats& current_all,
    const std::vector<MomentStats>& current_heldout,
    const OperatorEntry& entry,
    OperatorMomentRoute current_route,
    MomentStats& next_all,
    std::vector<MomentStats>& next_heldout,
    OperatorMomentRoute& next_route) {
    BandedLinearOperator banded;
    n4m_status_t st = build_aom_banded_operator(
        ctx, entry, current_all.n_features, banded);
    if (st == N4M_OK) {
        st = transform_moments_by_banded_operator(
            ctx, current_all, banded, next_all);
        if (st != N4M_OK) return st;
        next_heldout.resize(current_heldout.size());
        for (std::size_t fold = 0; fold < current_heldout.size(); ++fold) {
            st = transform_moments_by_banded_operator(
                ctx, current_heldout[fold], banded, next_heldout[fold]);
            if (st != N4M_OK) return st;
        }
        next_route = current_route;
        return N4M_OK;
    }
    if (st != N4M_ERR_UNSUPPORTED) return st;
    ctx.clear_error();

    if (entry.kind == N4M_OP_DETREND_POLY) {
        st = transform_moments_by_detrend_operator(
            ctx, current_all, entry, next_all);
        if (st != N4M_OK) return st;
        next_heldout.resize(current_heldout.size());
        for (std::size_t fold = 0; fold < current_heldout.size(); ++fold) {
            st = transform_moments_by_detrend_operator(
                ctx, current_heldout[fold], entry, next_heldout[fold]);
            if (st != N4M_OK) return st;
        }
        next_route = OperatorMomentRoute::kStructured;
        return N4M_OK;
    }
    if (entry.kind == N4M_OP_WHITTAKER) {
        st = transform_moments_by_whittaker_operator(
            ctx, current_all, entry, next_all);
        if (st != N4M_OK) return st;
        next_heldout.resize(current_heldout.size());
        for (std::size_t fold = 0; fold < current_heldout.size(); ++fold) {
            st = transform_moments_by_whittaker_operator(
                ctx, current_heldout[fold], entry, next_heldout[fold]);
            if (st != N4M_OK) return st;
        }
        next_route = OperatorMomentRoute::kStructured;
        return N4M_OK;
    }

    ctx.set_error("AOM operator is not supported by the structured moment route");
    return N4M_ERR_UNSUPPORTED;
}

[[nodiscard]] n4m_status_t transform_moment_set_by_chain_cached(
    Context& ctx,
    const MomentStats& all_base,
    const std::vector<MomentStats>& heldout_base,
    const ChainSpec& chain,
    bool allow_dense,
    bool allow_structured,
    MomentPrefixCache& cache,
    MomentStats& all_transformed,
    std::vector<MomentStats>& heldout_transformed,
    OperatorMomentRoute& route) {
    if (!allow_structured || !moment_prefix_cache_can_run(all_base, cache)) {
        return transform_moment_set_by_chain(
            ctx, all_base, heldout_base, chain, allow_dense, allow_structured,
            all_transformed, heldout_transformed, route);
    }
    if (chain.ops.empty()) {
        ctx.set_error("AOM sweep chain must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    MomentStats current_all = all_base;
    std::vector<MomentStats> current_heldout = heldout_base;
    OperatorMomentRoute current_route = OperatorMomentRoute::kBanded;

    for (std::size_t op_ix = 0; op_ix < chain.ops.size(); ++op_ix) {
        MomentPrefixKey key = make_moment_prefix_key(chain, op_ix + 1U);
        const auto found = cache.entries.find(key);
        if (found != cache.entries.end()) {
            ++cache.hits;
            current_all = found->second.all;
            current_heldout = found->second.heldout;
            current_route = found->second.route;
            continue;
        }

        ++cache.misses;
        MomentStats next_all;
        std::vector<MomentStats> next_heldout;
        OperatorMomentRoute next_route = current_route;
        n4m_status_t st = transform_one_operator_moment_set(
            ctx, current_all, current_heldout, chain.ops[op_ix],
            current_route, next_all, next_heldout, next_route);
        if (st != N4M_OK) {
            if (st == N4M_ERR_UNSUPPORTED && allow_dense) {
                ctx.clear_error();
                return transform_moment_set_by_chain(
                    ctx, all_base, heldout_base, chain,
                    allow_dense, allow_structured,
                    all_transformed, heldout_transformed, route);
            }
            return st;
        }

        current_all = std::move(next_all);
        current_heldout = std::move(next_heldout);
        current_route = next_route;
        if (moment_prefix_cache_can_store(cache)) {
            cache.entries.emplace(std::move(key),
                                  CachedMomentSet{
                                      current_all,
                                      current_heldout,
                                      current_route,
                                  });
        }
    }

    all_transformed = std::move(current_all);
    heldout_transformed = std::move(current_heldout);
    route = current_route;
    return N4M_OK;
}

void account_candidate_route(AomSweepResult& out,
                             OperatorMomentRoute route,
                             std::int32_t head_id,
                             std::size_t n_candidates) {
    const auto n = static_cast<std::int64_t>(n_candidates);
    if (route == OperatorMomentRoute::kNone) {
        out.n_materialized_candidates += n;
        if (head_id == 0) {
            out.n_ridge_materialized_candidates += n;
        } else if (head_id == 1) {
            out.n_pls_materialized_candidates += n;
        }
        return;
    }
    out.n_operator_moment_candidates += n;
    if (head_id == 0) {
        out.n_ridge_operator_moment_candidates += n;
    } else if (head_id == 1) {
        out.n_pls_operator_moment_candidates += n;
    }
    if (route == OperatorMomentRoute::kBanded) {
        out.n_banded_operator_moment_candidates += n;
    } else if (route == OperatorMomentRoute::kStructured) {
        out.n_structured_operator_moment_candidates += n;
    } else if (route == OperatorMomentRoute::kDense) {
        out.n_dense_operator_moment_candidates += n;
    }
}

void append_candidate(AomSweepResult& out,
                      std::int64_t candidate_id,
                      std::int32_t chain_id,
                      double head_id,
                      double param,
                      double rmse,
                      OperatorMomentRoute route) {
    out.candidate_scores.push_back(static_cast<double>(candidate_id));
    out.candidate_scores.push_back(static_cast<double>(chain_id));
    out.candidate_scores.push_back(head_id);
    out.candidate_scores.push_back(param);
    out.candidate_scores.push_back(rmse);
    out.candidate_routes.push_back(candidate_route_code(route));
}

void account_materialized_sweep(AomSweepResult& out,
                                const SweepResult& sweep) {
    const auto n_rows = static_cast<std::size_t>(sweep.n_candidates);
    out.n_materialized_candidates += static_cast<std::int64_t>(n_rows);
    for (std::size_t row = 0; row < n_rows; ++row) {
        const auto head_id =
            static_cast<std::int32_t>(sweep.candidate_scores[row * 4 + 1]);
        if (head_id == 0) {
            ++out.n_ridge_materialized_candidates;
        } else if (head_id == 1) {
            ++out.n_pls_materialized_candidates;
        }
    }
    out.n_pls_moment_cv_fits += sweep.n_pls_moment_cv_fits;
    out.n_pls_moment_host_cv_fits += sweep.n_pls_moment_host_cv_fits;
    out.n_pls_moment_cuda_device_cv_fits +=
        sweep.n_pls_moment_cuda_device_cv_fits;
    out.n_pls_moment_cuda_parallel_fold_batches +=
        sweep.n_pls_moment_cuda_parallel_fold_batches;
    out.n_pls_moment_cuda_parallel_fold_jobs +=
        sweep.n_pls_moment_cuda_parallel_fold_jobs;
    out.n_pls_moment_cuda_many_batched_batches +=
        sweep.n_pls_moment_cuda_many_batched_batches;
    out.n_pls_moment_cuda_many_batched_jobs +=
        sweep.n_pls_moment_cuda_many_batched_jobs;
    out.n_ridge_moment_cv_fits += sweep.n_ridge_moment_cv_fits;
    out.n_ridge_moment_eigen_path_preparations +=
        sweep.n_ridge_moment_eigen_path_preparations;
    out.n_ridge_moment_eigen_path_cv_fits +=
        sweep.n_ridge_moment_eigen_path_cv_fits;
    out.n_ridge_moment_direct_cv_fits +=
        sweep.n_ridge_moment_direct_cv_fits;
    out.n_ridge_dual_materialized_cv_fits +=
        sweep.n_ridge_dual_materialized_cv_fits;
    out.n_ridge_dual_cross_cv_fits += sweep.n_ridge_dual_cross_cv_fits;
    out.n_ridge_moment_score_batch_calls +=
        sweep.n_ridge_moment_score_batch_calls;
    out.n_ridge_moment_score_batch_jobs +=
        sweep.n_ridge_moment_score_batch_jobs;
    out.n_ridge_moment_final_fits += sweep.n_ridge_moment_final_fits;
    out.n_ridge_dual_materialized_final_fits +=
        sweep.n_ridge_dual_materialized_final_fits;
    out.n_pls_moment_score_batch_calls +=
        sweep.n_pls_moment_score_batch_calls;
    out.n_pls_moment_score_batch_jobs +=
        sweep.n_pls_moment_score_batch_jobs;
    out.n_pls_materialized_cv_fits += sweep.n_pls_materialized_cv_fits;
    out.n_pls_gcv_proxy_candidates += sweep.n_pls_gcv_proxy_candidates;
    out.n_pls_gcv_proxy_fits += sweep.n_pls_gcv_proxy_fits;
    out.n_pls_gcv_proxy_batch_calls +=
        sweep.n_pls_gcv_proxy_batch_calls;
    out.n_pls_gcv_proxy_batch_jobs +=
        sweep.n_pls_gcv_proxy_batch_jobs;
    out.n_pls_moment_final_fits += sweep.n_pls_moment_final_fits;
    out.n_pls_moment_host_final_fits += sweep.n_pls_moment_host_final_fits;
    out.n_pls_moment_cuda_device_final_fits +=
        sweep.n_pls_moment_cuda_device_final_fits;
    out.n_pls_materialized_final_fits += sweep.n_pls_materialized_final_fits;
}

void account_ridge_fit_counters(AomSweepResult& out,
                                const SweepResult& sweep) {
    out.n_ridge_moment_cv_fits += sweep.n_ridge_moment_cv_fits;
    out.n_ridge_moment_eigen_path_preparations +=
        sweep.n_ridge_moment_eigen_path_preparations;
    out.n_ridge_moment_eigen_path_cv_fits +=
        sweep.n_ridge_moment_eigen_path_cv_fits;
    out.n_ridge_moment_direct_cv_fits +=
        sweep.n_ridge_moment_direct_cv_fits;
    out.n_ridge_dual_materialized_cv_fits +=
        sweep.n_ridge_dual_materialized_cv_fits;
    out.n_ridge_dual_cross_cv_fits += sweep.n_ridge_dual_cross_cv_fits;
    out.n_ridge_moment_score_batch_calls +=
        sweep.n_ridge_moment_score_batch_calls;
    out.n_ridge_moment_score_batch_jobs +=
        sweep.n_ridge_moment_score_batch_jobs;
    out.n_ridge_moment_final_fits += sweep.n_ridge_moment_final_fits;
    out.n_ridge_dual_materialized_final_fits +=
        sweep.n_ridge_dual_materialized_final_fits;
}

void account_pls_fit_counters(AomSweepResult& out,
                              const SweepResult& sweep) {
    out.n_pls_moment_cv_fits += sweep.n_pls_moment_cv_fits;
    out.n_pls_moment_host_cv_fits += sweep.n_pls_moment_host_cv_fits;
    out.n_pls_moment_cuda_device_cv_fits +=
        sweep.n_pls_moment_cuda_device_cv_fits;
    out.n_pls_moment_cuda_parallel_fold_batches +=
        sweep.n_pls_moment_cuda_parallel_fold_batches;
    out.n_pls_moment_cuda_parallel_fold_jobs +=
        sweep.n_pls_moment_cuda_parallel_fold_jobs;
    out.n_pls_moment_cuda_many_batched_batches +=
        sweep.n_pls_moment_cuda_many_batched_batches;
    out.n_pls_moment_cuda_many_batched_jobs +=
        sweep.n_pls_moment_cuda_many_batched_jobs;
    out.n_pls_moment_score_batch_calls +=
        sweep.n_pls_moment_score_batch_calls;
    out.n_pls_moment_score_batch_jobs +=
        sweep.n_pls_moment_score_batch_jobs;
    out.n_pls_materialized_cv_fits += sweep.n_pls_materialized_cv_fits;
    out.n_pls_gcv_proxy_candidates += sweep.n_pls_gcv_proxy_candidates;
    out.n_pls_gcv_proxy_fits += sweep.n_pls_gcv_proxy_fits;
    out.n_pls_gcv_proxy_batch_calls +=
        sweep.n_pls_gcv_proxy_batch_calls;
    out.n_pls_gcv_proxy_batch_jobs +=
        sweep.n_pls_gcv_proxy_batch_jobs;
    out.n_pls_moment_final_fits += sweep.n_pls_moment_final_fits;
    out.n_pls_moment_host_final_fits += sweep.n_pls_moment_host_final_fits;
    out.n_pls_moment_cuda_device_final_fits +=
        sweep.n_pls_moment_cuda_device_final_fits;
    out.n_pls_materialized_final_fits += sweep.n_pls_materialized_final_fits;
}

[[nodiscard]] n4m_status_t run_aom_ridge_operator_moment_sweep(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    const std::vector<ChainSpec>& chains,
    const std::vector<std::int32_t>& ids,
    std::int32_t actual_cv,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    AomSweepResult& out) {
    MomentStats all_base;
    n4m_status_t st = compute_moments(ctx, X, Y, all_base);
    if (st != N4M_OK) return st;

    std::vector<MomentStats> heldout_base(fold_rows.size());
    for (std::size_t fold = 0; fold < fold_rows.size(); ++fold) {
        const auto& rows = fold_rows[fold];
        st = compute_moments_subset(
            ctx, X, Y, rows.data(), static_cast<std::int64_t>(rows.size()),
            heldout_base[fold]);
        if (st != N4M_OK) return st;
    }

    if (aom_score_only_policy(cfg)) {
        std::vector<MomentStats> all_by_chain;
        std::vector<std::vector<MomentStats>> heldout_by_chain;
        std::vector<OperatorMomentRoute> routes;
        all_by_chain.reserve(chains.size());
        heldout_by_chain.reserve(chains.size());
        routes.reserve(chains.size());

        MomentPrefixCache batch_cache;
        batch_cache.enabled = true;
        const auto p = static_cast<std::size_t>(X.cols);
        bool all_supported = !should_materialize_cpu_wide_ridge(X, fold_rows);
        if (all_supported) {
            for (const auto& chain : chains) {
                MomentStats all_transformed;
                std::vector<MomentStats> heldout_transformed;
                OperatorMomentRoute route = OperatorMomentRoute::kNone;
                st = transform_moment_set_by_chain_cached(
                    ctx, all_base, heldout_base, chain,
                    true,
                    p <= kMaxBandedRidgeMomentFeatures,
                    batch_cache, all_transformed, heldout_transformed, route);
                if (st != N4M_OK) {
                    if (st != N4M_ERR_UNSUPPORTED) return st;
                    all_supported = false;
                    break;
                }
                all_by_chain.push_back(std::move(all_transformed));
                heldout_by_chain.push_back(std::move(heldout_transformed));
                routes.push_back(route);
            }
        }

        if (all_supported) {
            std::vector<SweepResult> sweeps;
            st = score_ridge_moment_sweeps_score_only(
                ctx, cfg, all_by_chain, heldout_by_chain,
                ridge_lambdas, n_ridge_lambdas, sweeps);
            if (st != N4M_OK) return st;
            if (sweeps.size() != chains.size()) {
                ctx.set_error("AOM ridge moment batch path returned inconsistent sweep count");
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            double best = std::numeric_limits<double>::infinity();
            bool have_best = false;
            std::int64_t global_candidate_id = 0;
            for (std::size_t chain_ix = 0; chain_ix < chains.size(); ++chain_ix) {
                const auto& chain = chains[chain_ix];
                const auto& sweep = sweeps[chain_ix];
                const auto route = routes[chain_ix];
                const auto local_candidates =
                    static_cast<std::size_t>(sweep.n_candidates);
                account_ridge_fit_counters(out, sweep);
                account_candidate_route(out, route, 0, local_candidates);
                for (std::size_t row = 0; row < local_candidates; ++row) {
                    const double local_id = sweep.candidate_scores[row * 4 + 0];
                    const double head_id = sweep.candidate_scores[row * 4 + 1];
                    const double param = sweep.candidate_scores[row * 4 + 2];
                    const double rmse = sweep.candidate_scores[row * 4 + 3];
                    append_candidate(out, global_candidate_id, chain.id, head_id,
                                     param, rmse, route);
                    if (std::isfinite(rmse) && rmse < best) {
                        best = rmse;
                        have_best = true;
                        out.selected_candidate_id = global_candidate_id;
                        out.selected_chain_id = chain.id;
                        out.selected_sweep_candidate_id =
                            static_cast<std::int64_t>(local_id);
                        out.selected_head_id =
                            static_cast<std::int32_t>(head_id);
                        out.selected_param = param;
                        out.selected_cv_rmse = rmse;
                    }
                    ++global_candidate_id;
                }
            }

            if (!have_best) {
                ctx.set_error("AOM ridge moment batch sweep: all candidates failed");
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            out.fold_ids = ids;
            out.cv = actual_cv;
            out.n_samples = X.rows;
            out.n_features = X.cols;
            out.n_targets = Y.cols;
            out.profile = profile;
            out.aom_pls_score_mode = cfg.aom_pls_score_mode;
            out.n_chains = static_cast<std::int64_t>(chains.size());
            out.n_candidates = global_candidate_id;
            out.score_only = 1;
            copy_moment_prefix_cache_counters(batch_cache, out);
            export_chain_descriptor(chains, out);
            ctx.clear_error();
            return N4M_OK;
        }

        if (force_operator_moments_policy(cfg)) {
            return reject_forced_moment_fallback(
                ctx, "Ridge chain is not eligible for operator-moment scoring");
        }
        ctx.clear_error();
    }

    double best = std::numeric_limits<double>::infinity();
    bool have_best = false;
    std::int64_t global_candidate_id = 0;
    std::size_t selected_chain_index = 0;
    MomentPrefixCache moment_cache;
    moment_cache.enabled = true;

    for (std::size_t chain_ix = 0; chain_ix < chains.size(); ++chain_ix) {
        const auto& chain = chains[chain_ix];
        MomentStats all_transformed;
        std::vector<MomentStats> heldout_transformed;
        OperatorMomentRoute route = OperatorMomentRoute::kNone;
        const auto p = static_cast<std::size_t>(X.cols);
        if (should_materialize_cpu_wide_ridge(X, fold_rows)) {
            st = N4M_ERR_UNSUPPORTED;
        } else {
            st = transform_moment_set_by_chain_cached(
                ctx, all_base, heldout_base, chain,
                true,
                p <= kMaxBandedRidgeMomentFeatures,
                moment_cache, all_transformed, heldout_transformed, route);
        }
        if (st != N4M_OK) {
            if (st != N4M_ERR_UNSUPPORTED) return st;
            if (force_operator_moments_policy(cfg)) {
                return reject_forced_moment_fallback(
                    ctx, "Ridge chain is not eligible for operator-moment scoring");
            }
            ctx.clear_error();
            std::vector<double> Xt;
            st = transform_chain(ctx, chain, X, Xt);
            if (st != N4M_OK) return st;
            n4m_matrix_view_t Xtv = rowmajor_view(Xt, X.rows, X.cols);
            SweepResult sweep;
            st = run_moment_sweep(ctx, cfg, Xtv, Y, cv, fold_ids, n_fold_ids,
                                  ridge_lambdas, n_ridge_lambdas,
                                  nullptr, 0, kSweepHeadRidge, sweep);
            if (st != N4M_OK) return st;

            const auto local_candidates =
                static_cast<std::size_t>(sweep.n_candidates);
            account_ridge_fit_counters(out, sweep);
            account_candidate_route(out, OperatorMomentRoute::kNone, 0,
                                    local_candidates);
            for (std::size_t row = 0; row < local_candidates; ++row) {
                const double local_id = sweep.candidate_scores[row * 4 + 0];
                const double head_id = sweep.candidate_scores[row * 4 + 1];
                const double param = sweep.candidate_scores[row * 4 + 2];
                const double rmse = sweep.candidate_scores[row * 4 + 3];
                append_candidate(out, global_candidate_id, chain.id, head_id,
                                 param, rmse, OperatorMomentRoute::kNone);
                if (std::isfinite(rmse) && rmse < best) {
                    best = rmse;
                    have_best = true;
                    selected_chain_index = chain_ix;
                    out.selected_candidate_id = global_candidate_id;
                    out.selected_chain_id = chain.id;
                    out.selected_sweep_candidate_id =
                        static_cast<std::int64_t>(local_id);
                    out.selected_head_id = static_cast<std::int32_t>(head_id);
                    out.selected_param = param;
                    out.selected_cv_rmse = rmse;
                }
                ++global_candidate_id;
            }
            continue;
        }

        SweepResult sweep;
        st = score_ridge_moment_sweep(ctx, cfg, all_transformed,
                                      heldout_transformed,
                                      ridge_lambdas, n_ridge_lambdas,
                                      sweep);
        if (st != N4M_OK) return st;

        const auto local_candidates = static_cast<std::size_t>(sweep.n_candidates);
        account_ridge_fit_counters(out, sweep);
        account_candidate_route(out, route, 0, local_candidates);
        for (std::size_t row = 0; row < local_candidates; ++row) {
            const double local_id = sweep.candidate_scores[row * 4 + 0];
            const double head_id = sweep.candidate_scores[row * 4 + 1];
            const double param = sweep.candidate_scores[row * 4 + 2];
            const double rmse = sweep.candidate_scores[row * 4 + 3];
            append_candidate(out, global_candidate_id, chain.id, head_id,
                             param, rmse, route);
            if (std::isfinite(rmse) && rmse < best) {
                best = rmse;
                have_best = true;
                selected_chain_index = chain_ix;
                out.selected_candidate_id = global_candidate_id;
                out.selected_chain_id = chain.id;
                out.selected_sweep_candidate_id = static_cast<std::int64_t>(local_id);
                out.selected_head_id = static_cast<std::int32_t>(head_id);
                out.selected_param = param;
                out.selected_cv_rmse = rmse;
            }
            ++global_candidate_id;
        }
    }

    if (!have_best) {
        ctx.set_error("AOM ridge moment sweep: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    if (aom_score_only_policy(cfg)) {
        out.fold_ids = ids;
        out.cv = actual_cv;
        out.n_samples = X.rows;
        out.n_features = X.cols;
        out.n_targets = Y.cols;
        out.profile = profile;
        out.aom_pls_score_mode = cfg.aom_pls_score_mode;
        out.n_chains = static_cast<std::int64_t>(chains.size());
        out.n_candidates = global_candidate_id;
        out.score_only = 1;
        copy_moment_prefix_cache_counters(moment_cache, out);
        export_chain_descriptor(chains, out);
        ctx.clear_error();
        return N4M_OK;
    }

    std::vector<double> Xt;
    st = transform_chain(ctx, chains[selected_chain_index], X, Xt);
    if (st != N4M_OK) return st;
    n4m_matrix_view_t Xtv = rowmajor_view(Xt, X.rows, X.cols);
    SweepResult selected_sweep;
    const double selected_lambda = out.selected_param;
    st = run_moment_sweep(ctx, cfg, Xtv, Y, cv, fold_ids, n_fold_ids,
                          &selected_lambda, 1, nullptr, 0,
                          kSweepHeadRidge, selected_sweep);
    if (st != N4M_OK) return st;
    account_ridge_fit_counters(out, selected_sweep);

    out.predictions = std::move(selected_sweep.predictions);
    out.oof_predictions = std::move(selected_sweep.oof_predictions);
    out.coefficients = std::move(selected_sweep.coefficients);
    st = fold_chain_coefficients_to_input_space(
        ctx, chains[selected_chain_index], X.cols, Y.cols, out.coefficients,
        out.input_coefficients);
    if (st != N4M_OK) return st;
    out.intercept = std::move(selected_sweep.intercept);
    out.x_mean = std::move(selected_sweep.x_mean);
    out.x_scale = std::move(selected_sweep.x_scale);
    out.y_mean = std::move(selected_sweep.y_mean);
    out.fold_ids = std::move(selected_sweep.fold_ids);
    out.cv = selected_sweep.cv;
    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.profile = profile;
    out.aom_pls_score_mode = cfg.aom_pls_score_mode;
    out.n_chains = static_cast<std::int64_t>(chains.size());
    out.n_candidates = global_candidate_id;
    copy_moment_prefix_cache_counters(moment_cache, out);
    export_chain_descriptor(chains, out);
    ctx.clear_error();
    return N4M_OK;
}

[[nodiscard]] n4m_status_t run_aom_pls_operator_moment_score_sweep(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    const std::vector<ChainSpec>& chains,
    const std::vector<std::int32_t>& ids,
    std::int32_t actual_cv,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    AomSweepResult& out) {
    if (!aom_score_only_policy(cfg)) {
        ctx.set_error("AOM PLS moment batch path requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const bool use_pls_gcv_proxy = aom_pls_gcv_proxy_mode(cfg);

    MomentStats all_base;
    n4m_status_t st = compute_moments(ctx, X, Y, all_base);
    if (st != N4M_OK) return st;

    std::vector<MomentStats> heldout_base;
    if (!use_pls_gcv_proxy) {
        heldout_base.resize(fold_rows.size());
        for (std::size_t fold = 0; fold < fold_rows.size(); ++fold) {
            const auto& rows = fold_rows[fold];
            st = compute_moments_subset(
                ctx, X, Y, rows.data(), static_cast<std::int64_t>(rows.size()),
                heldout_base[fold]);
            if (st != N4M_OK) return st;
        }
    }

    std::vector<MomentStats> all_by_chain;
    std::vector<std::vector<MomentStats>> heldout_by_chain;
    std::vector<OperatorMomentRoute> routes;
    all_by_chain.reserve(chains.size());
    heldout_by_chain.reserve(chains.size());
    routes.reserve(chains.size());

    MomentPrefixCache moment_cache;
    moment_cache.enabled = true;
    const auto p = static_cast<std::size_t>(X.cols);
    const auto min_train =
        min_train_rows(static_cast<std::size_t>(X.rows), fold_rows);
    const bool allow_dense = p <= min_train ||
                             p <= kMaxWideOperatorMomentFeatures;

    for (const auto& chain : chains) {
        MomentStats all_transformed;
        std::vector<MomentStats> heldout_transformed;
        OperatorMomentRoute route = OperatorMomentRoute::kNone;
        st = transform_moment_set_by_chain_cached(
            ctx, all_base, heldout_base, chain,
            allow_dense,
            p <= kMaxBandedOperatorMomentFeatures,
            moment_cache, all_transformed, heldout_transformed, route);
        if (st != N4M_OK) {
            if (st == N4M_ERR_UNSUPPORTED) {
                ctx.set_error("AOM PLS moment batch path requires every chain to have an operator-moment route");
            }
            return st;
        }
        all_by_chain.push_back(std::move(all_transformed));
        heldout_by_chain.push_back(std::move(heldout_transformed));
        routes.push_back(route);
    }

    std::vector<SweepResult> sweeps;
    if (use_pls_gcv_proxy) {
        st = score_pls1_gcv_moment_screens_score_only(
            ctx, cfg, all_by_chain,
            pls_components, n_pls_components, sweeps);
    } else {
        st = score_pls1_moment_sweeps_score_only(
            ctx, cfg, all_by_chain, heldout_by_chain,
            pls_components, n_pls_components, sweeps);
    }
    if (st != N4M_OK) return st;
    if (sweeps.size() != chains.size()) {
        ctx.set_error("AOM PLS moment batch path returned inconsistent sweep count");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    double best = std::numeric_limits<double>::infinity();
    bool have_best = false;
    std::int64_t global_candidate_id = 0;
    for (std::size_t chain_ix = 0; chain_ix < chains.size(); ++chain_ix) {
        const auto& chain = chains[chain_ix];
        const auto& sweep = sweeps[chain_ix];
        const auto route = routes[chain_ix];
        const auto local_candidates =
            static_cast<std::size_t>(sweep.n_candidates);
        account_pls_fit_counters(out, sweep);
        account_candidate_route(out, route, 1, local_candidates);
        const auto pls_base_global_id = global_candidate_id;
        for (std::size_t row = 0; row < local_candidates; ++row) {
            const double local_id = sweep.candidate_scores[row * 4 + 0];
            const double head_id = sweep.candidate_scores[row * 4 + 1];
            const double param = sweep.candidate_scores[row * 4 + 2];
            const double rmse = sweep.candidate_scores[row * 4 + 3];
            append_candidate(out, global_candidate_id, chain.id, head_id,
                             param, rmse, route);
            if (std::isfinite(rmse) && rmse < best) {
                best = rmse;
                have_best = true;
                out.selected_candidate_id = global_candidate_id;
                out.selected_chain_id = chain.id;
                out.selected_sweep_candidate_id =
                    static_cast<std::int64_t>(local_id);
                out.selected_head_id = static_cast<std::int32_t>(head_id);
                out.selected_param = param;
                out.selected_cv_rmse = rmse;
            }
            ++global_candidate_id;
        }
        if (std::isfinite(sweep.selected_cv_rmse) &&
            sweep.selected_cv_rmse < best) {
            best = sweep.selected_cv_rmse;
            have_best = true;
            out.selected_candidate_id =
                pls_base_global_id + sweep.selected_candidate_id;
            out.selected_chain_id = chain.id;
            out.selected_sweep_candidate_id = sweep.selected_candidate_id;
            out.selected_head_id = sweep.selected_head_id;
            out.selected_param = sweep.selected_param;
            out.selected_cv_rmse = sweep.selected_cv_rmse;
        }
    }

    if (!have_best) {
        ctx.set_error("AOM PLS moment batch sweep: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    out.fold_ids = ids;
    out.cv = actual_cv;
    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.profile = profile;
    out.aom_pls_score_mode = cfg.aom_pls_score_mode;
    out.n_chains = static_cast<std::int64_t>(chains.size());
    out.n_candidates = global_candidate_id;
    out.score_only = 1;
    copy_moment_prefix_cache_counters(moment_cache, out);
    export_chain_descriptor(chains, out);
    ctx.clear_error();
    return N4M_OK;
}

[[nodiscard]] n4m_status_t run_aom_hybrid_operator_moment_sweep(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::int32_t heads_mask,
    const std::vector<ChainSpec>& chains,
    const std::vector<std::int32_t>& ids,
    std::int32_t actual_cv,
    const std::vector<std::vector<std::int64_t>>& fold_rows,
    AomSweepResult& out) {
    MomentStats all_base;
    n4m_status_t st = compute_moments(ctx, X, Y, all_base);
    if (st != N4M_OK) return st;

    std::vector<MomentStats> heldout_base(fold_rows.size());
    for (std::size_t fold = 0; fold < fold_rows.size(); ++fold) {
        const auto& rows = fold_rows[fold];
        st = compute_moments_subset(
            ctx, X, Y, rows.data(), static_cast<std::int64_t>(rows.size()),
            heldout_base[fold]);
        if (st != N4M_OK) return st;
    }

    double best = std::numeric_limits<double>::infinity();
    bool have_best = false;
    bool selected_is_ridge = false;
    std::int64_t global_candidate_id = 0;
    std::size_t selected_chain_index = 0;
    const bool do_ridge = (heads_mask & kSweepHeadRidge) != 0;
    const bool do_pls = (heads_mask & kSweepHeadPls) != 0;
    const bool force_operator_moments = force_operator_moments_policy(cfg);
    const bool use_pls_gcv_proxy = aom_pls_gcv_proxy_mode(cfg);
    if (use_pls_gcv_proxy && do_pls && !aom_score_only_policy(cfg)) {
        ctx.set_error("AOM PLS GCV proxy score mode requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const bool materialize_cpu_wide_ridge =
        should_materialize_cpu_wide_ridge(X, fold_rows);
    const bool materialize_cpu_wide_pls =
        use_pls_gcv_proxy ? false : should_materialize_cpu_wide_pls(X, fold_rows);
    const bool any_head_can_use_transformed_moments =
        (do_ridge && !materialize_cpu_wide_ridge) ||
        (do_pls && !materialize_cpu_wide_pls);
    MomentPrefixCache moment_cache;
    moment_cache.enabled = true;

    for (std::size_t chain_ix = 0; chain_ix < chains.size(); ++chain_ix) {
        const auto& chain = chains[chain_ix];
        MomentStats all_transformed;
        std::vector<MomentStats> heldout_transformed;
        OperatorMomentRoute route = OperatorMomentRoute::kNone;
        const auto p = static_cast<std::size_t>(X.cols);
        const auto min_train =
            min_train_rows(static_cast<std::size_t>(X.rows), fold_rows);
        const bool allow_dense = p <= min_train ||
                                 p <= kMaxWideOperatorMomentFeatures;
        bool have_transformed_moments = false;
        if (any_head_can_use_transformed_moments) {
            st = transform_moment_set_by_chain_cached(
                ctx, all_base, heldout_base, chain,
                allow_dense,
                p <= kMaxBandedOperatorMomentFeatures,
                moment_cache, all_transformed, heldout_transformed, route);
            have_transformed_moments = (st == N4M_OK);
            if (st != N4M_OK) {
                if (st != N4M_ERR_UNSUPPORTED) return st;
                ctx.clear_error();
            }
        }

        std::vector<double> Xt;
        n4m_matrix_view_t Xtv{};
        bool have_materialized = false;
        auto materialize_chain = [&]() -> n4m_status_t {
            if (!have_materialized) {
                n4m_status_t local_status = transform_chain(ctx, chain, X, Xt);
                if (local_status != N4M_OK) {
                    return local_status;
                }
                Xtv = rowmajor_view(Xt, X.rows, X.cols);
                have_materialized = true;
            }
            return N4M_OK;
        };

        std::size_t ridge_candidates = 0;
        if (do_ridge) {
            SweepResult ridge_sweep;
            OperatorMomentRoute ridge_route = OperatorMomentRoute::kNone;
            const bool ridge_moment_allowed =
                have_transformed_moments &&
                !materialize_cpu_wide_ridge &&
                ((route == OperatorMomentRoute::kDense &&
                  can_use_operator_moment_ridge_features(
                      cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas)) ||
                 (route == OperatorMomentRoute::kBanded &&
                  can_use_banded_operator_moment_ridge_features(
                      cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas)) ||
                 (route == OperatorMomentRoute::kStructured &&
                  can_use_banded_operator_moment_ridge_features(
                      cfg, X, fold_rows, ridge_lambdas, n_ridge_lambdas)));
            if (ridge_moment_allowed) {
                st = score_ridge_moment_sweep(ctx, cfg, all_transformed,
                                              heldout_transformed,
                                              ridge_lambdas, n_ridge_lambdas,
                                              ridge_sweep);
                if (st != N4M_OK) return st;
                ridge_route = route;
            } else {
                if (force_operator_moments) {
                    return reject_forced_moment_fallback(
                        ctx, "Ridge candidate is not eligible for operator-moment scoring");
                }
                st = materialize_chain();
                if (st != N4M_OK) return st;
                st = run_moment_sweep(ctx, cfg, Xtv, Y, cv, fold_ids, n_fold_ids,
                                      ridge_lambdas, n_ridge_lambdas,
                                      nullptr, 0, kSweepHeadRidge,
                                      ridge_sweep);
                if (st != N4M_OK) return st;
            }

            ridge_candidates = static_cast<std::size_t>(ridge_sweep.n_candidates);
            account_ridge_fit_counters(out, ridge_sweep);
            account_candidate_route(out, ridge_route, 0, ridge_candidates);
            for (std::size_t row = 0; row < ridge_candidates; ++row) {
                const double local_id = ridge_sweep.candidate_scores[row * 4 + 0];
                const double head_id = ridge_sweep.candidate_scores[row * 4 + 1];
                const double param = ridge_sweep.candidate_scores[row * 4 + 2];
                const double rmse = ridge_sweep.candidate_scores[row * 4 + 3];
                append_candidate(out, global_candidate_id, chain.id, head_id,
                                 param, rmse, ridge_route);
                if (std::isfinite(rmse) && rmse < best) {
                    best = rmse;
                    have_best = true;
                    selected_is_ridge = true;
                    selected_chain_index = chain_ix;
                    out.selected_candidate_id = global_candidate_id;
                    out.selected_chain_id = chain.id;
                    out.selected_sweep_candidate_id = static_cast<std::int64_t>(local_id);
                    out.selected_head_id = static_cast<std::int32_t>(head_id);
                    out.selected_param = param;
                    out.selected_cv_rmse = rmse;
                }
                ++global_candidate_id;
            }
        }

        if (do_pls) {
            SweepResult pls_sweep;
            OperatorMomentRoute pls_route = OperatorMomentRoute::kNone;
            if (have_transformed_moments && !materialize_cpu_wide_pls) {
                if (use_pls_gcv_proxy) {
                    st = score_pls1_gcv_moment_screen(
                        ctx, cfg, all_transformed,
                        pls_components, n_pls_components,
                        pls_sweep);
                } else {
                    st = score_pls1_moment_sweep(ctx, cfg, all_transformed,
                                                 heldout_transformed,
                                                 pls_components, n_pls_components,
                                                 pls_sweep);
                }
                if (st == N4M_OK) {
                    pls_route = route;
                }
            } else {
                st = N4M_ERR_UNSUPPORTED;
            }
            if (st != N4M_OK) {
                if (force_operator_moments || use_pls_gcv_proxy) {
                    return reject_forced_moment_fallback(
                        ctx, "PLS candidate is not eligible for operator-moment scoring");
                }
                ctx.clear_error();
                st = materialize_chain();
                if (st != N4M_OK) return st;
                st = run_moment_sweep(ctx, cfg, Xtv, Y, cv,
                                      fold_ids, n_fold_ids,
                                      nullptr, 0,
                                      pls_components, n_pls_components,
                                      kSweepHeadPls, pls_sweep);
                if (st != N4M_OK) return st;
            }

            const auto pls_base_global_id = global_candidate_id;
            const auto pls_candidates =
                static_cast<std::size_t>(pls_sweep.n_candidates);
            account_pls_fit_counters(out, pls_sweep);
            account_candidate_route(out, pls_route, 1, pls_candidates);
            for (std::size_t row = 0; row < pls_candidates; ++row) {
                const double head_id = pls_sweep.candidate_scores[row * 4 + 1];
                const double param = pls_sweep.candidate_scores[row * 4 + 2];
                const double rmse = pls_sweep.candidate_scores[row * 4 + 3];
                append_candidate(out, global_candidate_id, chain.id, head_id,
                                 param, rmse, pls_route);
                ++global_candidate_id;
            }
            if (std::isfinite(pls_sweep.selected_cv_rmse) &&
                pls_sweep.selected_cv_rmse < best) {
                best = pls_sweep.selected_cv_rmse;
                have_best = true;
                selected_is_ridge = false;
                selected_chain_index = chain_ix;
                out.selected_candidate_id =
                    pls_base_global_id + pls_sweep.selected_candidate_id;
                out.selected_chain_id = chain.id;
                out.selected_sweep_candidate_id =
                    static_cast<std::int64_t>(ridge_candidates) +
                    pls_sweep.selected_candidate_id;
                out.selected_head_id = pls_sweep.selected_head_id;
                out.selected_param = pls_sweep.selected_param;
                out.selected_cv_rmse = pls_sweep.selected_cv_rmse;
            }
        }
    }

    if (!have_best) {
        ctx.set_error("AOM hybrid moment sweep: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    if (aom_score_only_policy(cfg)) {
        out.fold_ids = ids;
        out.cv = actual_cv;
        out.n_samples = X.rows;
        out.n_features = X.cols;
        out.n_targets = Y.cols;
        out.profile = profile;
        out.aom_pls_score_mode = cfg.aom_pls_score_mode;
        out.n_chains = static_cast<std::int64_t>(chains.size());
        out.n_candidates = global_candidate_id;
        out.score_only = 1;
        copy_moment_prefix_cache_counters(moment_cache, out);
        export_chain_descriptor(chains, out);
        ctx.clear_error();
        return N4M_OK;
    }

    {
        std::vector<double> Xt;
        st = transform_chain(ctx, chains[selected_chain_index], X, Xt);
        if (st != N4M_OK) return st;
        n4m_matrix_view_t Xtv = rowmajor_view(Xt, X.rows, X.cols);
        SweepResult selected_sweep;
        std::int32_t selected_component = 0;
        const double selected_lambda = out.selected_param;
        if (selected_is_ridge) {
            st = run_moment_sweep(ctx, cfg, Xtv, Y, cv, fold_ids, n_fold_ids,
                                  &selected_lambda, 1, nullptr, 0,
                                  kSweepHeadRidge, selected_sweep);
        } else {
            selected_component = static_cast<std::int32_t>(out.selected_param);
            st = run_moment_sweep(ctx, cfg, Xtv, Y, cv, fold_ids, n_fold_ids,
                                  nullptr, 0, &selected_component, 1,
                                  kSweepHeadPls, selected_sweep);
        }
    if (st != N4M_OK) return st;
    account_ridge_fit_counters(out, selected_sweep);
    account_pls_fit_counters(out, selected_sweep);

        out.predictions = std::move(selected_sweep.predictions);
        out.oof_predictions = std::move(selected_sweep.oof_predictions);
        out.coefficients = std::move(selected_sweep.coefficients);
        st = fold_chain_coefficients_to_input_space(
            ctx, chains[selected_chain_index], X.cols, Y.cols, out.coefficients,
            out.input_coefficients);
        if (st != N4M_OK) return st;
        out.intercept = std::move(selected_sweep.intercept);
        out.x_mean = std::move(selected_sweep.x_mean);
        out.x_scale = std::move(selected_sweep.x_scale);
        out.y_mean = std::move(selected_sweep.y_mean);
        out.fold_ids = std::move(selected_sweep.fold_ids);
        out.cv = selected_sweep.cv;
    }

    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.profile = profile;
    out.aom_pls_score_mode = cfg.aom_pls_score_mode;
    out.n_chains = static_cast<std::int64_t>(chains.size());
    out.n_candidates = global_candidate_id;
    copy_moment_prefix_cache_counters(moment_cache, out);
    export_chain_descriptor(chains, out);
    ctx.clear_error();
    return N4M_OK;
}

[[nodiscard]] n4m_status_t run_aom_sweep_over_chains(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::int32_t heads_mask,
    const std::vector<ChainSpec>& chains,
    AomSweepResult& out) {
    if (chains.empty()) {
        ctx.set_error("AOM sweep requires at least one chain");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<std::int32_t> ids;
    std::int32_t actual_cv = 0;
    n4m_status_t fold_status = build_aom_fold_ids(
        ctx, X.rows, cv, fold_ids, n_fold_ids, ids, actual_cv);
    const bool force_operator_moments = force_operator_moments_policy(cfg);
    const bool use_pls_gcv_proxy =
        aom_pls_gcv_proxy_mode(cfg) &&
        ((heads_mask & kSweepHeadPls) != 0);
    if (use_pls_gcv_proxy && !aom_score_only_policy(cfg)) {
        ctx.set_error("AOM PLS GCV proxy score mode requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const bool allow_operator_moments =
        cfg.aom_moment_policy !=
        static_cast<std::int32_t>(N4M_AOM_MOMENT_MATERIALIZED);
    if (allow_operator_moments && fold_status == N4M_OK) {
        const auto fold_rows = aom_fold_rows_from_ids(ids, actual_cv);
        const bool cpu_wide_ridge_materialized =
            should_materialize_cpu_wide_ridge(X, fold_rows);
        const bool cpu_wide_pls_materialized =
            use_pls_gcv_proxy ? false : should_materialize_cpu_wide_pls(X, fold_rows);
        const bool ridge_path =
            !cpu_wide_ridge_materialized &&
            can_use_operator_moment_ridge_path(
                cfg, X, fold_rows, heads_mask,
                ridge_lambdas, n_ridge_lambdas);
        if (ridge_path) {
            out = AomSweepResult{};
            return run_aom_ridge_operator_moment_sweep(
                ctx, cfg, X, Y, profile, cv, fold_ids, n_fold_ids,
                ridge_lambdas, n_ridge_lambdas, chains, ids, actual_cv,
                fold_rows, out);
        }
        const bool ridge_part =
            !cpu_wide_ridge_materialized &&
            can_use_operator_moment_ridge_part(
                cfg, X, fold_rows, heads_mask,
                ridge_lambdas, n_ridge_lambdas);
        const bool pls_part =
            !cpu_wide_pls_materialized &&
            can_use_operator_moment_pls_part(
                cfg, X, Y, fold_rows, heads_mask);
        const bool request_ridge = (heads_mask & kSweepHeadRidge) != 0;
        if (!request_ridge && pls_part && aom_score_only_policy(cfg)) {
            out = AomSweepResult{};
            const n4m_status_t pls_batch_status =
                run_aom_pls_operator_moment_score_sweep(
                    ctx, cfg, X, Y, profile,
                    pls_components, n_pls_components,
                    chains, ids, actual_cv, fold_rows, out);
            if (pls_batch_status == N4M_OK) {
                return N4M_OK;
            }
            if (force_operator_moments &&
                pls_batch_status == N4M_ERR_UNSUPPORTED) {
                return reject_forced_moment_fallback(
                    ctx, "PLS candidate is not eligible for operator-moment scoring");
            }
            if (pls_batch_status != N4M_ERR_UNSUPPORTED) {
                return pls_batch_status;
            }
            ctx.clear_error();
        }
        if (ridge_part || (!request_ridge && pls_part)) {
            out = AomSweepResult{};
            return run_aom_hybrid_operator_moment_sweep(
                ctx, cfg, X, Y, profile, cv, fold_ids, n_fold_ids,
                ridge_lambdas, n_ridge_lambdas,
                pls_components, n_pls_components,
                heads_mask,
                chains, ids, actual_cv, fold_rows, out);
        }
    } else if (fold_status != N4M_OK) {
        if (force_operator_moments) return fold_status;
        ctx.clear_error();
    }

    if (force_operator_moments || use_pls_gcv_proxy) {
        return reject_forced_moment_fallback(
            ctx, "requested chain/head grid has no complete operator-moment route");
    }

    double best = std::numeric_limits<double>::infinity();
    std::int64_t global_candidate_id = 0;
    bool have_best = false;

    for (const auto& chain : chains) {
        std::vector<double> Xt;
        n4m_status_t st = transform_chain(ctx, chain, X, Xt);
        if (st != N4M_OK) return st;
        n4m_matrix_view_t Xtv = rowmajor_view(Xt, X.rows, X.cols);

        SweepResult sweep;
        st = run_moment_sweep(ctx, cfg, Xtv, Y, cv, fold_ids, n_fold_ids,
                              ridge_lambdas, n_ridge_lambdas,
                              pls_components, n_pls_components,
                              heads_mask, sweep);
        if (st != N4M_OK) return st;

        const auto local_candidates = static_cast<std::size_t>(sweep.n_candidates);
        account_materialized_sweep(out, sweep);
        for (std::size_t row = 0; row < local_candidates; ++row) {
            const double local_id = sweep.candidate_scores[row * 4 + 0];
            const double head_id = sweep.candidate_scores[row * 4 + 1];
            const double param = sweep.candidate_scores[row * 4 + 2];
            const double rmse = sweep.candidate_scores[row * 4 + 3];
            append_candidate(out, global_candidate_id, chain.id, head_id,
                             param, rmse, OperatorMomentRoute::kNone);
            if (std::isfinite(rmse) && rmse < best) {
                best = rmse;
                have_best = true;
                out.selected_candidate_id = global_candidate_id;
                out.selected_chain_id = chain.id;
                out.selected_sweep_candidate_id = static_cast<std::int64_t>(local_id);
                out.selected_head_id = static_cast<std::int32_t>(head_id);
                out.selected_param = param;
                out.selected_cv_rmse = rmse;
                if (!aom_score_only_policy(cfg)) {
                    out.predictions = sweep.predictions;
                    out.oof_predictions = sweep.oof_predictions;
                    out.coefficients = sweep.coefficients;
                    st = fold_chain_coefficients_to_input_space(
                        ctx, chain, X.cols, Y.cols, out.coefficients,
                        out.input_coefficients);
                    if (st != N4M_OK) return st;
                    out.intercept = sweep.intercept;
                    out.x_mean = sweep.x_mean;
                    out.x_scale = sweep.x_scale;
                    out.y_mean = sweep.y_mean;
                }
                out.fold_ids = sweep.fold_ids;
                out.cv = sweep.cv;
            }
            ++global_candidate_id;
        }
    }

    if (!have_best) {
        ctx.set_error("AOM sweep: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.profile = profile;
    out.aom_pls_score_mode = cfg.aom_pls_score_mode;
    if (out.cv == 0) out.cv = cv;
    out.n_chains = static_cast<std::int64_t>(chains.size());
    out.n_candidates = global_candidate_id;
    out.score_only = aom_score_only_policy(cfg) ? 1 : 0;
    export_chain_descriptor(chains, out);
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace

n4m_status_t run_aom_sweep(Context& ctx,
                           const Config& cfg,
                           const n4m_matrix_view_t& X,
                           const n4m_matrix_view_t& Y,
                           std::int32_t profile,
                           std::int32_t cv,
                           const std::int32_t* fold_ids,
                           std::int64_t n_fold_ids,
                           const double* ridge_lambdas,
                           std::int64_t n_ridge_lambdas,
                           const std::int32_t* pls_components,
                           std::int64_t n_pls_components,
                           std::int32_t heads_mask,
                           AomSweepResult& out) {
    out = AomSweepResult{};
    if (profile != static_cast<std::int32_t>(AomRobustHpoProfile::kCompact) &&
        profile != static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        ctx.set_error("AOM sweep profile must be 0=compact or 1=wide");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;

    const auto chains = build_bank(profile);
    return run_aom_sweep_over_chains(ctx, cfg, X, Y, profile, cv,
                                     fold_ids, n_fold_ids,
                                     ridge_lambdas, n_ridge_lambdas,
                                     pls_components, n_pls_components,
                                     heads_mask, chains, out);
}

n4m_status_t run_aom_chain_sweep(Context& ctx,
                                 const Config& cfg,
                                 const n4m_matrix_view_t& X,
                                 const n4m_matrix_view_t& Y,
                                 std::int32_t cv,
                                 const std::int32_t* fold_ids,
                                 std::int64_t n_fold_ids,
                                 const std::int32_t* chain_offsets,
                                 std::int64_t n_chain_offsets,
                                 const std::int32_t* op_kinds,
                                 std::int64_t n_op_kinds,
                                 const std::int32_t* param_offsets,
                                 std::int64_t n_param_offsets,
                                 const double* params,
                                 std::int64_t n_params,
                                 const double* ridge_lambdas,
                                 std::int64_t n_ridge_lambdas,
                                 const std::int32_t* pls_components,
                                 std::int64_t n_pls_components,
                                 std::int32_t heads_mask,
                                 AomSweepResult& out) {
    out = AomSweepResult{};
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;

    std::vector<ChainSpec> chains;
    st = build_chains_from_descriptor(ctx, chain_offsets, n_chain_offsets,
                                      op_kinds, n_op_kinds,
                                      param_offsets, n_param_offsets,
                                      params, n_params, chains);
    if (st != N4M_OK) return st;

    return run_aom_sweep_over_chains(ctx, cfg, X, Y, -1, cv,
                                     fold_ids, n_fold_ids,
                                     ridge_lambdas, n_ridge_lambdas,
                                     pls_components, n_pls_components,
                                     heads_mask, chains, out);
}

n4m_status_t run_aom_chain_fixed_fit(Context& ctx,
                                     const Config& cfg,
                                     const n4m_matrix_view_t& X,
                                     const n4m_matrix_view_t& Y,
                                     const std::int32_t* chain_offsets,
                                     std::int64_t n_chain_offsets,
                                     const std::int32_t* op_kinds,
                                     std::int64_t n_op_kinds,
                                     const std::int32_t* param_offsets,
                                     std::int64_t n_param_offsets,
                                     const double* params,
                                     std::int64_t n_params,
                                     std::int32_t head_id,
                                     double param,
                                     AomSweepResult& out) {
    out = AomSweepResult{};
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;
    if (head_id != 0 && head_id != 1) {
        ctx.set_error("AOM fixed fit head_id must be 0=Ridge or 1=PLS");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<ChainSpec> chains;
    st = build_chains_from_descriptor(ctx, chain_offsets, n_chain_offsets,
                                      op_kinds, n_op_kinds,
                                      param_offsets, n_param_offsets,
                                      params, n_params, chains);
    if (st != N4M_OK) return st;
    if (chains.size() != 1U) {
        ctx.set_error("AOM fixed fit requires exactly one chain");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> Xt;
    st = transform_chain(ctx, chains[0], X, Xt);
    if (st != N4M_OK) return st;
    n4m_matrix_view_t Xtv = rowmajor_view(Xt, X.rows, X.cols);

    SweepResult fit;
    st = run_moment_final_fit(ctx, cfg, Xtv, Y, head_id, param, fit);
    if (st != N4M_OK) return st;

    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.profile = -1;
    out.cv = 0;
    out.n_chains = 1;
    out.n_candidates = 1;
    out.selected_candidate_id = 0;
    out.selected_chain_id = chains[0].id;
    out.selected_sweep_candidate_id = 0;
    out.selected_head_id = head_id;
    out.selected_param = param;
    out.selected_cv_rmse = fit.selected_cv_rmse;
    out.score_only = 0;
    out.aom_pls_score_mode = cfg.aom_pls_score_mode;
    out.n_pls_moment_cv_fits = fit.n_pls_moment_cv_fits;
    out.n_pls_moment_host_cv_fits = fit.n_pls_moment_host_cv_fits;
    out.n_pls_moment_cuda_device_cv_fits =
        fit.n_pls_moment_cuda_device_cv_fits;
    out.n_pls_moment_cuda_parallel_fold_batches =
        fit.n_pls_moment_cuda_parallel_fold_batches;
    out.n_pls_moment_cuda_parallel_fold_jobs =
        fit.n_pls_moment_cuda_parallel_fold_jobs;
    out.n_pls_moment_cuda_many_batched_batches =
        fit.n_pls_moment_cuda_many_batched_batches;
    out.n_pls_moment_cuda_many_batched_jobs =
        fit.n_pls_moment_cuda_many_batched_jobs;
    out.n_pls_materialized_cv_fits = fit.n_pls_materialized_cv_fits;
    out.n_pls_gcv_proxy_candidates = fit.n_pls_gcv_proxy_candidates;
    out.n_pls_gcv_proxy_fits = fit.n_pls_gcv_proxy_fits;
    out.n_pls_moment_score_batch_calls = fit.n_pls_moment_score_batch_calls;
    out.n_pls_moment_score_batch_jobs = fit.n_pls_moment_score_batch_jobs;
    out.n_pls_gcv_proxy_batch_calls = fit.n_pls_gcv_proxy_batch_calls;
    out.n_pls_gcv_proxy_batch_jobs = fit.n_pls_gcv_proxy_batch_jobs;
    out.n_pls_moment_final_fits = fit.n_pls_moment_final_fits;
    out.n_pls_moment_host_final_fits = fit.n_pls_moment_host_final_fits;
    out.n_pls_moment_cuda_device_final_fits =
        fit.n_pls_moment_cuda_device_final_fits;
    out.n_pls_materialized_final_fits = fit.n_pls_materialized_final_fits;
    out.candidate_scores = {
        0.0,
        static_cast<double>(chains[0].id),
        static_cast<double>(head_id),
        param,
        out.selected_cv_rmse,
    };
    out.candidate_routes = {candidate_route_code(OperatorMomentRoute::kNone)};
    out.predictions = std::move(fit.predictions);
    out.oof_predictions = std::move(fit.oof_predictions);
    out.coefficients = std::move(fit.coefficients);
    st = fold_chain_coefficients_to_input_space(
        ctx, chains[0], X.cols, Y.cols, out.coefficients,
        out.input_coefficients);
    if (st != N4M_OK) return st;
    out.intercept = std::move(fit.intercept);
    out.x_mean = std::move(fit.x_mean);
    out.x_scale = std::move(fit.x_scale);
    out.y_mean = std::move(fit.y_mean);
    export_chain_descriptor(chains, out);
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
