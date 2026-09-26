// SPDX-License-Identifier: CECILL-2.1
//
// Estimator adapters of the AOM / POP family. The sweeps, the AOM / POP-PLS
// selectors, the Ridge blender, the operator PLS stack and the robust-HPO
// screen fold their selected preprocessing into the input space, so their
// fitted state is that affine predictor (N4MM). The branch calibration keeps
// its coefficient state and predicts through its own kernel (its SNV / MSC
// branches are not affine); AOM preprocessing is a stateless transformer.
// Fit and predict run the AOM kernels; the adapters map parameters to their
// arguments and frame the state.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/method_result.hpp"

namespace n4m::estimator {

namespace {

struct BankDeleter {
    void operator()(n4m_operator_bank_t* b) const noexcept { n4m_operator_bank_destroy(b); }
};
struct GateDeleter {
    void operator()(n4m_gating_strategy_t* g) const noexcept { n4m_gating_strategy_destroy(g); }
};
struct PlanDeleter {
    void operator()(n4m_validation_plan_t* p) const noexcept { n4m_validation_plan_destroy(p); }
};
struct ResultDeleter {
    void operator()(n4m_method_result_t* r) const noexcept { n4m_method_result_destroy(r); }
};
using BankPtr = std::unique_ptr<n4m_operator_bank_t, BankDeleter>;
using GatePtr = std::unique_ptr<n4m_gating_strategy_t, GateDeleter>;
using PlanPtr = std::unique_ptr<n4m_validation_plan_t, PlanDeleter>;
using ResultPtr = std::unique_ptr<n4m_method_result_t, ResultDeleter>;

std::vector<std::int32_t> i32s(const std::vector<std::int64_t>& values) {
    std::vector<std::int32_t> out;
    out.reserve(values.size());
    for (std::int64_t v : values) out.push_back(to_i32(v));
    return out;
}

template <typename T>
const T* data_or_null(const std::vector<T>& v) {
    return v.empty() ? nullptr : v.data();
}

template <typename T>
std::int64_t size_of(const std::vector<T>& v) {
    return static_cast<std::int64_t>(v.size());
}

// Caller fold ids replace the kernels' balanced `cv` plan (cv 0 = inferred).
struct Folds {
    std::vector<std::int32_t> ids;
    std::int32_t cv = 0;
};

Folds folds_of(const Params& params, const FitInputs& in) {
    Folds f;
    for (std::int64_t i = 0; i < in.n_fold_ids; ++i) f.ids.push_back(to_i32(in.fold_ids[i]));
    if (f.ids.empty()) f.cv = to_i32(params.get_int("cv"));
    return f;
}

// Fold of every row for the kernels that take explicit folds: the caller's
// ids, or row i in fold (i * cv) / n as the n4m reference assigns them.
n4m_status_t fold_ids(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                      std::vector<std::int32_t>& ids, std::int32_t& n_folds) {
    const std::int64_t n = in.X->rows;
    ids.assign(static_cast<std::size_t>(n), 0);
    if (in.fold_ids != nullptr) {
        for (std::int64_t i = 0; i < n; ++i) {
            ids[static_cast<std::size_t>(i)] = to_i32(in.fold_ids[i]);
        }
        n_folds = 1 + *std::max_element(ids.begin(), ids.end());
    } else {
        const std::int64_t cv = params.get_int("cv");
        if (cv > n) {
            set_error(ctx, "cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (std::int64_t i = 0; i < n; ++i) {
            ids[static_cast<std::size_t>(i)] = to_i32((i * cv) / n);
        }
        n_folds = to_i32(cv);
    }
    return N4M_OK;
}

n4m_status_t validation_plan(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                             PlanPtr& out) {
    std::vector<std::int32_t> ids;
    std::int32_t k = 0;
    n4m_status_t st = fold_ids(ctx, params, in, ids, k);
    if (st != N4M_OK) return st;
    n4m_validation_plan_t* raw = nullptr;
    st = n4m_validation_plan_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    st = n4m_validation_plan_set_n_samples(raw, in.X->rows);
    for (std::int32_t f = 0; st == N4M_OK && f < k; ++f) {
        std::vector<std::int64_t> train, test;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            (ids[i] == f ? test : train).push_back(static_cast<std::int64_t>(i));
        }
        if (train.empty() || test.empty()) {
            set_error(ctx, "every fold must have test rows and leave train rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        st = n4m_validation_plan_add_fold(raw, train.data(), size_of(train), test.data(),
                                          size_of(test));
    }
    return st;
}

// `heads` [ridge, pls, ridge_pls] as the kernels' mask (1 Ridge, 2 PLS).
std::int32_t heads_mask(const Params& params) { return to_i32(params.get_int("heads") + 1); }

n4m_status_t set_moment_policy(const Params& params, n4m_config_t* cfg) {
    return n4m_config_set_aom_moment_policy(
        cfg, static_cast<n4m_aom_moment_policy_t>(params.get_int("moment_policy")));
}

// Ridge lambdas and PLS component counts of a sweep; a requested head needs
// its grid (an empty PLS grid would fall back to the config's count).
struct Grid {
    std::vector<double> lambdas;
    std::vector<std::int32_t> components;
    std::int32_t heads = 0;
};

n4m_status_t sweep_grid(n4m_context_t* ctx, const Params& params, Grid& g) {
    g.lambdas = params.get_doubles("ridge_lambdas");
    g.components = i32s(params.get_ints("pls_components"));
    g.heads = heads_mask(params);
    if ((g.heads & 1) != 0 && g.lambdas.empty()) {
        set_error(ctx, "ridge_lambdas must not be empty when the ridge head is enabled");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if ((g.heads & 2) != 0 && g.components.empty()) {
        set_error(ctx, "pls_components must not be empty when the pls head is enabled");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

// One strict-linear operator list: op_kinds[i] with the parameters
// values[param_offsets[i] .. param_offsets[i + 1]).
struct Operators {
    std::vector<std::int32_t> kinds;
    std::vector<std::int32_t> offsets;
    std::vector<double> values;
};

Operators operators_of(const Params& params, const char* values_name) {
    return {i32s(params.get_ints("op_kinds")), i32s(params.get_ints("param_offsets")),
            params.get_doubles(values_name)};
}

n4m_status_t operator_bank(n4m_context_t* ctx, const Params& params, BankPtr& out) {
    const Operators ops = operators_of(params, "op_params");
    const std::size_t n = ops.kinds.size();
    bool valid = n > 0 && ops.offsets.size() == n + 1 && ops.offsets.front() == 0 &&
                 ops.offsets.back() == size_of(ops.values);
    for (std::size_t i = 0; valid && i < n; ++i) valid = ops.offsets[i] <= ops.offsets[i + 1];
    if (!valid) {
        set_error(ctx, "op_kinds, param_offsets and op_params do not describe an operator list");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_operator_bank_t* raw = nullptr;
    n4m_status_t st = n4m_operator_bank_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    for (std::size_t i = 0; st == N4M_OK && i < n; ++i) {
        const std::int32_t count = ops.offsets[i + 1] - ops.offsets[i];
        st = n4m_operator_bank_add(raw, static_cast<n4m_operator_kind_t>(ops.kinds[i]),
                                   count > 0 ? ops.values.data() + ops.offsets[i] : nullptr,
                                   count);
    }
    return st;
}

// Chains of one operator each, for the kernels that take chain descriptors.
std::vector<std::int32_t> single_operator_chains(std::size_t n_operators) {
    std::vector<std::int32_t> offsets(n_operators + 1);
    for (std::size_t i = 0; i < offsets.size(); ++i) offsets[i] = static_cast<std::int32_t>(i);
    return offsets;
}

// ---- Affine regressors --------------------------------------------------

n4m_status_t fit_aom_sweep(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                           const FitInputs& in, n4m_method_result_t** out) {
    Grid g;
    n4m_status_t st = sweep_grid(ctx, params, g);
    if (st == N4M_OK) st = set_moment_policy(params, cfg);
    if (st != N4M_OK) return st;
    const Folds f = folds_of(params, in);
    return n4m_model_selection_aom_sweep_run(
        ctx, cfg, in.X, in.Y, to_i32(params.get_int("profile")), f.cv, data_or_null(f.ids),
        size_of(f.ids), data_or_null(g.lambdas), size_of(g.lambdas), data_or_null(g.components),
        size_of(g.components), g.heads, out);
}

n4m_status_t fit_aom_chain_sweep(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                                 const FitInputs& in, n4m_method_result_t** out) {
    Grid g;
    n4m_status_t st = sweep_grid(ctx, params, g);
    if (st == N4M_OK) st = set_moment_policy(params, cfg);
    if (st != N4M_OK) return st;
    const Folds f = folds_of(params, in);
    const std::vector<std::int32_t> chains = i32s(params.get_ints("chain_offsets"));
    const Operators ops = operators_of(params, "chain_params");
    return n4m_model_selection_aom_chain_sweep_run(
        ctx, cfg, in.X, in.Y, f.cv, data_or_null(f.ids), size_of(f.ids), chains.data(),
        size_of(chains), ops.kinds.data(), size_of(ops.kinds), ops.offsets.data(),
        size_of(ops.offsets), data_or_null(ops.values), size_of(ops.values),
        data_or_null(g.lambdas), size_of(g.lambdas), data_or_null(g.components),
        size_of(g.components), g.heads, out);
}

// One chain (the operator list) with one head and parameter, fitted on every
// row without CV.
n4m_status_t fit_aom_chain_fixed(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                                 const FitInputs& in, n4m_method_result_t** out) {
    const n4m_status_t st = set_moment_policy(params, cfg);
    if (st != N4M_OK) return st;
    const Operators ops = operators_of(params, "chain_params");
    const std::int32_t chain[] = {0, static_cast<std::int32_t>(ops.kinds.size())};
    return n4m_model_selection_aom_chain_fixed_fit_run(
        ctx, cfg, in.X, in.Y, chain, 2, data_or_null(ops.kinds), size_of(ops.kinds),
        data_or_null(ops.offsets), size_of(ops.offsets), data_or_null(ops.values),
        size_of(ops.values), to_i32(params.get_int("head")), params.get_double("param"), out);
}

// One operator per chain, Ridge head only: the operator and alpha with the
// lowest CV RMSE.
n4m_status_t fit_aom_ridge_global(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                                  const FitInputs& in, n4m_method_result_t** out) {
    const n4m_status_t st = set_moment_policy(params, cfg);
    if (st != N4M_OK) return st;
    const Folds f = folds_of(params, in);
    const Operators ops = operators_of(params, "op_params");
    const std::vector<std::int32_t> chains = single_operator_chains(ops.kinds.size());
    const std::vector<double> lambdas = params.get_doubles("ridge_lambdas");
    return n4m_model_selection_aom_chain_sweep_run(
        ctx, cfg, in.X, in.Y, f.cv, data_or_null(f.ids), size_of(f.ids), chains.data(),
        size_of(chains), data_or_null(ops.kinds), size_of(ops.kinds), data_or_null(ops.offsets),
        size_of(ops.offsets), data_or_null(ops.values), size_of(ops.values),
        data_or_null(lambdas), size_of(lambdas), nullptr, 0, 1, out);
}

n4m_status_t fit_aom_robust_hpo(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                                const FitInputs& in, n4m_method_result_t** out) {
    return n4m_model_selection_robust_hpo_fit(ctx, cfg, in.X, in.Y,
                                              to_i32(params.get_int("profile")),
                                              to_i32(params.get_int("cv")), heads_mask(params),
                                              out);
}

n4m_status_t fit_aom_ridge_blender(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                                   const FitInputs& in, n4m_method_result_t** out) {
    const Folds f = folds_of(params, in);
    const std::vector<double> lambdas = params.get_doubles("ridge_lambdas");
    return n4m_ensemble_aom_ridge_blender_fit(
        ctx, cfg, in.X, in.Y, to_i32(params.get_int("profile")), f.cv, data_or_null(f.ids),
        size_of(f.ids), data_or_null(lambdas), size_of(lambdas),
        params.get_double("regularizer"), out);
}

n4m_status_t fit_aom_operator_pls_stack(n4m_context_t* ctx, n4m_config_t* cfg,
                                        const Params& params, const FitInputs& in,
                                        n4m_method_result_t** out) {
    const Folds f = folds_of(params, in);
    const std::vector<std::int32_t> components = i32s(params.get_ints("components"));
    const std::vector<double> alphas = params.get_doubles("alphas");
    return n4m_ensemble_aom_operator_pls_stack_fit(
        ctx, cfg, in.X, in.Y, to_i32(params.get_int("profile")), f.cv, data_or_null(f.ids),
        size_of(f.ids), data_or_null(components), size_of(components), data_or_null(alphas),
        size_of(alphas), params.get_double("std_penalty"), params.get_double("gap_penalty"),
        out);
}

// AOM / POP-PLS selectors return their own result handles; their outputs are
// copied into a MethodResult under the getter names.
template <typename R>
using MatrixGetter = n4m_status_t (*)(const R*, const double**, std::int64_t*, std::int64_t*);

template <typename R>
n4m_status_t copy_matrix(n4m_method_result_s& out, const char* name, const R* r,
                         MatrixGetter<R> get) {
    const double* data = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t st = get(r, &data, &rows, &cols);
    if (st == N4M_OK) {
        out.set_double_matrix(name, std::vector<double>(data, data + rows * cols), rows, cols);
    }
    return st;
}

// The selectors' PLS: regression SIMPLS, as the n4m reference configures it.
n4m_status_t selector_setup(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                            const FitInputs& in, BankPtr& bank, PlanPtr& plan) {
    n4m_status_t st = n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
    if (st == N4M_OK) st = n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    if (st == N4M_OK) st = n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
    if (st == N4M_OK) st = operator_bank(ctx, params, bank);
    if (st == N4M_OK) st = validation_plan(ctx, params, in, plan);
    return st;
}

n4m_status_t fit_aom_pls(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                         const FitInputs& in, n4m_method_result_t** out) {
    BankPtr bank;
    PlanPtr plan;
    n4m_status_t st = selector_setup(ctx, cfg, params, in, bank, plan);
    if (st != N4M_OK) return st;
    n4m_aom_global_result_t* raw = nullptr;
    st = n4m_model_selection_aom_pls_select(ctx, cfg, bank.get(), in.X, in.Y, plan.get(),
                                            to_i32(params.get_int("max_components")), &raw);
    const std::unique_ptr<n4m_aom_global_result_t, void (*)(n4m_aom_global_result_t*)> r(
        raw, n4m_model_selection_aom_pls_result_destroy);
    if (st != N4M_OK) return st;
    using G = n4m_aom_global_result_t;
    auto result = std::make_unique<n4m_method_result_s>();
    st = copy_matrix<G>(*result, "predictions", r.get(),
                        n4m_model_selection_aom_pls_result_get_predictions);
    if (st == N4M_OK) {
        st = copy_matrix<G>(*result, "coefficients", r.get(),
                            n4m_model_selection_aom_pls_result_get_coefficients);
    }
    if (st == N4M_OK) {
        st = copy_matrix<G>(*result, "input_coefficients", r.get(),
                            n4m_model_selection_aom_pls_result_get_input_coefficients);
    }
    if (st == N4M_OK) {
        st = copy_matrix<G>(*result, "intercept", r.get(),
                            n4m_model_selection_aom_pls_result_get_intercept);
    }
    std::int32_t op = 0, k = 0;
    double score = 0.0;
    if (st == N4M_OK) {
        st = n4m_model_selection_aom_pls_result_get_selected_operator_index(r.get(), &op);
    }
    if (st == N4M_OK) {
        st = n4m_model_selection_aom_pls_result_get_selected_n_components(r.get(), &k);
    }
    if (st == N4M_OK) st = n4m_model_selection_aom_pls_result_get_best_score(r.get(), &score);
    if (st != N4M_OK) return st;
    result->set_scalar("selected_operator_index", op);
    result->set_scalar("selected_n_components", k);
    result->set_scalar("best_score", score);
    *out = result.release();
    return N4M_OK;
}

n4m_status_t fit_pop_pls(n4m_context_t* ctx, n4m_config_t* cfg, const Params& params,
                         const FitInputs& in, n4m_method_result_t** out) {
    BankPtr bank;
    PlanPtr plan;
    n4m_status_t st = selector_setup(ctx, cfg, params, in, bank, plan);
    if (st != N4M_OK) return st;
    n4m_aom_per_component_result_t* raw = nullptr;
    st = n4m_model_selection_pop_pls_select(ctx, cfg, bank.get(), in.X, in.Y, plan.get(),
                                            to_i32(params.get_int("max_components")), &raw);
    const std::unique_ptr<n4m_aom_per_component_result_t,
                          void (*)(n4m_aom_per_component_result_t*)>
        r(raw, n4m_model_selection_pop_pls_result_destroy);
    if (st != N4M_OK) return st;
    using P = n4m_aom_per_component_result_t;
    auto result = std::make_unique<n4m_method_result_s>();
    st = copy_matrix<P>(*result, "predictions", r.get(),
                        n4m_model_selection_pop_pls_result_get_predictions);
    if (st == N4M_OK) {
        st = copy_matrix<P>(*result, "input_coefficients", r.get(),
                            n4m_model_selection_pop_pls_result_get_input_coefficients);
    }
    if (st == N4M_OK) {
        st = copy_matrix<P>(*result, "intercept", r.get(),
                            n4m_model_selection_pop_pls_result_get_intercept);
    }
    const std::int32_t* ops = nullptr;
    std::int32_t n_ops = 0, k = 0;
    double score = 0.0;
    if (st == N4M_OK) {
        st = n4m_model_selection_pop_pls_result_get_selected_operator_indices(r.get(), &ops,
                                                                              &n_ops);
    }
    if (st == N4M_OK) {
        st = n4m_model_selection_pop_pls_result_get_selected_n_components(r.get(), &k);
    }
    if (st == N4M_OK) st = n4m_model_selection_pop_pls_result_get_best_score(r.get(), &score);
    if (st != N4M_OK) return st;
    result->set_int64_vector("selected_operator_indices",
                             std::vector<std::int64_t>(ops, ops + n_ops));
    result->set_scalar("selected_n_components", k);
    result->set_scalar("best_score", score);
    *out = result.release();
    return N4M_OK;
}

// ---- Branch calibration ---------------------------------------------------

// The versioned strict10-gaussian-v1 operator bank of the n4m reference
// (n4m.model_selection.aom_calibration.BANK): widths in samples.
struct BankOperator {
    const char* name;
    n4m_operator_kind_t kind;
    double params[3];
    std::int32_t n_params;
};

constexpr BankOperator kStrict10[] = {
    {"identity", N4M_OP_IDENTITY, {0.0, 0.0, 0.0}, 0},
    {"sg_smooth_w11_p2", N4M_OP_SAVGOL_SMOOTH, {11.0, 2.0, 0.0}, 2},
    {"sg_smooth_w21_p3", N4M_OP_SAVGOL_SMOOTH, {21.0, 3.0, 0.0}, 2},
    {"sg_d1_w11_p2", N4M_OP_SAVGOL_DERIVATIVE, {11.0, 2.0, 1.0}, 3},
    {"sg_d1_w21_p3", N4M_OP_SAVGOL_DERIVATIVE, {21.0, 3.0, 1.0}, 3},
    {"sg_d2_w11_p2", N4M_OP_SAVGOL_DERIVATIVE, {11.0, 2.0, 2.0}, 3},
    {"detrend_d1", N4M_OP_DETREND_POLY, {1.0, 0.0, 0.0}, 1},
    {"detrend_d2", N4M_OP_DETREND_POLY, {2.0, 0.0, 0.0}, 1},
    {"gauss_d0_s1", N4M_OP_GAUSSIAN, {1.0, 4.0, 0.0}, 3},
    {"gauss_d0_s2", N4M_OP_GAUSSIAN, {2.0, 4.0, 0.0}, 3},
};

// Operators of the smoother, derivative and detrend roles in kStrict10.
struct Role {
    std::size_t ops[4];
    std::size_t n;
};
constexpr Role kRoles[] = {{{1, 2, 8, 9}, 4}, {{3, 4, 5, 0}, 3}, {{6, 7, 0, 0}, 2}};

void extend_chains(std::vector<std::size_t>& chain, std::uint32_t used, std::int64_t max_depth,
                   std::vector<std::vector<std::size_t>>& out) {
    for (std::size_t role = 0; role < 3; ++role) {
        if ((used & (1u << role)) != 0) continue;
        for (std::size_t k = 0; k < kRoles[role].n; ++k) {
            chain.push_back(kRoles[role].ops[k]);
            out.push_back(chain);
            if (static_cast<std::int64_t>(chain.size()) < max_depth) {
                extend_chains(chain, used | (1u << role), max_depth, out);
            }
            chain.pop_back();
        }
    }
}

// strict_chain_bank(max_depth) of the n4m reference: identity, then every
// chain of distinct roles in any order, sorted by length and by the operator
// names joined with '>'.
std::vector<std::vector<std::size_t>> strict_chains(std::int64_t max_depth) {
    std::vector<std::vector<std::size_t>> chains = {{0}};
    std::vector<std::size_t> chain;
    extend_chains(chain, 0, max_depth, chains);
    auto key = [](const std::vector<std::size_t>& c) {
        std::string name;
        for (std::size_t op : c) {
            name += (name.empty() ? "" : ">") + std::string(kStrict10[op].name);
        }
        return name;
    };
    std::stable_sort(chains.begin(), chains.end(), [&](const auto& a, const auto& b) {
        return a.size() != b.size() ? a.size() < b.size() : key(a) < key(b);
    });
    return chains;
}

// numpy.logspace(-6, 6, 50): the trace-relative Ridge grid of the reference.
std::vector<double> relative_alphas() {
    constexpr int kCount = 50;
    const double step = 12.0 / (kCount - 1);
    std::vector<double> alphas(kCount);
    for (int i = 0; i < kCount; ++i) {
        const double exponent = i == kCount - 1 ? 6.0 : static_cast<double>(i) * step + -6.0;
        alphas[static_cast<std::size_t>(i)] = std::pow(10.0, exponent);
    }
    return alphas;
}

constexpr std::uint32_t kTagCalibration = 0x314C4143u;  // "CAL1"
// Calibration heads of n4m_model_selection_aom_calibration_fit.
constexpr std::int32_t kHeadPls = 0;
constexpr std::int32_t kHeadRidgeAbsolute = 1;
constexpr std::int32_t kHeadRidgeRelative = 2;

// Global (or Fast leader-screened) branch x chain x parameter calibration.
// State: the selected branch (0 raw, 1 SNV, 2 MSC), the input-space
// coefficients and the kernel state vector (branch mean, MSC reference,
// intercept, alpha base).
class CalibrationAdapter final : public Adapter {
  public:
    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_PREDICT | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return size_of(coef_); }
    std::int64_t n_outputs() const noexcept override { return coef_.empty() ? 0 : 1; }
    const n4m_method_result_t* fit_result() const noexcept override { return result_.get(); }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        reset();
        if (in.Y->cols != 1) {
            set_error(ctx, "AOM calibration requires one target");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        const std::vector<std::int32_t> branches = i32s(params.get_ints("branches"));
        std::vector<std::int32_t> sorted(branches);
        std::sort(sorted.begin(), sorted.end());
        if (branches.empty() || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            set_error(ctx, "branches must be non-empty and unique");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<std::int32_t> folds;
        std::int32_t n_folds = 0;
        n4m_status_t st = fold_ids(ctx, params, in, folds, n_folds);
        if (st != N4M_OK) return st;

        const auto chains = strict_chains(params.get_int("max_depth"));
        n4m_operator_bank_t* raw_bank = nullptr;
        st = n4m_operator_bank_create(&raw_bank);
        if (st != N4M_OK) return st;
        const BankPtr bank(raw_bank);
        std::vector<std::int32_t> offsets = {0};
        for (const auto& chain : chains) {
            for (std::size_t op : chain) {
                const BankOperator& b = kStrict10[op];
                st = n4m_operator_bank_add(raw_bank, b.kind, b.n_params > 0 ? b.params : nullptr,
                                           b.n_params);
                if (st != N4M_OK) return st;
            }
            offsets.push_back(offsets.back() + static_cast<std::int32_t>(chain.size()));
        }

        const bool ridge = params.get_int("head") == 1;
        std::vector<double> alphas = params.get_doubles("alphas");
        std::int32_t head = kHeadPls;
        if (ridge) {
            head = alphas.empty() ? kHeadRidgeRelative : kHeadRidgeAbsolute;
            if (alphas.empty()) alphas = relative_alphas();
        } else {
            alphas.clear();
        }
        const std::int32_t count = ridge ? static_cast<std::int32_t>(alphas.size())
                                         : to_i32(params.get_int("max_components"));
        const bool fast = params.get_bool("fast");
        const std::int64_t n_chains = size_of(chains);
        const std::int64_t n_scores = fast ? count : size_of(branches) * n_chains * count;
        const std::int64_t p = in.X->cols;
        std::vector<double> coef(static_cast<std::size_t>(p));
        std::vector<double> state(static_cast<std::size_t>(2 * p + 2));
        std::vector<double> scores(static_cast<std::size_t>(n_scores));
        std::int32_t selected[3] = {0, 0, 0};
        st = n4m_model_selection_aom_calibration_fit(
            ctx, in.X, in.Y, bank.get(), offsets.data(), static_cast<std::int32_t>(n_chains),
            branches.data(), static_cast<std::int32_t>(branches.size()), folds.data(), n_folds,
            head, fast ? 1 : 0, to_i32(params.get_int("rank")), data_or_null(alphas), count,
            coef.data(), p, state.data(), size_of(state), scores.data(), n_scores, selected);
        if (st != N4M_OK) return st;

        auto result = std::make_unique<n4m_method_result_s>();
        const std::int64_t score_rows = fast ? 1 : size_of(branches) * n_chains;
        result->set_double_matrix("cv_scores", std::move(scores), score_rows, count);
        result->set_double_matrix("coefficients", coef, p, 1);
        result->set_double_matrix("intercept", {state[static_cast<std::size_t>(2 * p)]}, 1, 1);
        result->set_scalar("selected_branch", branches[static_cast<std::size_t>(selected[0])]);
        result->set_scalar("selected_chain_index", selected[1]);
        result->set_scalar("selected_parameter_index", selected[2]);
        if (ridge) {
            result->set_scalar("selected_alpha", alphas[static_cast<std::size_t>(selected[2])] *
                                                     state[static_cast<std::size_t>(2 * p + 1)]);
        } else {
            result->set_scalar("selected_n_components", selected[2] + 1);
        }
        branch_ = branches[static_cast<std::size_t>(selected[0])];
        coef_ = std::move(coef);
        state_ = std::move(state);
        result_.reset(result.release());
        return N4M_OK;
    }

    n4m_status_t predict(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                         n4m_matrix_view_t& out) const override {
        const n4m_status_t st = n4m_model_selection_aom_calibration_predict(
            &X, branch_, coef_.data(), size_of(coef_), state_.data(), size_of(state_), &out);
        if (st != N4M_OK) set_error(ctx, "AOM calibration predict failed");
        return st;
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t w;
        n4m_state_write_i64(&w, branch_);
        n4m_state_write_f64_array(&w, coef_.data(), size_of(coef_));
        n4m_state_write_f64_array(&w, state_.data(), size_of(state_));
        StateBlock block;
        block.tag = kTagCalibration;
        block.bytes = std::move(w.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params&,
                            const std::vector<StateBlock>& blocks) override {
        reset();
        std::int64_t branch = -1, p = 0;
        bool ok = blocks.size() == 1 && blocks[0].tag == kTagCalibration;
        n4m_state_reader_t r(ok ? blocks[0].bytes.data() : nullptr,
                             ok ? blocks[0].bytes.size() : 0);
        ok = ok && n4m_state_read_i64(&r, &branch) && branch >= 0 && branch <= 2 &&
             n4m_state_peek_array_length(&r, std::int64_t{1} << 31, &p) && p > 0;
        if (ok) {
            coef_.resize(static_cast<std::size_t>(p));
            state_.resize(static_cast<std::size_t>(2 * p + 2));
            ok = n4m_state_read_f64_array(&r, coef_.data(), p) &&
                 n4m_state_read_f64_array(&r, state_.data(), 2 * p + 2) && r.remaining() == 0;
        }
        if (!ok) {
            reset();
            set_error(ctx, "AOM calibration state does not match the method");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        branch_ = static_cast<std::int32_t>(branch);
        return N4M_OK;
    }

  private:
    void reset() noexcept {
        branch_ = 0;
        coef_.clear();
        state_.clear();
        result_.reset();
    }
    std::int32_t branch_ = 0;
    std::vector<double> coef_;
    std::vector<double> state_;
    ResultPtr result_;
};

// ---- AOM preprocessing ------------------------------------------------------

constexpr std::uint32_t kTagWidth = 0x31504F41u;  // "AOP1"

// The strict-linear operator bank mixed by the gating weights (soft: mean of
// the operator outputs, hard: the first operator). Row-wise and stateless:
// the fitted state is the input width.
class AomPreprocessingAdapter final : public Adapter {
  public:
    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return width_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override { return width_; }

    // Fit checks the operators on the training rows.
    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        n4m_status_t st = build(ctx, params, in.X->cols);
        ResultPtr result;
        if (st == N4M_OK) st = apply(ctx, *in.X, result);
        if (st != N4M_OK) width_ = 0;
        return st;
    }

    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        ResultPtr result;
        n4m_status_t st = apply(ctx, X, result);
        const double* values = nullptr;
        std::int64_t rows = 0, cols = 0;
        if (st == N4M_OK) {
            st = n4m_method_result_get_double_matrix(result.get(), "transformed", &values, &rows,
                                                     &cols);
        }
        if (st != N4M_OK) return st;
        if (out.dtype != N4M_DTYPE_F64) return N4M_ERR_DTYPE_MISMATCH;
        if (rows != out.rows || cols != out.cols) return N4M_ERR_SHAPE_MISMATCH;
        auto* dst = static_cast<double*>(out.data);
        for (std::int64_t i = 0; i < rows; ++i) {
            for (std::int64_t j = 0; j < cols; ++j) {
                dst[i * out.row_stride + j * out.col_stride] = values[i * cols + j];
            }
        }
        return N4M_OK;
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t w;
        n4m_state_write_i64(&w, width_);
        StateBlock block;
        block.tag = kTagWidth;
        block.bytes = std::move(w.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        width_ = 0;
        std::int64_t width = 0;
        bool ok = blocks.size() == 1 && blocks[0].tag == kTagWidth;
        n4m_state_reader_t r(ok ? blocks[0].bytes.data() : nullptr,
                             ok ? blocks[0].bytes.size() : 0);
        ok = ok && n4m_state_read_i64(&r, &width) && width > 0 &&
             width <= (std::int64_t{1} << 31) && r.remaining() == 0;
        if (!ok) {
            set_error(ctx, "AOM preprocessing state must be one AOP1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        return build(ctx, params, width);
    }

  private:
    n4m_status_t build(n4m_context_t* ctx, const Params& params, std::int64_t width) {
        width_ = 0;
        gate_.reset();
        n4m_status_t st = operator_bank(ctx, params, bank_);
        if (st != N4M_OK) return st;
        n4m_gating_strategy_t* gate = nullptr;
        st = n4m_gating_strategy_create(
            &gate, static_cast<n4m_gating_mode_t>(params.get_int("gating_mode")));
        if (st != N4M_OK) return st;
        gate_.reset(gate);
        width_ = width;
        return N4M_OK;
    }

    n4m_status_t apply(n4m_context_t* ctx, const n4m_matrix_view_t& X, ResultPtr& out) const {
        n4m_method_result_t* raw = nullptr;
        const n4m_status_t st = n4m_model_selection_aom_preprocessing_fit(
            ctx, bank_.get(), gate_.get(), &X, nullptr, &raw);
        out.reset(raw);
        return st;
    }

    BankPtr bank_;
    GatePtr gate_;
    std::int64_t width_ = 0;
};

}  // namespace

std::unique_ptr<Adapter> make_aom_sweep(const MethodSpec&) {
    return input_affine(fit_aom_sweep, "intercept");
}

std::unique_ptr<Adapter> make_aom_chain_sweep(const MethodSpec&) {
    return input_affine(fit_aom_chain_sweep, "intercept");
}

std::unique_ptr<Adapter> make_aom_chain_fixed_fit(const MethodSpec&) {
    return input_affine(fit_aom_chain_fixed, "intercept");
}

std::unique_ptr<Adapter> make_aom_ridge_global(const MethodSpec&) {
    return input_affine(fit_aom_ridge_global, "intercept");
}

std::unique_ptr<Adapter> make_aom_robust_hpo(const MethodSpec&) {
    return input_affine(fit_aom_robust_hpo, "intercept");
}

std::unique_ptr<Adapter> make_aom_ridge_blender(const MethodSpec&) {
    return input_affine(fit_aom_ridge_blender, "intercept");
}

std::unique_ptr<Adapter> make_aom_operator_pls_stack(const MethodSpec&) {
    return input_affine(fit_aom_operator_pls_stack, "input_intercept");
}

std::unique_ptr<Adapter> make_aom_pls(const MethodSpec&) {
    return input_affine(fit_aom_pls, "intercept");
}

std::unique_ptr<Adapter> make_pop_pls(const MethodSpec&) {
    return input_affine(fit_pop_pls, "intercept");
}

std::unique_ptr<Adapter> make_aom_calibration(const MethodSpec&) {
    return std::make_unique<CalibrationAdapter>();
}

std::unique_ptr<Adapter> make_aom_preprocessing(const MethodSpec&) {
    return std::make_unique<AomPreprocessingAdapter>();
}

}  // namespace n4m::estimator
