// SPDX-License-Identifier: CECILL-2.1
//
// Selector role adapters: a fitted selector keeps the input width and the
// selected column indices; transform returns those columns in ascending input
// order (the SelectorMixin convention every binding follows). The selection
// itself is the existing native kernel's; the adapters only build its inputs.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/spec.hpp"
#include "core/method_result.hpp"

namespace n4m::estimator {

namespace {

constexpr std::uint32_t kTagSelection = 0x314C4553u;  // "SEL1"

struct PlanDeleter {
    void operator()(n4m_validation_plan_t* p) const noexcept { n4m_validation_plan_destroy(p); }
};
struct ConfigDeleter {
    void operator()(n4m_config_t* c) const noexcept { n4m_config_destroy(c); }
};
struct ResultDeleter {
    void operator()(n4m_method_result_t* r) const noexcept { n4m_method_result_destroy(r); }
};
using PlanPtr = std::unique_ptr<n4m_validation_plan_t, PlanDeleter>;
using ConfigPtr = std::unique_ptr<n4m_config_t, ConfigDeleter>;
using ResultPtr = std::unique_ptr<n4m_method_result_t, ResultDeleter>;

// Internal-CV plan: the caller's fold ids, or the canonical contiguous plan
// (fold size n / k, the last fold absorbs the remainder).
n4m_status_t make_plan(n4m_context_t* ctx, const FitInputs& in, std::int64_t n_folds,
                       PlanPtr& out) {
    const std::int64_t n = in.X->rows;
    std::vector<std::int64_t> fold(static_cast<std::size_t>(n));
    std::int64_t k = n_folds;
    if (in.fold_ids != nullptr) {
        k = 1 + *std::max_element(in.fold_ids, in.fold_ids + n);
        std::copy(in.fold_ids, in.fold_ids + n, fold.begin());
    } else {
        if (k < 2 || k > n) {
            set_error(ctx, "cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const std::int64_t size = std::max<std::int64_t>(1, n / k);
        for (std::int64_t i = 0; i < n; ++i) {
            fold[static_cast<std::size_t>(i)] = std::min(i / size, k - 1);
        }
    }
    n4m_validation_plan_t* raw = nullptr;
    n4m_status_t st = n4m_validation_plan_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    st = n4m_validation_plan_set_n_samples(raw, n);
    for (std::int64_t f = 0; st == N4M_OK && f < k; ++f) {
        std::vector<std::int64_t> train, test;
        for (std::int64_t i = 0; i < n; ++i) {
            (fold[static_cast<std::size_t>(i)] == f ? test : train).push_back(i);
        }
        if (train.empty() || test.empty()) {
            set_error(ctx, "every fold must have test rows and leave train rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        st = n4m_validation_plan_add_fold(raw, train.data(), static_cast<std::int64_t>(train.size()),
                                          test.data(), static_cast<std::int64_t>(test.size()));
    }
    return st;
}

void put_u64(std::vector<unsigned char>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

std::uint64_t get_u64(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

// Selectors whose kernel returns a MethodResult with int64 "selected_indices".
class SelectorAdapter final : public Adapter {
  public:
    using FitFn = std::function<n4m_status_t(n4m_context_t*, const Params&, const FitInputs&,
                                             n4m_method_result_t**)>;
    explicit SelectorAdapter(FitFn fit_fn) : fit_fn_(std::move(fit_fn)) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SELECTED_INDICES | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override {
        return static_cast<std::int64_t>(selected_.size());
    }
    const std::vector<std::int64_t>* selected_indices() const noexcept override {
        return &selected_;
    }
    const n4m_method_result_t* fit_result() const noexcept override { return result_.get(); }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        result_.reset();
        selected_.clear();
        n_features_ = 0;
        n4m_method_result_t* raw = nullptr;
        n4m_status_t st = fit_fn_(ctx, params, in, &raw);
        ResultPtr result(raw);
        if (st != N4M_OK) return st;
        const std::int64_t* idx = nullptr;
        std::int64_t k = 0;
        st = n4m_method_result_get_int64_vector(result.get(), "selected_indices", &idx, &k);
        if (st != N4M_OK || k <= 0) {
            set_error(ctx, "selector returned no selected_indices");
            return st != N4M_OK ? st : N4M_ERR_NUMERICAL_FAILURE;
        }
        std::vector<std::int64_t> selected(idx, idx + k);
        st = validate(ctx, selected, in.X->cols);
        if (st != N4M_OK) return st;
        selected_ = std::move(selected);
        n_features_ = in.X->cols;
        result_ = std::move(result);
        return N4M_OK;
    }

    // Selected columns in ascending input order.
    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        if (X.dtype != N4M_DTYPE_F64 || out.dtype != N4M_DTYPE_F64 ||
            out.cols != transform_cols()) {
            set_error(ctx, "selector transform expects F64 views of the selected width");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        std::vector<std::int64_t> cols(selected_);
        std::sort(cols.begin(), cols.end());
        const auto* src = static_cast<const double*>(X.data);
        auto* dst = static_cast<double*>(out.data);
        for (std::int64_t i = 0; i < X.rows; ++i) {
            for (std::size_t j = 0; j < cols.size(); ++j) {
                dst[i * out.row_stride + static_cast<std::int64_t>(j) * out.col_stride] =
                    src[i * X.row_stride + cols[j] * X.col_stride];
            }
        }
        return N4M_OK;
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        StateBlock block;
        block.tag = kTagSelection;
        put_u64(block.bytes, static_cast<std::uint64_t>(n_features_));
        put_u64(block.bytes, selected_.size());
        for (std::int64_t v : selected_) put_u64(block.bytes, static_cast<std::uint64_t>(v));
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params&,
                            const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagSelection || blocks[0].bytes.size() < 16) {
            set_error(ctx, "selector state must be one SEL1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        const auto& b = blocks[0].bytes;
        const std::uint64_t p = get_u64(b.data());
        const std::uint64_t k = get_u64(b.data() + 8);
        if (p == 0 || p > (std::uint64_t{1} << 31) || k == 0 || k > p ||
            b.size() != 16 + 8 * static_cast<std::size_t>(k)) {
            set_error(ctx, "selector state has inconsistent sizes");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        std::vector<std::int64_t> selected(static_cast<std::size_t>(k));
        for (std::size_t j = 0; j < selected.size(); ++j) {
            selected[j] = static_cast<std::int64_t>(get_u64(b.data() + 16 + 8 * j));
        }
        n4m_status_t st = validate(ctx, selected, static_cast<std::int64_t>(p));
        if (st != N4M_OK) return N4M_ERR_CORRUPT_BUFFER;
        selected_ = std::move(selected);
        n_features_ = static_cast<std::int64_t>(p);
        result_.reset();
        return N4M_OK;
    }

  private:
    static n4m_status_t validate(n4m_context_t* ctx, const std::vector<std::int64_t>& selected,
                                 std::int64_t p) {
        if (selected.empty()) {
            set_error(ctx, "selector selected no column");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<std::int64_t> sorted(selected);
        std::sort(sorted.begin(), sorted.end());
        if (sorted.front() < 0 || sorted.back() >= p ||
            std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            set_error(ctx, "selected indices must be unique input columns");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        return N4M_OK;
    }

    FitFn fit_fn_;
    std::vector<std::int64_t> selected_;
    std::int64_t n_features_ = 0;
    ResultPtr result_;
};

// PLS configuration shared by every selector kernel: SIMPLS, centered X and Y,
// unscaled (the n4m reference selector configuration).
n4m_status_t selector_config(const Params& p, bool store_scores, ConfigPtr& out) {
    n4m_config_t* raw = nullptr;
    n4m_status_t st = n4m_config_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    const std::int64_t k = p.get_int("n_components");
    if (k > std::numeric_limits<std::int32_t>::max()) return N4M_ERR_INVALID_ARGUMENT;
    if (st == N4M_OK) st = n4m_config_set_algorithm(raw, N4M_ALGO_PLS_REGRESSION);
    if (st == N4M_OK) st = n4m_config_set_solver(raw, N4M_SOLVER_SIMPLS);
    if (st == N4M_OK) st = n4m_config_set_deflation(raw, N4M_DEFLATION_REGRESSION);
    if (st == N4M_OK) st = n4m_config_set_n_components(raw, static_cast<std::int32_t>(k));
    if (st == N4M_OK) st = n4m_config_set_center_x(raw, 1);
    if (st == N4M_OK) st = n4m_config_set_scale_x(raw, 0);
    if (st == N4M_OK) st = n4m_config_set_center_y(raw, 1);
    if (st == N4M_OK) st = n4m_config_set_scale_y(raw, 0);
    if (st == N4M_OK) st = n4m_config_set_tol(raw, 1e-6);
    if (st == N4M_OK) st = n4m_config_set_max_iter(raw, 500);
    if (st == N4M_OK) st = n4m_config_set_store_scores(raw, store_scores ? 1 : 0);
    return st;
}

std::int32_t i32(std::int64_t v) {
    if (v > std::numeric_limits<std::int32_t>::max() ||
        v < std::numeric_limits<std::int32_t>::min()) {
        throw std::out_of_range("integer parameter exceeds int32");
    }
    return static_cast<std::int32_t>(v);
}

// Size parameters whose 0 means "n_components" / "n_features".
std::int32_t or_components(const Params& p, const char* name) {
    const std::int64_t v = p.get_int(name);
    return i32(v == 0 ? p.get_int("n_components") : v);
}
std::int32_t or_features(const Params& p, const char* name, const FitInputs& in) {
    const std::int64_t v = p.get_int(name);
    return i32(v == 0 ? in.X->cols : v);
}
std::uint64_t u64(const Params& p, const char* name) {
    return static_cast<std::uint64_t>(p.get_int(name));
}

using PlanKernel = std::function<n4m_status_t(n4m_context_t*, const n4m_config_t*,
                                              const n4m_validation_plan_t*, const Params&,
                                              const FitInputs&, n4m_method_result_t**)>;
using ConfigKernel = std::function<n4m_status_t(n4m_context_t*, const n4m_config_t*,
                                                const Params&, const FitInputs&,
                                                n4m_method_result_t**)>;

// Kernels taking (config, internal-CV plan).
std::unique_ptr<Adapter> with_plan(PlanKernel kernel) {
    return std::make_unique<SelectorAdapter>(
        [kernel](n4m_context_t* ctx, const Params& p, const FitInputs& in,
                 n4m_method_result_t** out) {
            ConfigPtr cfg;
            n4m_status_t st = selector_config(p, false, cfg);
            if (st != N4M_OK) return st;
            PlanPtr plan;
            st = make_plan(ctx, in, p.get_int("cv"), plan);
            return st != N4M_OK ? st : kernel(ctx, cfg.get(), plan.get(), p, in, out);
        });
}

// Kernels taking a config but no plan.
std::unique_ptr<Adapter> with_config(ConfigKernel kernel) {
    return std::make_unique<SelectorAdapter>(
        [kernel](n4m_context_t* ctx, const Params& p, const FitInputs& in,
                 n4m_method_result_t** out) {
            ConfigPtr cfg;
            n4m_status_t st = selector_config(p, false, cfg);
            return st != N4M_OK ? st : kernel(ctx, cfg.get(), p, in, out);
        });
}

}  // namespace

std::unique_ptr<Adapter> make_select_spa(const MethodSpec&) {
    return with_config([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_feature_selection_spa_select(ctx, cfg, in.X, in.Y, i32(p.get_int("top_k")), out);
    });
}

std::unique_ptr<Adapter> make_select_vip_spa(const MethodSpec&) {
    return with_config([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_feature_selection_vip_spa_select(ctx, cfg, in.X, in.Y,
                                                    p.get_double("vip_threshold"),
                                                    i32(p.get_int("top_k")), out);
    });
}

std::unique_ptr<Adapter> make_select_randomization(const MethodSpec&) {
    return with_config([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_feature_selection_randomization_select(
            ctx, cfg, in.X, in.Y, i32(p.get_int("n_permutations")),
            u64(p, "randomization_seed"), p.get_double("alpha"), out);
    });
}

std::unique_ptr<Adapter> make_select_stability(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_stability_select(ctx, cfg, in.X, in.Y, plan,
                                                      i32(p.get_int("top_k")), out);
    });
}

// When UVE keeps fewer than min_features columns (-1 means n_components, 0
// disables), the highest real stability scores complete the selection,
// appended after the native picks (non-finite scores rank last, ties to the
// lower index) -- the n4m reference rule, now native.
std::unique_ptr<Adapter> make_select_uve(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        n4m_method_result_t** out) {
        n4m_status_t st = n4m_feature_selection_uve_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("noise_features")), u64(p, "noise_seed"),
            out);
        if (st != N4M_OK) return st;
        const std::int64_t min_features = p.get_int("min_features") < 0
                                              ? p.get_int("n_components")
                                              : p.get_int("min_features");
        auto& result = static_cast<n4m_method_result_s&>(**out);
        auto& selected = result.int64_arrays["selected_indices"];
        if (static_cast<std::int64_t>(selected.size()) >= min_features) return N4M_OK;
        const auto scores_it = result.double_arrays.find("real_stability_scores");
        const std::int64_t n_features = in.X->cols;
        if (scores_it == result.double_arrays.end() ||
            static_cast<std::int64_t>(scores_it->second.size()) != n_features) {
            set_error(ctx, "UVE selected fewer than min_features and has no stability scores");
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const auto& scores = scores_it->second;
        std::vector<std::int64_t> order(static_cast<std::size_t>(n_features));
        for (std::int64_t j = 0; j < n_features; ++j) order[static_cast<std::size_t>(j)] = j;
        auto score = [&](std::int64_t j) {
            const double v = scores[static_cast<std::size_t>(j)];
            return std::isfinite(v) ? v : -std::numeric_limits<double>::infinity();
        };
        std::stable_sort(order.begin(), order.end(),
                         [&](std::int64_t a, std::int64_t b) { return score(a) > score(b); });
        for (std::int64_t j : order) {
            if (static_cast<std::int64_t>(selected.size()) >= min_features) break;
            if (std::find(selected.begin(), selected.end(), j) == selected.end()) {
                selected.push_back(j);
            }
        }
        return N4M_OK;
    });
}

std::unique_ptr<Adapter> make_select_cars(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_cars_select(ctx, cfg, in.X, in.Y, plan,
                                                 i32(p.get_int("n_iterations")),
                                                 or_components(p, "min_features"), out);
    });
}

std::unique_ptr<Adapter> make_select_random_frog(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_random_frog_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_iterations")),
            i32(p.get_int("initial_size")), or_components(p, "min_size"),
            or_features(p, "max_size", in), i32(p.get_int("top_k")), u64(p, "seed"), out);
    });
}

std::unique_ptr<Adapter> make_select_scars(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_scars_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_iterations")),
            or_components(p, "min_features"), p.get_double("sample_fraction"), u64(p, "seed"),
            out);
    });
}

std::unique_ptr<Adapter> make_select_ga(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_ga_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_generations")),
            i32(p.get_int("population_size")), or_components(p, "min_features"),
            or_features(p, "max_features", in), p.get_double("mutation_rate"), u64(p, "seed"),
            out);
    });
}

std::unique_ptr<Adapter> make_select_pso(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_pso_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_swarm")), i32(p.get_int("n_iterations")),
            p.get_double("w"), p.get_double("c1"), p.get_double("c2"), p.get_double("v_max"),
            u64(p, "seed"), out);
    });
}

std::unique_ptr<Adapter> make_select_vissa(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_vissa_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_iterations")),
            i32(p.get_int("n_submodels")), p.get_double("ratio_kept"), p.get_double("threshold"),
            p.get_double("floor_probability"), u64(p, "seed"), out);
    });
}

std::unique_ptr<Adapter> make_select_shaving(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_shaving_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_steps")),
            or_components(p, "min_features"), p.get_double("shave_fraction"), out);
    });
}

std::unique_ptr<Adapter> make_select_bve(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_bve_select(ctx, cfg, in.X, in.Y, plan,
                                                i32(p.get_int("n_steps")),
                                                or_components(p, "min_features"), out);
    });
}

std::unique_ptr<Adapter> make_select_rep(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_rep_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_steps")),
            or_components(p, "min_features"), i32(p.get_int("remove_count")), out);
    });
}

std::unique_ptr<Adapter> make_select_ipw(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_ipw_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_iterations")), i32(p.get_int("top_k")),
            p.get_double("damping"), p.get_double("weight_floor"), out);
    });
}

std::unique_ptr<Adapter> make_select_st(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        const auto thresholds = p.get_doubles("thresholds");
        return n4m_feature_selection_st_select(
            ctx, cfg, in.X, in.Y, plan, thresholds.data(),
            static_cast<std::int64_t>(thresholds.size()), or_components(p, "min_selected"), out);
    });
}

std::unique_ptr<Adapter> make_select_t2(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        const auto alphas = p.get_doubles("alpha_thresholds");
        return n4m_feature_selection_t2_select(
            ctx, cfg, in.X, in.Y, plan, alphas.data(), static_cast<std::int64_t>(alphas.size()),
            or_components(p, "min_selected"), out);
    });
}

std::unique_ptr<Adapter> make_select_bipls(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_bipls_select(ctx, cfg, in.X, in.Y, plan,
                                                  i32(p.get_int("interval_width")),
                                                  i32(p.get_int("min_intervals")), out);
    });
}

std::unique_ptr<Adapter> make_select_sipls(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_sipls_select(ctx, cfg, in.X, in.Y, plan,
                                                  i32(p.get_int("interval_width")),
                                                  i32(p.get_int("combination_size")), out);
    });
}

std::unique_ptr<Adapter> make_select_emcuve(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_emcuve_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("noise_features")), u64(p, "noise_seed"),
            i32(p.get_int("n_ensembles")), p.get_double("vote_threshold"), out);
    });
}

std::unique_ptr<Adapter> make_select_iriv(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_iriv_select(ctx, cfg, in.X, in.Y, plan,
                                                 i32(p.get_int("max_rounds")), u64(p, "seed"),
                                                 out);
    });
}

std::unique_ptr<Adapter> make_select_irf(const MethodSpec&) {
    return with_plan([](auto ctx, auto cfg, auto plan, const Params& p, const FitInputs& in,
                        auto out) {
        return n4m_feature_selection_irf_select(
            ctx, cfg, in.X, in.Y, plan, i32(p.get_int("n_iterations")),
            i32(p.get_int("window_size")), i32(p.get_int("initial_intervals")),
            i32(p.get_int("top_k")), u64(p, "seed"), out);
    });
}

// Weighted variable combination takes no config: n_components is explicit.
std::unique_ptr<Adapter> make_select_wvc(const MethodSpec&) {
    return std::make_unique<SelectorAdapter>(
        [](n4m_context_t* ctx, const Params& p, const FitInputs& in, n4m_method_result_t** out) {
            return n4m_feature_selection_wvc_select(ctx, in.X, in.Y, i32(p.get_int("n_components")),
                                                    i32(p.get_int("top_k")),
                                                    p.get_bool("normalize") ? 1 : 0, out);
        });
}

std::unique_ptr<Adapter> make_select_wvc_threshold(const MethodSpec&) {
    return std::make_unique<SelectorAdapter>(
        [](n4m_context_t* ctx, const Params& p, const FitInputs& in, n4m_method_result_t** out) {
            return n4m_feature_selection_wvc_threshold_select(
                ctx, in.X, in.Y, i32(p.get_int("n_components")), p.get_bool("normalize") ? 1 : 0,
                p.get_double("score_threshold"), p.get_double("threshold_factor"),
                or_components(p, "min_selected"), out);
        });
}

// Ranks the variables of a PLS model fitted with stored scores (VIP,
// |coefficient| or selectivity ratio).
std::unique_ptr<Adapter> make_select_variable_rank(const MethodSpec&) {
    return std::make_unique<SelectorAdapter>(
        [](n4m_context_t* ctx, const Params& p, const FitInputs& in, n4m_method_result_t** out) {
            ConfigPtr cfg;
            n4m_status_t st = selector_config(p, true, cfg);
            if (st != N4M_OK) return st;
            n4m_model_t* raw = nullptr;
            st = n4m_model_fit(ctx, cfg.get(), in.X, in.Y, &raw);
            if (st != N4M_OK) return st;
            std::unique_ptr<n4m_model_t, void (*)(n4m_model_t*)> model(raw, n4m_model_destroy);
            return n4m_feature_selection_variable_select_rank(
                ctx, model.get(), in.X, i32(p.get_int("rank_method")), i32(p.get_int("top_k")),
                out);
        });
}

}  // namespace n4m::estimator
