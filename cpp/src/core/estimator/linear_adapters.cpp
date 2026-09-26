// SPDX-License-Identifier: CECILL-2.1
//
// Estimator adapters whose fitted state is one N4MM model:
//   - PLS/PCR fitted through n4m_model_fit (predict + latent transform);
//   - methods whose MethodResult is an affine predictor, promoted with
//     n4m_model_from_method_result (predict only).
// The adapters call the public kernels; they add no numerics.

#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/spec.hpp"

namespace n4m::estimator {

namespace {

constexpr std::uint32_t kTagN4MM = 0x4D4D344Eu;  // "N4MM"

struct ConfigDeleter {
    void operator()(n4m_config_t* c) const noexcept { n4m_config_destroy(c); }
};
struct ModelDeleter {
    void operator()(n4m_model_t* m) const noexcept { n4m_model_destroy(m); }
};
struct ResultDeleter {
    void operator()(n4m_method_result_t* r) const noexcept { n4m_method_result_destroy(r); }
};
using ConfigPtr = std::unique_ptr<n4m_config_t, ConfigDeleter>;
using ModelPtr = std::unique_ptr<n4m_model_t, ModelDeleter>;
using ResultPtr = std::unique_ptr<n4m_method_result_t, ResultDeleter>;

// Builds a config from the parameters the method declares; undeclared
// settings keep the n4m_config defaults.
n4m_status_t make_config(const Params& params, ConfigPtr& out) {
    n4m_config_t* raw = nullptr;
    n4m_status_t st = n4m_config_create(&raw);
    if (st != N4M_OK) return st;
    out.reset(raw);
    const MethodSpec& spec = params.spec();
    auto has = [&](const char* name) { return param_index(spec, name) >= 0; };
    if (has("n_components")) {
        const std::int64_t n = params.get_int("n_components");
        if (n > std::numeric_limits<std::int32_t>::max()) return N4M_ERR_INVALID_ARGUMENT;
        st = n4m_config_set_n_components(raw, static_cast<std::int32_t>(n));
        if (st != N4M_OK) return st;
    }
    struct Flag {
        const char* name;
        n4m_status_t (*setter)(n4m_config_t*, std::int32_t);
    };
    const Flag flags[] = {{"center_x", n4m_config_set_center_x},
                          {"scale_x", n4m_config_set_scale_x},
                          {"center_y", n4m_config_set_center_y},
                          {"scale_y", n4m_config_set_scale_y}};
    for (const Flag& f : flags) {
        if (!has(f.name)) continue;
        st = f.setter(raw, params.get_bool(f.name) ? 1 : 0);
        if (st != N4M_OK) return st;
    }
    return N4M_OK;
}

n4m_status_t export_model(const n4m_model_t* model, std::vector<StateBlock>& out) {
    std::size_t size = 0;
    n4m_status_t st = n4m_model_export_size(model, &size);
    if (st != N4M_OK) return st;
    StateBlock block;
    block.tag = kTagN4MM;
    block.bytes.resize(size);
    std::size_t written = 0;
    st = n4m_model_export_to_buffer(model, block.bytes.data(), size, &written);
    if (st != N4M_OK) return st;
    block.bytes.resize(written);
    out.push_back(std::move(block));
    return N4M_OK;
}

// Common state and operations for adapters backed by one N4MM model.
class ModelAdapter : public Adapter {
  public:
    explicit ModelAdapter(std::uint64_t caps) : caps_(caps) {}

    std::uint64_t capabilities() const noexcept override { return caps_; }
    std::int64_t n_features_in() const noexcept override { return dim(n4m_model_get_n_features); }
    std::int64_t n_outputs() const noexcept override { return dim(n4m_model_get_n_targets); }
    std::int64_t transform_cols() const noexcept override {
        return (caps_ & N4M_CAP_TRANSFORM) != 0 ? dim(n4m_model_get_n_components) : 0;
    }

    n4m_status_t predict(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                         n4m_matrix_view_t& out) const override {
        return n4m_model_predict(ctx, model_.get(), &X, &out);
    }
    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        if ((caps_ & N4M_CAP_TRANSFORM) == 0) return N4M_ERR_UNSUPPORTED;
        return n4m_model_transform(ctx, model_.get(), &X, &out);
    }
    const n4m_method_result_t* fit_result() const noexcept override { return result_.get(); }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        return export_model(model_.get(), out);
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params&,
                            const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagN4MM) {
            set_error(ctx, "estimator state must be exactly one N4MM block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        const auto& bytes = blocks[0].bytes;
        n4m_serialized_model_info_v1_t info{};
        n4m_status_t st = n4m_serialization_inspect_model_v1(bytes.data(), bytes.size(), &info);
        if (st != N4M_OK) return st;
        const bool wants_transform = (caps_ & N4M_CAP_TRANSFORM) != 0;
        const bool has_transform =
            (info.capabilities & N4M_SERIALIZED_MODEL_CAPABILITY_TRANSFORM) != 0;
        if ((info.capabilities & N4M_SERIALIZED_MODEL_CAPABILITY_PREDICT) == 0 ||
            wants_transform != has_transform) {
            set_error(ctx, "N4MM block does not match the method's capabilities");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_model_t* model = nullptr;
        st = n4m_model_import_from_buffer(ctx, bytes.data(), bytes.size(), &model);
        if (st != N4M_OK) return st;
        model_.reset(model);
        result_.reset();
        return N4M_OK;
    }

  protected:
    void reset() noexcept {
        model_.reset();
        result_.reset();
    }
    ModelPtr model_;
    ResultPtr result_;

  private:
    std::int64_t dim(n4m_status_t (*getter)(const n4m_model_t*, std::int32_t*)) const noexcept {
        std::int32_t v = 0;
        return model_ != nullptr && getter(model_.get(), &v) == N4M_OK ? v : 0;
    }
    std::uint64_t caps_;
};

// PLS / PCR through n4m_model_fit.
class ModelFitAdapter final : public ModelAdapter {
  public:
    using Configure = std::function<n4m_status_t(const Params&, n4m_config_t*)>;
    explicit ModelFitAdapter(Configure configure)
        : ModelAdapter(N4M_CAP_PREDICT | N4M_CAP_TRANSFORM | N4M_CAP_AFFINE |
                       N4M_CAP_SERIALIZABLE),
          configure_(std::move(configure)) {}

    n4m_status_t fit(n4m_context_t* ctx, const Params& params,
                     const FitInputs& in) override {
        reset();
        ConfigPtr cfg;
        n4m_status_t st = make_config(params, cfg);
        if (st == N4M_OK) st = configure_(params, cfg.get());
        if (st != N4M_OK) return st;
        n4m_model_t* model = nullptr;
        st = n4m_model_fit(ctx, cfg.get(), in.X, in.Y, &model);
        if (st == N4M_OK) model_.reset(model);
        return st;
    }

  private:
    Configure configure_;
};

// MethodResult kernels promoted to a predict-only affine N4MM model.
class AffineResultAdapter final : public ModelAdapter {
  public:
    using FitFn = std::function<n4m_status_t(n4m_context_t*, const n4m_config_t*,
                                             const Params&, const FitInputs&,
                                             n4m_method_result_t**)>;
    using Configure = std::function<n4m_status_t(const Params&, n4m_config_t*)>;
    AffineResultAdapter(FitFn fit_fn, Configure configure)
        : ModelAdapter(N4M_CAP_PREDICT | N4M_CAP_AFFINE | N4M_CAP_SERIALIZABLE),
          fit_fn_(std::move(fit_fn)),
          configure_(std::move(configure)) {}

    n4m_status_t fit(n4m_context_t* ctx, const Params& params,
                     const FitInputs& in) override {
        reset();
        ConfigPtr cfg;
        n4m_status_t st = make_config(params, cfg);
        if (st == N4M_OK && configure_) st = configure_(params, cfg.get());
        if (st != N4M_OK) return st;
        n4m_method_result_t* raw = nullptr;
        st = fit_fn_(ctx, cfg.get(), params, in, &raw);
        ResultPtr result(raw);
        if (st != N4M_OK) return st;
        n4m_model_t* model = nullptr;
        st = n4m_model_from_method_result(ctx, result.get(), &model);
        if (st != N4M_OK) return st;
        model_.reset(model);
        result_ = std::move(result);
        return N4M_OK;
    }

  private:
    FitFn fit_fn_;
    Configure configure_;
};

std::int32_t narrow(std::int64_t v) {
    if (v > std::numeric_limits<std::int32_t>::max() ||
        v < std::numeric_limits<std::int32_t>::min()) {
        throw std::out_of_range("integer parameter exceeds int32");
    }
    return static_cast<std::int32_t>(v);
}

std::unique_ptr<Adapter> affine(AffineResultAdapter::FitFn fn,
                                AffineResultAdapter::Configure configure = {}) {
    return std::make_unique<AffineResultAdapter>(std::move(fn), std::move(configure));
}

}  // namespace

std::unique_ptr<Adapter> make_pls_regression(const MethodSpec&) {
    return std::make_unique<ModelFitAdapter>([](const Params& p, n4m_config_t* cfg) {
        n4m_status_t st = n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
        if (st != N4M_OK) return st;
        return n4m_config_set_solver(cfg, static_cast<n4m_solver_t>(p.get_int("solver")));
    });
}

// Same model as n4m_estimators_pls_fit: SIMPLS with centered and scaled X/Y.
std::unique_ptr<Adapter> make_pls_fit_simple(const MethodSpec&) {
    return std::make_unique<ModelFitAdapter>([](const Params&, n4m_config_t* cfg) {
        n4m_status_t st = n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
        return st != N4M_OK ? st : n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    });
}

// Same configuration as n4m_estimators_pcr_fit: PCR + SVD.
std::unique_ptr<Adapter> make_pcr(const MethodSpec&) {
    return std::make_unique<ModelFitAdapter>([](const Params&, n4m_config_t* cfg) {
        n4m_status_t st = n4m_config_set_algorithm(cfg, N4M_ALGO_PCR);
        return st != N4M_OK ? st : n4m_config_set_solver(cfg, N4M_SOLVER_SVD);
    });
}

std::unique_ptr<Adapter> make_affine_cppls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_cppls_fit(ctx, cfg, in.X, in.Y, p.get_double("gamma"), out);
    });
}

std::unique_ptr<Adapter> make_affine_robust_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_robust_pls_fit(ctx, cfg, in.X, in.Y, p.get_double("huber_k"),
                                             narrow(p.get_int("max_irls_iter")), out);
    });
}

std::unique_ptr<Adapter> make_affine_ridge_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_ridge_pls_fit(ctx, cfg, in.X, in.Y, p.get_double("ridge_lambda"),
                                            out);
    });
}

// SIMPLS selects the canonical Stone-Brooks continuum (NIPALS keeps the
// legacy rescaling), as the n4m Python reference does.
std::unique_ptr<Adapter> make_affine_continuum_regression(const MethodSpec&) {
    return affine(
        [](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
            return n4m_estimators_continuum_regression_fit(ctx, cfg, in.X, in.Y,
                                                           p.get_double("tau"), out);
        },
        [](const Params&, n4m_config_t* cfg) {
            return n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
        });
}

std::unique_ptr<Adapter> make_affine_ridge(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        const double lambda = p.get_double("alpha");
        return n4m_estimators_ridge_fit(ctx, cfg, in.X, in.Y, &lambda, 1, out);
    });
}

std::unique_ptr<Adapter> make_affine_sparse_simpls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_sparse_simpls_fit(ctx, cfg, in.X, in.Y,
                                                p.get_double("sparsity_lambda"), out);
    });
}

std::unique_ptr<Adapter> make_affine_fused_sparse_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_fused_sparse_pls_fit(ctx, cfg, in.X, in.Y,
                                                   p.get_double("l1_lambda"),
                                                   p.get_double("fusion_lambda"), out);
    });
}

std::unique_ptr<Adapter> make_affine_group_sparse_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        std::vector<std::int32_t> groups(static_cast<std::size_t>(in.n_feature_groups));
        for (std::size_t i = 0; i < groups.size(); ++i) groups[i] = narrow(in.feature_groups[i]);
        return n4m_estimators_group_sparse_pls_fit(ctx, cfg, in.X, in.Y, groups.data(),
                                                   in.n_feature_groups,
                                                   p.get_double("group_lambda"), out);
    });
}

std::unique_ptr<Adapter> make_affine_mir_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params&, const FitInputs& in, auto out) {
        return n4m_estimators_mir_pls_fit(ctx, cfg, in.X, in.Y, out);
    });
}

std::unique_ptr<Adapter> make_affine_mb_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params&, const FitInputs& in, auto out) {
        return n4m_estimators_mb_pls_fit(ctx, cfg, in.X, in.Y, in.block_sizes, in.n_blocks, out);
    });
}

std::unique_ptr<Adapter> make_affine_ecr(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_ecr_fit(ctx, cfg, in.X, in.Y, p.get_double("alpha"), out);
    });
}

std::unique_ptr<Adapter> make_affine_n_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_estimators_n_pls_fit(ctx, cfg, in.X, narrow(p.get_int("mode_j")),
                                        narrow(p.get_int("mode_k")), in.Y, out);
    });
}

// The native DI-PLS kernel requires SIMPLS with centered, unscaled X and Y.
std::unique_ptr<Adapter> make_affine_di_pls(const MethodSpec&) {
    return affine(
        [](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
            return n4m_domain_adaptation_di_pls_fit(ctx, cfg, in.X, in.Y, in.X_target,
                                                    p.get_double("di_lambda"), out);
        },
        [](const Params&, n4m_config_t* cfg) {
            n4m_status_t st = n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
            if (st == N4M_OK) st = n4m_config_set_scale_x(cfg, 0);
            if (st == N4M_OK) st = n4m_config_set_scale_y(cfg, 0);
            return st;
        });
}

std::unique_ptr<Adapter> make_affine_bagging_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_ensemble_bagging_pls_fit(ctx, cfg, in.X, in.Y,
                                            narrow(p.get_int("n_estimators")), in.seed, out);
    });
}

std::unique_ptr<Adapter> make_affine_boosting_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_ensemble_boosting_pls_fit(ctx, cfg, in.X, in.Y,
                                             narrow(p.get_int("n_estimators")),
                                             p.get_double("learning_rate"), out);
    });
}

std::unique_ptr<Adapter> make_affine_random_subspace_pls(const MethodSpec&) {
    return affine([](auto ctx, auto cfg, const Params& p, const FitInputs& in, auto out) {
        return n4m_ensemble_random_subspace_pls_fit(
            ctx, cfg, in.X, in.Y, narrow(p.get_int("n_estimators")),
            narrow(p.get_int("features_per_subspace")), in.seed, out);
    });
}

}  // namespace n4m::estimator
