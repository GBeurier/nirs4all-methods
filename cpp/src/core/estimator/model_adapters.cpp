// SPDX-License-Identifier: CECILL-2.1
//
// Estimator adapters whose fitted state is a core kernel struct rather than
// an N4MM model: the non-affine regressors (kernel PLS, GPR-on-PLS, LW-PLS),
// missing-aware NIPALS, the OnPLS joint-score transformer and the DS / PDS
// calibration-transfer maps. Fit, predict and transform run the core
// kernels; the adapters only frame their state for N4ME.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/config.hpp"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/extra_pls.hpp"
#include "core/gpr_pls.hpp"
#include "core/kernel_pls.hpp"
#include "core/lw_pls.hpp"
#include "core/multiblock_extensions.hpp"

namespace n4m::estimator {

std::vector<n4m_matrix_view_t> block_views(const n4m_matrix_view_t& X,
                                           const std::vector<std::int64_t>& sizes) {
    std::vector<n4m_matrix_view_t> views;
    std::int64_t offset = 0;
    for (std::int64_t size : sizes) {
        n4m_matrix_view_t v = X;
        v.data = static_cast<double*>(X.data) + static_cast<std::ptrdiff_t>(offset * X.col_stride);
        v.cols = size;
        views.push_back(v);
        offset += size;
    }
    return views;
}

namespace {

constexpr std::uint32_t kTagRegressor = 0x31474552u;  // "REG1"
constexpr std::uint32_t kTagOnPls = 0x314C504Fu;      // "OPL1"

// Row-major values into a possibly strided output view.
n4m_status_t write_rows(const std::vector<double>& values, n4m_matrix_view_t& out) {
    if (values.size() != static_cast<std::size_t>(out.rows * out.cols)) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    auto* dst = static_cast<double*>(out.data);
    for (std::int64_t i = 0; i < out.rows; ++i) {
        for (std::int64_t j = 0; j < out.cols; ++j) {
            dst[i * out.row_stride + j * out.col_stride] =
                values[static_cast<std::size_t>(i * out.cols + j)];
        }
    }
    return N4M_OK;
}

void write_f64s(n4m_state_writer_t* w, const std::vector<double>& v) {
    n4m_state_write_f64_array(w, v.data(), static_cast<std::int64_t>(v.size()));
}

bool read_f64s(n4m_state_reader_t* r, std::int64_t n, std::vector<double>& out) {
    out.resize(static_cast<std::size_t>(n));
    return n4m_state_read_f64_array(r, out.data(), n) != 0;
}

// Upper bound for the row count of a retained training set in a payload.
constexpr std::int64_t kMaxRows = std::int64_t{1} << 31;

// Regressors over a core kernel state: the adapter keeps the input width and
// the target count, frames the kernel fields in one block and writes the
// kernel's row-major predictions into the caller's view.
class CoreRegressor : public Adapter {
  public:
    explicit CoreRegressor(std::uint64_t extra_caps) : caps_(extra_caps) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_PREDICT | N4M_CAP_SERIALIZABLE | caps_;
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return n_outputs_; }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        n_features_ = 0;
        n_outputs_ = 0;
        const n4m_status_t st = fit_kernel(*ctx, params, in);
        if (st != N4M_OK) return st;
        n_features_ = in.X->cols;
        n_outputs_ = in.Y->cols;
        return N4M_OK;
    }

    n4m_status_t predict(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                         n4m_matrix_view_t& out) const override {
        std::vector<double> values;
        const n4m_status_t st = predict_kernel(*ctx, X, values);
        return st != N4M_OK ? st : write_rows(values, out);
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t w;
        n4m_state_write_i64(&w, n_features_);
        n4m_state_write_i64(&w, n_outputs_);
        save_kernel(&w);
        StateBlock block;
        block.tag = kTagRegressor;
        block.bytes = std::move(w.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        n_features_ = 0;
        n_outputs_ = 0;
        if (blocks.size() != 1 || blocks[0].tag != kTagRegressor) return N4M_ERR_CORRUPT_BUFFER;
        n4m_state_reader_t r(blocks[0].bytes.data(), blocks[0].bytes.size());
        std::int64_t p = 0, q = 0;
        if (!n4m_state_read_i64(&r, &p) || !n4m_state_read_i64(&r, &q) || p <= 0 ||
            p > (std::int64_t{1} << 31) || q <= 0 || q > (std::int64_t{1} << 20) ||
            !load_kernel(params, &r, p, q) || r.remaining() != 0) {
            set_error(ctx, "regressor state does not match the method");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n_features_ = p;
        n_outputs_ = q;
        return N4M_OK;
    }

  protected:
    virtual n4m_status_t fit_kernel(core::Context& ctx, const Params& params,
                                    const FitInputs& in) = 0;
    virtual n4m_status_t predict_kernel(core::Context& ctx, const n4m_matrix_view_t& X,
                                        std::vector<double>& out) const = 0;
    virtual void save_kernel(n4m_state_writer_t* w) const = 0;
    // Restores the kernel state for p features and q targets; false when the
    // payload does not match.
    virtual bool load_kernel(const Params& params, n4m_state_reader_t* r, std::int64_t p,
                             std::int64_t q) = 0;

  private:
    std::uint64_t caps_;
    std::int64_t n_features_ = 0;
    std::int64_t n_outputs_ = 0;
};

// Non-linear kernel PLS: SIMPLS on the centered training Gram matrix; new
// rows are predicted from their kernel against the retained training rows.
class KernelPlsRegressor final : public CoreRegressor {
  public:
    KernelPlsRegressor() : CoreRegressor(N4M_CAP_RETAINS_TRAINING_ROWS) {}

  protected:
    n4m_status_t fit_kernel(core::Context& ctx, const Params& params,
                            const FitInputs& in) override {
        core::Config cfg;
        cfg.n_components = to_i32(params.get_int("n_components"));
        return core::fit_kernel_pls(ctx, cfg,
                                    static_cast<core::KernelType>(params.get_int("kernel")),
                                    params.get_double("gamma"), params.get_double("coef0"),
                                    to_i32(params.get_int("degree")), *in.X, *in.Y, model_);
    }
    n4m_status_t predict_kernel(core::Context& ctx, const n4m_matrix_view_t& X,
                                std::vector<double>& out) const override {
        return core::predict_kernel_pls(ctx, model_, X, out);
    }
    // The kernel parameters come from the estimator parameters, except gamma,
    // whose 0 default resolves to 1 / n_features at fit.
    void save_kernel(n4m_state_writer_t* w) const override {
        n4m_state_write_f64(w, model_.gamma);
        write_f64s(w, model_.x_train);
        write_f64s(w, model_.alpha);
        write_f64s(w, model_.y_mean);
        write_f64s(w, model_.K_train_row_means);
        n4m_state_write_f64(w, model_.K_train_global_mean);
    }
    bool load_kernel(const Params& params, n4m_state_reader_t* r, std::int64_t p,
                     std::int64_t q) override {
        model_ = core::KernelPlsResult{};
        std::int64_t cells = 0;
        if (!n4m_state_read_f64(r, &model_.gamma) ||
            !n4m_state_peek_array_length(r, kMaxRows * p, &cells) || cells % p != 0) {
            return false;
        }
        const std::int64_t n = cells / p;
        model_.n_train = n;
        model_.n_features = static_cast<std::int32_t>(p);
        model_.n_targets = static_cast<std::int32_t>(q);
        model_.n_components = to_i32(params.get_int("n_components"));
        model_.kernel_type = static_cast<core::KernelType>(params.get_int("kernel"));
        model_.coef0 = params.get_double("coef0");
        model_.degree = to_i32(params.get_int("degree"));
        return n >= 2 && read_f64s(r, n * p, model_.x_train) && read_f64s(r, n * q, model_.alpha) &&
               read_f64s(r, q, model_.y_mean) && read_f64s(r, n, model_.K_train_row_means) &&
               n4m_state_read_f64(r, &model_.K_train_global_mean);
    }

  private:
    core::KernelPlsResult model_;
};

// GPR on PLS scores: the posterior mean needs the rotation, the training
// scores and the GP dual weights.
class GprPlsRegressor final : public CoreRegressor {
  public:
    GprPlsRegressor() : CoreRegressor(N4M_CAP_RETAINS_TRAINING_ROWS) {}

  protected:
    n4m_status_t fit_kernel(core::Context& ctx, const Params& params,
                            const FitInputs& in) override {
        core::Config cfg;
        cfg.n_components = to_i32(params.get_int("n_components"));
        return core::fit_gpr_pls(ctx, cfg, *in.X, *in.Y, params.get_double("length_scale"),
                                 params.get_double("noise_level"), 0, model_);
    }
    n4m_status_t predict_kernel(core::Context& ctx, const n4m_matrix_view_t& X,
                                std::vector<double>& out) const override {
        return core::predict_gpr_pls(ctx, model_, X, out);
    }
    void save_kernel(n4m_state_writer_t* w) const override {
        write_f64s(w, model_.x_mean);
        write_f64s(w, model_.rotation_r);
        write_f64s(w, model_.gp.T_train);
        write_f64s(w, model_.gp.alpha);
        n4m_state_write_f64(w, model_.gp.y_mean_scalar);
    }
    bool load_kernel(const Params& params, n4m_state_reader_t* r, std::int64_t p,
                     std::int64_t q) override {
        model_ = core::GprPlsResult{};
        const std::int64_t k = params.get_int("n_components");
        std::int64_t cells = 0;
        if (q != 1 || k > p || !read_f64s(r, p, model_.x_mean) ||
            !read_f64s(r, p * k, model_.rotation_r) ||
            !n4m_state_peek_array_length(r, kMaxRows * k, &cells) || cells % k != 0) {
            return false;
        }
        const std::int64_t n = cells / k;
        model_.n_features = static_cast<std::int32_t>(p);
        model_.n_samples = static_cast<std::int32_t>(n);
        model_.n_components = static_cast<std::int32_t>(k);
        model_.length_scale = params.get_double("length_scale");
        model_.noise_level = params.get_double("noise_level");
        model_.gp.n_train = static_cast<std::int32_t>(n);
        model_.gp.n_components = static_cast<std::int32_t>(k);
        model_.gp.length_scale = model_.length_scale;
        model_.gp.noise_level = model_.noise_level;
        return n >= 1 && read_f64s(r, n * k, model_.gp.T_train) &&
               read_f64s(r, n, model_.gp.alpha) && n4m_state_read_f64(r, &model_.gp.y_mean_scalar);
    }

  private:
    core::GprPlsResult model_;
};

// LW-PLS refits a local PLS per query row, so its state is the training set.
class LwPlsRegressor final : public CoreRegressor {
  public:
    LwPlsRegressor() : CoreRegressor(N4M_CAP_RETAINS_TRAINING_ROWS) {}

  protected:
    n4m_status_t fit_kernel(core::Context& ctx, const Params& params,
                            const FitInputs& in) override {
        configure(params);
        core::LwPlsResult result;
        const n4m_status_t st =
            core::fit_predict_lw_pls(ctx, cfg_, *in.X, *in.Y, n_neighbors_, result);
        if (st != N4M_OK) return st;
        std::vector<double> x_storage, y_storage;
        const n4m_matrix_view_t x = contiguous_view(*in.X, x_storage);
        const n4m_matrix_view_t y = contiguous_view(*in.Y, y_storage);
        const auto* xd = static_cast<const double*>(x.data);
        const auto* yd = static_cast<const double*>(y.data);
        x_train_.assign(xd, xd + static_cast<std::ptrdiff_t>(x.rows * x.cols));
        y_train_.assign(yd, yd + static_cast<std::ptrdiff_t>(y.rows * y.cols));
        n_train_ = x.rows;
        return N4M_OK;
    }
    n4m_status_t predict_kernel(core::Context& ctx, const n4m_matrix_view_t& X,
                                std::vector<double>& out) const override {
        return core::predict_lw_pls(ctx, cfg_, x_train_, y_train_, n_train_, n_neighbors_, X,
                                    out);
    }
    void save_kernel(n4m_state_writer_t* w) const override {
        write_f64s(w, x_train_);
        write_f64s(w, y_train_);
    }
    bool load_kernel(const Params& params, n4m_state_reader_t* r, std::int64_t p,
                     std::int64_t q) override {
        configure(params);
        std::int64_t cells = 0;
        if (!n4m_state_peek_array_length(r, kMaxRows * p, &cells) || cells % p != 0) return false;
        n_train_ = cells / p;
        return n_train_ >= n_neighbors_ && read_f64s(r, cells, x_train_) &&
               read_f64s(r, n_train_ * q, y_train_);
    }

  private:
    // The Gaussian-weighted mode is the kernel default; SIMPLS selects its
    // k-NN cutoff variant.
    void configure(const Params& params) {
        cfg_ = core::Config{};
        cfg_.n_components = to_i32(params.get_int("n_components"));
        if (params.get_int("mode") == 1) cfg_.solver = N4M_SOLVER_SIMPLS;
        n_neighbors_ = to_i32(params.get_int("n_neighbors"));
    }
    core::Config cfg_;
    std::int32_t n_neighbors_ = 0;
    std::vector<double> x_train_, y_train_;
    std::int64_t n_train_ = 0;
};

// Missing-aware NIPALS: mean-imputed SIMPLS; prediction imputes the same way.
class MissingAwareNipalsRegressor final : public CoreRegressor {
  public:
    MissingAwareNipalsRegressor() : CoreRegressor(0) {}

  protected:
    n4m_status_t fit_kernel(core::Context& ctx, const Params& params,
                            const FitInputs& in) override {
        core::Config cfg;
        cfg.n_components = to_i32(params.get_int("n_components"));
        return core::fit_missing_aware_nipals(ctx, cfg, *in.X, *in.Y, model_);
    }
    n4m_status_t predict_kernel(core::Context&, const n4m_matrix_view_t& X,
                                std::vector<double>& out) const override {
        core::predict_missing_aware_nipals(model_, X, out);
        return N4M_OK;
    }
    void save_kernel(n4m_state_writer_t* w) const override {
        write_f64s(w, model_.coefficients);
        write_f64s(w, model_.x_mean);
        write_f64s(w, model_.y_mean);
    }
    bool load_kernel(const Params&, n4m_state_reader_t* r, std::int64_t p,
                     std::int64_t q) override {
        model_ = core::WeightedPlsResult{};
        model_.n_features = static_cast<std::int32_t>(p);
        model_.n_targets = static_cast<std::int32_t>(q);
        return read_f64s(r, p * q, model_.coefficients) && read_f64s(r, p, model_.x_mean) &&
               read_f64s(r, q, model_.y_mean);
    }

  private:
    core::WeightedPlsResult model_;
};

// OnPLS as a transformer: the joint scores of every block, concatenated in
// block order (n × n_blocks·n_joint).
class OnPlsTransformer final : public Adapter {
  public:
    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override {
        return static_cast<std::int64_t>(sizes_.size()) * model_.n_joint;
    }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        reset();
        const std::vector<std::int64_t> unique = params.get_ints("n_unique_per_block");
        std::vector<std::int32_t> unique32(unique.size());
        for (std::size_t b = 0; b < unique.size(); ++b) unique32[b] = to_i32(unique[b]);
        std::vector<std::int64_t> sizes(in.block_sizes, in.block_sizes + in.n_blocks);
        const n4m_status_t st =
            core::fit_on_pls(*ctx, core::Config{}, block_views(*in.X, sizes),
                             to_i32(params.get_int("n_joint")), unique32, model_);
        if (st != N4M_OK) {
            reset();
            return st;
        }
        sizes_ = std::move(sizes);
        n_features_ = in.X->cols;
        return N4M_OK;
    }

    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        std::vector<std::vector<double>> scores;
        const n4m_status_t st = core::on_pls_joint_scores(*ctx, model_, block_views(X, sizes_), scores);
        if (st != N4M_OK) return st;
        const auto k = static_cast<std::size_t>(model_.n_joint);
        const auto nb = sizes_.size();
        std::vector<double> values(static_cast<std::size_t>(X.rows) * nb * k);
        for (std::size_t i = 0; i < static_cast<std::size_t>(X.rows); ++i) {
            for (std::size_t b = 0; b < nb; ++b) {
                for (std::size_t c = 0; c < k; ++c) {
                    values[(i * nb + b) * k + c] = scores[b][i * k + c];
                }
            }
        }
        return write_rows(values, out);
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t w;
        n4m_state_write_i64_array(&w, sizes_.data(), static_cast<std::int64_t>(sizes_.size()));
        for (std::size_t b = 0; b < sizes_.size(); ++b) {
            write_f64s(&w, model_.x_mean_per_block[b]);
            write_f64s(&w, model_.unique_weights_per_block[b]);
            write_f64s(&w, model_.unique_loadings_per_block[b]);
            write_f64s(&w, model_.joint_weights_per_block[b]);
            write_f64s(&w, model_.joint_loadings_per_block[b]);
        }
        StateBlock block;
        block.tag = kTagOnPls;
        block.bytes = std::move(w.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        reset();
        if (blocks.size() != 1 || blocks[0].tag != kTagOnPls || !load(params, blocks[0])) {
            reset();
            set_error(ctx, "OnPLS state does not match the method");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        return N4M_OK;
    }

  private:
    bool load(const Params& params, const StateBlock& block) {
        n4m_state_reader_t r(block.bytes.data(), block.bytes.size());
        const std::vector<std::int64_t> unique = params.get_ints("n_unique_per_block");
        const auto nb = static_cast<std::int64_t>(unique.size());
        const std::int64_t k = params.get_int("n_joint");
        std::int64_t n_blocks = 0;
        if (!n4m_state_peek_array_length(&r, nb, &n_blocks) || n_blocks != nb || nb < 2) {
            return false;
        }
        sizes_.resize(static_cast<std::size_t>(nb));
        if (!n4m_state_read_i64_array(&r, sizes_.data(), nb)) return false;
        model_.n_blocks = static_cast<std::int32_t>(nb);
        model_.n_joint = static_cast<std::int32_t>(k);
        model_.x_mean_per_block.assign(sizes_.size(), {});
        model_.unique_weights_per_block.assign(sizes_.size(), {});
        model_.unique_loadings_per_block.assign(sizes_.size(), {});
        model_.joint_weights_per_block.assign(sizes_.size(), {});
        model_.joint_loadings_per_block.assign(sizes_.size(), {});
        for (std::size_t b = 0; b < sizes_.size(); ++b) {
            const std::int64_t p = sizes_[b];
            std::int64_t extracted = 0;
            if (p <= 0 || p > (std::int64_t{1} << 31) || !read_f64s(&r, p, model_.x_mean_per_block[b]) ||
                !n4m_state_peek_array_length(&r, unique[b] * p, &extracted) || extracted % p != 0 ||
                !read_f64s(&r, extracted, model_.unique_weights_per_block[b]) ||
                !read_f64s(&r, p * unique[b], model_.unique_loadings_per_block[b]) ||
                !read_f64s(&r, p * k, model_.joint_weights_per_block[b]) ||
                !read_f64s(&r, p * k, model_.joint_loadings_per_block[b])) {
                return false;
            }
            model_.n_unique_per_block.push_back(static_cast<std::int32_t>(unique[b]));
            n_features_ += p;
        }
        return r.remaining() == 0;
    }

    void reset() noexcept {
        model_ = core::OnPlsResult{};
        sizes_.clear();
        n_features_ = 0;
    }

    core::OnPlsResult model_;
    std::vector<std::int64_t> sizes_;
    std::int64_t n_features_ = 0;
};

template <typename T>
void destroy_result(T* h) {
    delete h;
}

// DS / PDS: paired source (X) -> target-domain maps; fit and state are set
// by the method.
template <typename T>
FittedKernel<T> transfer_map(n4m_status_t (*apply)(const T&, const n4m_matrix_view_t&,
                                                   std::vector<double>&)) {
    FittedKernel<T> kernel;
    kernel.create = [](T** h, const Params&) {
        *h = new T();
        return N4M_OK;
    };
    kernel.transform = [apply](const T* h, n4m_matrix_view_t x, n4m_matrix_view_t out) {
        std::vector<double> values;
        const n4m_status_t st = apply(*h, x, values);
        return st != N4M_OK ? st : write_rows(values, out);
    };
    kernel.destroy = destroy_result<T>;
    return kernel;
}

}  // namespace

std::unique_ptr<Adapter> make_kernel_pls(const MethodSpec&) {
    return std::make_unique<KernelPlsRegressor>();
}

std::unique_ptr<Adapter> make_gpr_pls(const MethodSpec&) {
    return std::make_unique<GprPlsRegressor>();
}

std::unique_ptr<Adapter> make_lw_pls(const MethodSpec&) {
    return std::make_unique<LwPlsRegressor>();
}

std::unique_ptr<Adapter> make_missing_aware_nipals(const MethodSpec&) {
    return std::make_unique<MissingAwareNipalsRegressor>();
}

std::unique_ptr<Adapter> make_tr_on_pls(const MethodSpec&) {
    return std::make_unique<OnPlsTransformer>();
}

std::unique_ptr<Adapter> make_tr_ds(const MethodSpec&) {
    using T = core::DsResult;
    FittedKernel<T> kernel = transfer_map<T>(core::apply_ds);
    kernel.fit = [](n4m_context_t* ctx, T* h, const FitInputs& in) {
        return core::fit_ds(*ctx, *in.X, *in.X_target, *h);
    };
    kernel.save = [](const T* h, n4m_state_writer_t* w) {
        write_f64s(w, h->transformation);
        write_f64s(w, h->bias);
        return N4M_OK;
    };
    kernel.load = [](T* h, n4m_state_reader_t* r, std::int64_t p) {
        return read_f64s(r, p * p, h->transformation) && read_f64s(r, p, h->bias)
                   ? N4M_OK
                   : N4M_ERR_CORRUPT_BUFFER;
    };
    return fitted(std::move(kernel));
}

std::unique_ptr<Adapter> make_tr_pds(const MethodSpec&) {
    using T = core::PdsResult;
    FittedKernel<T> kernel = transfer_map<T>(core::apply_pds);
    // The half-width is kept on the result, which fit_pds resets and refills.
    kernel.create = [](T** h, const Params& p) {
        *h = new T();
        (*h)->window_half_width = to_i32(p.get_int("window_half_width"));
        return N4M_OK;
    };
    kernel.fit = [](n4m_context_t* ctx, T* h, const FitInputs& in) {
        return core::fit_pds(*ctx, *in.X, *in.X_target, h->window_half_width, *h);
    };
    kernel.save = [](const T* h, n4m_state_writer_t* w) {
        write_f64s(w, h->transformation);
        return N4M_OK;
    };
    kernel.load = [](T* h, n4m_state_reader_t* r, std::int64_t p) {
        return read_f64s(r, p * p, h->transformation) ? N4M_OK : N4M_ERR_CORRUPT_BUFFER;
    };
    return fitted(std::move(kernel));
}

}  // namespace n4m::estimator
