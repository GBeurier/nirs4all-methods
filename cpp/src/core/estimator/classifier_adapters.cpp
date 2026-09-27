// SPDX-License-Identifier: CECILL-2.1
//
// Classifier role adapters. Class labels are arbitrary int64 ids; the fitted
// state keeps them sorted and the kernels see 0-based indices. Every
// classifier exposes labels and its decision scores; probabilities only when
// the method defines them (logistic: softmax of its logits; QDA: normalized
// posteriors). Decisions are computed by the core kernels.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "core/config.hpp"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/extra_pls.hpp"
#include "core/pls_lda.hpp"
#include "core/pls_logistic.hpp"

namespace n4m::estimator {

namespace {

constexpr std::uint32_t kTagClassifier = 0x31534C43u;  // "CLS1"

struct ModelDeleter {
    void operator()(n4m_model_t* m) const noexcept { n4m_model_destroy(m); }
};
struct ConfigDeleter {
    void operator()(n4m_config_t* c) const noexcept { n4m_config_destroy(c); }
};
using ModelPtr = std::unique_ptr<n4m_model_t, ModelDeleter>;
using ConfigPtr = std::unique_ptr<n4m_config_t, ConfigDeleter>;

n4m_status_t export_model(const n4m_model_t* model, n4m_state_writer_t* w) {
    std::size_t size = 0;
    n4m_status_t st = n4m_model_export_size(model, &size);
    if (st != N4M_OK) return st;
    std::vector<unsigned char> bytes(size);
    std::size_t written = 0;
    st = n4m_model_export_to_buffer(model, bytes.data(), size, &written);
    if (st != N4M_OK) return st;
    // Stored as an i64 array of bytes is wasteful; pack 8 bytes per value.
    n4m_state_write_i64(w, static_cast<std::int64_t>(written));
    std::vector<double> packed((written + 7) / 8, 0.0);
    std::memcpy(packed.data(), bytes.data(), written);
    n4m_state_write_f64_array(w, packed.data(), static_cast<std::int64_t>(packed.size()));
    return N4M_OK;
}

n4m_status_t import_model(n4m_context_t* ctx, n4m_state_reader_t* r, ModelPtr& out) {
    std::int64_t size = 0;
    if (!n4m_state_read_i64(r, &size) || size <= 0 || size > (std::int64_t{1} << 30)) {
        return N4M_ERR_CORRUPT_BUFFER;
    }
    std::vector<double> packed(static_cast<std::size_t>((size + 7) / 8));
    if (!n4m_state_read_f64_array(r, packed.data(), static_cast<std::int64_t>(packed.size()))) {
        return N4M_ERR_CORRUPT_BUFFER;
    }
    n4m_model_t* raw = nullptr;
    const n4m_status_t st =
        n4m_model_import_from_buffer(ctx, packed.data(), static_cast<std::size_t>(size), &raw);
    if (st == N4M_OK) out.reset(raw);
    return st;
}

// Common life cycle: label encoding, decision -> labels, state framing.
class ClassifierAdapter : public Adapter {
  public:
    explicit ClassifierAdapter(bool probabilities) : probabilities_(probabilities) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_PREDICT_LABELS | N4M_CAP_DECISION_FUNCTION | N4M_CAP_SERIALIZABLE |
               (probabilities_ ? N4M_CAP_PREDICT_PROBA : 0);
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override {
        return static_cast<std::int64_t>(classes_.size());
    }
    const std::vector<std::int64_t>* classes() const noexcept override { return &classes_; }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        classes_.assign(in.labels, in.labels + in.n_labels);
        std::sort(classes_.begin(), classes_.end());
        classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
        if (classes_.size() < 2) {
            classes_.clear();
            set_error(ctx, "a classifier needs at least two classes");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<std::int32_t> encoded(static_cast<std::size_t>(in.n_labels));
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            encoded[i] = static_cast<std::int32_t>(
                std::lower_bound(classes_.begin(), classes_.end(), in.labels[i]) -
                classes_.begin());
        }
        n_features_ = in.X->cols;
        const n4m_status_t st = fit_encoded(ctx, params, in, encoded);
        if (st != N4M_OK) {
            classes_.clear();
            n_features_ = 0;
        }
        return st;
    }

    n4m_status_t decision_function(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                                   n4m_matrix_view_t& out) const override {
        std::vector<double> decision, proba;
        n4m_status_t st = scores(ctx, X, decision, proba);
        if (st == N4M_OK) st = write(decision, out);
        return st;
    }

    n4m_status_t predict_proba(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                               n4m_matrix_view_t& out) const override {
        std::vector<double> decision, proba;
        n4m_status_t st = scores(ctx, X, decision, proba);
        if (st == N4M_OK) st = write(proba, out);
        return st;
    }

    n4m_status_t predict_labels(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                                std::int64_t* out) const override {
        std::vector<double> decision, proba;
        const n4m_status_t st = scores(ctx, X, decision, proba);
        if (st != N4M_OK) return st;
        const std::size_t c = classes_.size();
        for (std::int64_t i = 0; i < X.rows; ++i) {
            const double* row = decision.data() + static_cast<std::size_t>(i) * c;
            out[i] = classes_[static_cast<std::size_t>(std::max_element(row, row + c) - row)];
        }
        return N4M_OK;
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t w;
        n4m_state_write_i64(&w, n_features_);
        n4m_state_write_i64_array(&w, classes_.data(), static_cast<std::int64_t>(classes_.size()));
        const n4m_status_t st = save_head(&w);
        if (st != N4M_OK) return st;
        StateBlock block;
        block.tag = kTagClassifier;
        block.bytes = std::move(w.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagClassifier) return N4M_ERR_CORRUPT_BUFFER;
        n4m_state_reader_t r(blocks[0].bytes.data(), blocks[0].bytes.size());
        std::int64_t width = 0, n_classes = 0;
        if (!n4m_state_read_i64(&r, &width) || width <= 0 ||
            !n4m_state_peek_array_length(&r, std::int64_t{1} << 20, &n_classes) || n_classes < 2) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        classes_.resize(static_cast<std::size_t>(n_classes));
        if (!n4m_state_read_i64_array(&r, classes_.data(), n_classes) ||
            !std::is_sorted(classes_.begin(), classes_.end()) ||
            std::adjacent_find(classes_.begin(), classes_.end()) != classes_.end()) {
            classes_.clear();
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n_features_ = width;
        const n4m_status_t st = load_head(ctx, params, &r);
        if (st != N4M_OK || r.remaining() != 0) {
            classes_.clear();
            n_features_ = 0;
            return N4M_ERR_CORRUPT_BUFFER;
        }
        return N4M_OK;
    }

  protected:
    // Fits on 0-based labels; the class count is classes_.size().
    virtual n4m_status_t fit_encoded(n4m_context_t* ctx, const Params& params,
                                     const FitInputs& in,
                                     const std::vector<std::int32_t>& labels) = 0;
    // Decision (n x C) and, when defined, probabilities (n x C).
    virtual n4m_status_t scores(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                                std::vector<double>& decision,
                                std::vector<double>& proba) const = 0;
    virtual n4m_status_t save_head(n4m_state_writer_t* w) const = 0;
    virtual n4m_status_t load_head(n4m_context_t* ctx, const Params& params,
                                   n4m_state_reader_t* r) = 0;

    std::int32_t n_classes() const noexcept { return static_cast<std::int32_t>(classes_.size()); }

    // An embedded N4MM class-score model maps the fitted width to one column
    // per class.
    n4m_status_t check_model_shape(const n4m_model_t* model) const {
        std::int32_t p = 0, c = 0;
        if (n4m_model_get_n_features(model, &p) != N4M_OK ||
            n4m_model_get_n_targets(model, &c) != N4M_OK || p != n_features_ ||
            c != n_classes()) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        return N4M_OK;
    }

  private:
    static n4m_status_t write(const std::vector<double>& values, n4m_matrix_view_t& out) {
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

    bool probabilities_;
    std::vector<std::int64_t> classes_;
    std::int64_t n_features_ = 0;
};

// Heads on the scores of a PLS model fitted to the one-hot class matrix
// (SIMPLS, centered, unscaled: the n4m reference configuration).
class PlsHeadClassifier : public ClassifierAdapter {
  public:
    PlsHeadClassifier(bool probabilities, n4m_algorithm_t algorithm)
        : ClassifierAdapter(probabilities), algorithm_(algorithm) {}

  protected:
    n4m_status_t fit_encoded(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                             const std::vector<std::int32_t>& labels) override {
        const std::int64_t n = in.X->rows;
        const std::int32_t c = n_classes();
        std::vector<double> dummy(static_cast<std::size_t>(n * c), 0.0);
        for (std::size_t i = 0; i < labels.size(); ++i) {
            dummy[i * static_cast<std::size_t>(c) + static_cast<std::size_t>(labels[i])] = 1.0;
        }
        n4m_matrix_view_t Y{};
        n4m_matrix_view_init_rowmajor(&Y, dummy.data(), n, c, N4M_DTYPE_F64);
        n4m_config_t* raw_cfg = nullptr;
        n4m_status_t st = n4m_config_create(&raw_cfg);
        if (st != N4M_OK) return st;
        ConfigPtr cfg(raw_cfg);
        const std::int32_t k = to_i32(params.get_int("n_components"));
        st = n4m_config_set_algorithm(raw_cfg, algorithm_);
        if (st == N4M_OK) st = n4m_config_set_solver(raw_cfg, N4M_SOLVER_SIMPLS);
        if (st == N4M_OK) st = n4m_config_set_deflation(raw_cfg, N4M_DEFLATION_REGRESSION);
        if (st == N4M_OK) st = n4m_config_set_n_components(raw_cfg, k);
        if (st == N4M_OK) st = n4m_config_set_scale_x(raw_cfg, 0);
        if (st == N4M_OK) st = n4m_config_set_scale_y(raw_cfg, 0);
        if (st != N4M_OK) return st;
        n4m_model_t* raw_model = nullptr;
        st = n4m_model_fit(ctx, raw_cfg, in.X, &Y, &raw_model);
        if (st != N4M_OK) return st;
        model_.reset(raw_model);
        std::vector<double> train_scores;
        st = latent_scores(ctx, *in.X, train_scores);
        return st != N4M_OK ? st : fit_head(ctx, params, train_scores, labels, n, k);
    }

    n4m_status_t scores(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                        std::vector<double>& decision, std::vector<double>& proba) const override {
        std::vector<double> latent;
        const n4m_status_t st = latent_scores(ctx, X, latent);
        if (st != N4M_OK) return st;
        const std::size_t size = static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(n_classes());
        decision.assign(size, 0.0);
        proba.assign(size, 0.0);
        head_scores(latent.data(), X.rows, decision.data(), proba.data());
        return N4M_OK;
    }

    n4m_status_t save_head(n4m_state_writer_t* w) const override {
        const n4m_status_t st = export_model(model_.get(), w);
        return st != N4M_OK ? st : save_head_params(w);
    }

    n4m_status_t load_head(n4m_context_t* ctx, const Params& params,
                           n4m_state_reader_t* r) override {
        n4m_status_t st = import_model(ctx, r, model_);
        if (st == N4M_OK) st = check_model_shape(model_.get());
        return st != N4M_OK ? st : load_head_params(params, r);
    }

    // The latent model records the fitted component count.
    n4m_status_t check_params(n4m_context_t* ctx, const Params& params) const override {
        return check_int(ctx, params, "n_components", components());
    }

    virtual n4m_status_t fit_head(n4m_context_t* ctx, const Params& params,
                                  const std::vector<double>& scores,
                                  const std::vector<std::int32_t>& labels, std::int64_t n,
                                  std::int32_t k) = 0;
    virtual void head_scores(const double* latent, std::int64_t n, double* decision,
                             double* proba) const = 0;
    virtual n4m_status_t save_head_params(n4m_state_writer_t* w) const = 0;
    virtual n4m_status_t load_head_params(const Params& params, n4m_state_reader_t* r) = 0;

    std::int32_t components() const noexcept {
        std::int32_t k = 0;
        n4m_model_get_n_components(model_.get(), &k);
        return k;
    }

  private:
    n4m_status_t latent_scores(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                               std::vector<double>& out) const {
        const std::int32_t k = components();
        out.assign(static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(k), 0.0);
        n4m_matrix_view_t T{};
        n4m_matrix_view_init_rowmajor(&T, out.data(), X.rows, k, N4M_DTYPE_F64);
        return n4m_model_transform(ctx, model_.get(), &X, &T);
    }

    n4m_algorithm_t algorithm_;
    ModelPtr model_;
};

class PlsLdaClassifier final : public PlsHeadClassifier {
  public:
    PlsLdaClassifier() : PlsHeadClassifier(false, N4M_ALGO_PLS_DA) {}

  protected:
    n4m_status_t fit_head(n4m_context_t* ctx, const Params&, const std::vector<double>& scores,
                          const std::vector<std::int32_t>& labels, std::int64_t n,
                          std::int32_t k) override {
        if (n <= n_classes()) {
            set_error(ctx, "PLS-LDA requires more samples than classes");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        return core::fit_pls_lda_head(*ctx, scores, labels, n, k, n_classes(), head_);
    }
    void head_scores(const double* latent, std::int64_t n, double* decision,
                     double*) const override {
        core::pls_lda_decision(head_, latent, n, decision);
    }
    n4m_status_t save_head_params(n4m_state_writer_t* w) const override {
        n4m_state_write_f64_array(w, head_.inv_means.data(),
                                  static_cast<std::int64_t>(head_.inv_means.size()));
        n4m_state_write_f64_array(w, head_.constants.data(),
                                  static_cast<std::int64_t>(head_.constants.size()));
        return N4M_OK;
    }
    n4m_status_t load_head_params(const Params&, n4m_state_reader_t* r) override {
        head_.n_classes = n_classes();
        head_.n_components = components();
        head_.inv_means.resize(static_cast<std::size_t>(head_.n_classes * head_.n_components));
        head_.constants.resize(static_cast<std::size_t>(head_.n_classes));
        const bool ok =
            n4m_state_read_f64_array(r, head_.inv_means.data(),
                                     static_cast<std::int64_t>(head_.inv_means.size())) &&
            n4m_state_read_f64_array(r, head_.constants.data(), head_.n_classes);
        return ok ? N4M_OK : N4M_ERR_CORRUPT_BUFFER;
    }

  private:
    core::PlsLdaHead head_;
};

class PlsLogisticClassifier final : public PlsHeadClassifier {
  public:
    PlsLogisticClassifier() : PlsHeadClassifier(true, N4M_ALGO_PLS_REGRESSION) {}

  protected:
    n4m_status_t fit_head(n4m_context_t* ctx, const Params& params,
                          const std::vector<double>& scores,
                          const std::vector<std::int32_t>& labels, std::int64_t n,
                          std::int32_t k) override {
        return core::fit_pls_logistic_head(*ctx, scores, labels, n, k, n_classes(),
                                           to_i32(params.get_int("max_iter")), head_);
    }
    void head_scores(const double* latent, std::int64_t n, double* decision,
                     double* proba) const override {
        core::pls_logistic_predict(head_.intercepts, head_.coefficients, n_classes(),
                                   components(), latent, n, decision, proba);
    }
    n4m_status_t save_head_params(n4m_state_writer_t* w) const override {
        n4m_state_write_f64_array(w, head_.intercepts.data(),
                                  static_cast<std::int64_t>(head_.intercepts.size()));
        n4m_state_write_f64_array(w, head_.coefficients.data(),
                                  static_cast<std::int64_t>(head_.coefficients.size()));
        return N4M_OK;
    }
    n4m_status_t load_head_params(const Params&, n4m_state_reader_t* r) override {
        const auto tail = static_cast<std::size_t>(n_classes() - 1);
        head_.intercepts.resize(tail);
        head_.coefficients.resize(tail * static_cast<std::size_t>(components()));
        const bool ok =
            n4m_state_read_f64_array(r, head_.intercepts.data(), static_cast<std::int64_t>(tail)) &&
            n4m_state_read_f64_array(r, head_.coefficients.data(),
                                     static_cast<std::int64_t>(head_.coefficients.size()));
        return ok ? N4M_OK : N4M_ERR_CORRUPT_BUFFER;
    }

  private:
    core::PlsLogisticResult head_;
};

// PLS-QDA: the kernel's latent model and class Gaussians; decision from
// core::pls_qda_decision, probabilities the normalized posteriors.
class PlsQdaClassifier final : public ClassifierAdapter {
  public:
    PlsQdaClassifier() : ClassifierAdapter(true) {}

  protected:
    n4m_status_t fit_encoded(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                             const std::vector<std::int32_t>& labels) override {
        core::Config cfg;
        cfg.n_components = to_i32(params.get_int("n_components"));
        return core::fit_pls_qda(*ctx, cfg, *in.X, labels, model_);
    }

    n4m_status_t scores(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                        std::vector<double>& decision, std::vector<double>& proba) const override {
        const n4m_status_t st = core::pls_qda_decision(*ctx, model_, X, decision);
        if (st != N4M_OK) return st;
        const auto c = static_cast<std::size_t>(model_.n_classes);
        proba.assign(decision.size(), 0.0);
        for (std::size_t i = 0; i * c < decision.size(); ++i) {
            const double* row = decision.data() + i * c;
            const double top = *std::max_element(row, row + c);
            double sum = 0.0;
            for (std::size_t j = 0; j < c; ++j) sum += proba[i * c + j] = std::exp(row[j] - top);
            for (std::size_t j = 0; j < c; ++j) proba[i * c + j] /= sum;
        }
        return N4M_OK;
    }

    n4m_status_t save_head(n4m_state_writer_t* w) const override {
        n4m_state_write_i64(w, model_.n_components);
        for (const auto* v : {&model_.x_mean, &model_.rotations_r, &model_.class_means,
                              &model_.class_covariances, &model_.log_class_priors}) {
            n4m_state_write_f64_array(w, v->data(), static_cast<std::int64_t>(v->size()));
        }
        return N4M_OK;
    }

    // The kernel keeps min(n_components, n - 1, p) components; the training
    // row count is not stored, so the requested count is an upper bound.
    n4m_status_t check_params(n4m_context_t* ctx, const Params& params) const override {
        return model_.n_components <= params.get_int("n_components") ? N4M_OK
                                                                     : contradicts(ctx, "n_components");
    }

    n4m_status_t load_head(n4m_context_t*, const Params&, n4m_state_reader_t* r) override {
        std::int64_t k = 0;
        if (!n4m_state_read_i64(r, &k) || k <= 0 || k > (std::int64_t{1} << 16)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        const auto p = static_cast<std::size_t>(n_features_in());
        const auto a = static_cast<std::size_t>(k);
        const auto c = static_cast<std::size_t>(n_classes());
        model_.n_components = static_cast<std::int32_t>(k);
        model_.n_classes = n_classes();
        model_.x_mean.resize(p);
        model_.rotations_r.resize(p * a);
        model_.class_means.resize(c * a);
        model_.class_covariances.resize(c * a * a);
        model_.log_class_priors.resize(c);
        for (auto* v : {&model_.x_mean, &model_.rotations_r, &model_.class_means,
                        &model_.class_covariances, &model_.log_class_priors}) {
            if (!n4m_state_read_f64_array(r, v->data(), static_cast<std::int64_t>(v->size()))) {
                return N4M_ERR_CORRUPT_BUFFER;
            }
        }
        return N4M_OK;
    }

  private:
    core::PlsQdaResult model_;
};

// Sparse PLS-DA: affine class scores from the kernel's coefficients.
class SparsePlsDaClassifier final : public ClassifierAdapter {
  public:
    SparsePlsDaClassifier() : ClassifierAdapter(false) {}

  protected:
    n4m_status_t fit_encoded(n4m_context_t* ctx, const Params& params, const FitInputs& in,
                             const std::vector<std::int32_t>& labels) override {
        n4m_config_t* raw_cfg = nullptr;
        n4m_status_t st = n4m_config_create(&raw_cfg);
        if (st != N4M_OK) return st;
        ConfigPtr cfg(raw_cfg);
        st = n4m_config_set_n_components(raw_cfg, to_i32(params.get_int("n_components")));
        if (st == N4M_OK) st = n4m_config_set_solver(raw_cfg, N4M_SOLVER_SIMPLS);
        if (st == N4M_OK) st = n4m_config_set_scale_x(raw_cfg, 0);
        if (st == N4M_OK) st = n4m_config_set_scale_y(raw_cfg, 0);
        if (st != N4M_OK) return st;
        // The kernel reads the soft-threshold from the core config, which has
        // no public setter.
        static_cast<core::Config*>(raw_cfg)->sparsity_lambda = params.get_double("sparsity_lambda");
        n4m_method_result_t* raw = nullptr;
        st = n4m_estimators_sparse_pls_da_fit(ctx, raw_cfg, in.X, labels.data(),
                                              static_cast<std::int64_t>(labels.size()), &raw);
        if (st != N4M_OK) return st;
        std::unique_ptr<n4m_method_result_t, void (*)(n4m_method_result_t*)> result(
            raw, n4m_method_result_destroy);
        // Class scores are the affine map (x - x_mean) B + y_mean; the result's
        // "predictions" are one-hot argmax rows, so it is not an affine result.
        const double *coef = nullptr, *x_mean = nullptr, *y_mean = nullptr;
        std::int64_t p = 0, c = 0, rows = 0, cols = 0;
        st = n4m_method_result_get_double_matrix(result.get(), "coefficients", &coef, &p, &c);
        if (st == N4M_OK) st = n4m_method_result_get_double_matrix(result.get(), "x_mean", &x_mean, &rows, &cols);
        if (st == N4M_OK) st = n4m_method_result_get_double_matrix(result.get(), "y_mean", &y_mean, &rows, &cols);
        if (st != N4M_OK) return st;
        std::vector<double> intercept(static_cast<std::size_t>(c));
        for (std::int64_t j = 0; j < c; ++j) {
            double v = y_mean[j];
            for (std::int64_t f = 0; f < p; ++f) v -= x_mean[f] * coef[f * c + j];
            intercept[static_cast<std::size_t>(j)] = v;
        }
        n4m_linear_predictor_spec_t spec{};
        spec.source_training_samples = in.X->rows;
        spec.n_features = to_i32(p);
        spec.n_targets = to_i32(c);
        spec.coefficients = coef;
        spec.intercept = intercept.data();
        n4m_model_t* model = nullptr;
        st = n4m_model_import_linear_predictor(ctx, &spec, &model);
        if (st == N4M_OK) model_.reset(model);
        return st;
    }

    n4m_status_t scores(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                        std::vector<double>& decision, std::vector<double>&) const override {
        decision.assign(static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(n_classes()), 0.0);
        n4m_matrix_view_t out{};
        n4m_matrix_view_init_rowmajor(&out, decision.data(), X.rows, n_classes(), N4M_DTYPE_F64);
        return n4m_model_predict(ctx, model_.get(), &X, &out);
    }

    n4m_status_t save_head(n4m_state_writer_t* w) const override { return export_model(model_.get(), w); }
    n4m_status_t load_head(n4m_context_t* ctx, const Params&, n4m_state_reader_t* r) override {
        const n4m_status_t st = import_model(ctx, r, model_);
        return st != N4M_OK ? st : check_model_shape(model_.get());
    }

    // The affine class-score map keeps no latent dimension: n_components and
    // sparsity_lambda are provenance.
    n4m_status_t check_params(n4m_context_t*, const Params&) const override { return N4M_OK; }

  private:
    ModelPtr model_;
};

}  // namespace

std::unique_ptr<Adapter> make_cls_pls_lda(const MethodSpec&) {
    return std::make_unique<PlsLdaClassifier>();
}
std::unique_ptr<Adapter> make_cls_pls_logistic(const MethodSpec&) {
    return std::make_unique<PlsLogisticClassifier>();
}
std::unique_ptr<Adapter> make_cls_pls_qda(const MethodSpec&) {
    return std::make_unique<PlsQdaClassifier>();
}
std::unique_ptr<Adapter> make_cls_sparse_pls_da(const MethodSpec&) {
    return std::make_unique<SparsePlsDaClassifier>();
}

}  // namespace n4m::estimator
