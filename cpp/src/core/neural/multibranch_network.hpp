// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Jointly trained multibranch neural regressor (models.neural.multibranch_regressor).
//
// X is the concatenation of ordered blocks. Each block feeds one branch:
//   dense   [width]  -> embedding (dense layer, leaky ReLU);
//   conv1d  [length] -> valid 1-D convolution (filters x kernel_size), leaky
//                       ReLU, average pooling (pool_size), dense embedding;
//   conv2d  [H,W,C]  -> valid 2-D convolution (filters x C x k x k) on the HWC
//                       block, leaky ReLU, pool_size x pool_size average
//                       pooling, dense embedding.
// The branch embeddings are concatenated in block order and a common head
// (optional hidden layer, then a linear output) regresses the standardized
// targets. Every convolution kernel, embedding and head weight is trained
// jointly by backpropagation of the mean squared error, with Adam and
// decoupled weight decay, for a fixed number of epochs. Input and target
// standardization is fitted on the training rows only. Initialization and
// minibatch order come from the core PCG64 engine seeded with a u64.
//
// conv1d is computed by the same code as conv2d with H = length, W = 1,
// C = 1, a kernel_size x 1 kernel and a pool_size x 1 window.
//
// Internal C++ (never part of the public C ABI). Invalid declarations throw
// std::invalid_argument with a message naming the branch; the estimator
// adapter maps it to N4M_ERR_INVALID_ARGUMENT.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/rng_pcg64.h"
#include "core/estimator/state_io.h"

namespace n4m::core::neural {

constexpr std::int64_t kKindDense = 0;
constexpr std::int64_t kKindConv1d = 1;
constexpr std::int64_t kKindConv2d = 2;

// Admission budget: trainable parameters and per-sample activation values.
constexpr std::size_t kMaxParameters = 2000000;
constexpr std::size_t kMaxActivations = 16777216;
// Negative slope of the leaky ReLU used after every hidden layer.
constexpr double kLeakySlope = 0.01;
// Adam constants.
constexpr double kAdamBeta1 = 0.9;
constexpr double kAdamBeta2 = 0.999;
constexpr double kAdamEpsilon = 1e-8;

// Declared branch: kind and 3 dimensions (dense/conv1d: [width,1,1];
// conv2d: [H,W,C]).
struct BranchShape {
    std::int64_t kind = kKindDense;
    std::int64_t d0 = 0;
    std::int64_t d1 = 1;
    std::int64_t d2 = 1;
};

// Architecture hyperparameters shared by every branch.
struct Hyper {
    std::int64_t filters = 4;
    std::int64_t kernel_size = 5;
    std::int64_t pool_size = 2;
    std::int64_t embedding_dim = 8;
    std::int64_t head_hidden = 8;
};

// Optimization settings (provenance: not needed to predict).
struct Training {
    std::int64_t epochs = 100;
    std::int64_t batch_size = 16;
    double learning_rate = 0.01;
    double weight_decay = 1e-4;
    std::uint64_t seed = 0;
};

// One parameter tensor inside the flat parameter vector.
struct Slice {
    std::size_t offset = 0;
    std::size_t size = 0;
    bool decay = false;  // weights/kernels decay, biases do not
};

// Resolved geometry of one branch.
struct BranchLayout {
    BranchShape shape;
    std::size_t width = 0;   // block columns
    std::size_t offset = 0;  // first block column in X
    bool conv = false;
    // Convolution geometry (conv1d: in_w = 1, in_c = 1, k_w = 1, p_w = 1).
    std::size_t in_h = 0, in_w = 1, in_c = 1;
    std::size_t k_h = 1, k_w = 1, p_h = 1, p_w = 1;
    std::size_t out_h = 0, out_w = 0;    // convolution output
    std::size_t pool_h = 0, pool_w = 0;  // pooled output
    std::size_t conv_size = 0;           // filters * out_h * out_w
    std::size_t embed_in = 0;            // dense input width of the embedding
    std::size_t n_scale = 0;             // standardization statistics
    Slice conv_k, conv_b, emb_w, emb_b;
};

class MultibranchNetwork {
  public:
    // Validates the declaration against the block widths (their sum is
    // n_features) and lays out the parameters, zero-initialized.
    void configure(std::int64_t n_cols, std::int64_t n_out,
                   const std::vector<BranchShape>& shapes, const Hyper& h);

    // Fits the standardization on X/Y (row-major, contiguous, finite), seeds
    // and trains the network. Returns the loss history: epochs + 1 values of
    // the training-set MSE in standardized target units (index 0: initial
    // weights).
    std::vector<double> fit(const double* X, const double* Y, std::size_t n, const Training& t);

    // Row-major contiguous X (n x n_features, finite) -> row-major out.
    void predict(const double* X, std::size_t n, double* out) const;          // n x n_targets
    void transform(const double* X, std::size_t n, double* out) const;        // n x embedding_width

    void save(n4m_state_writer_t* w) const;
    // Restores architecture, scaling and weights; false on any mismatch.
    bool load(n4m_state_reader_t* r);

    std::size_t n_features() const noexcept { return p_; }
    std::size_t n_targets() const noexcept { return q_; }
    std::size_t n_branches() const noexcept { return branches_.size(); }
    std::size_t embedding_width() const noexcept { return z_size_; }
    std::size_t n_parameters() const noexcept { return theta_.size(); }
    const Hyper& hyper() const noexcept { return hyper_; }
    const std::vector<BranchLayout>& branches() const noexcept { return branches_; }

    // ---- Numerical surface shared by fit and the gradient probe ----------
    // Training-only statistics of X and Y.
    void fit_scaling(const double* X, const double* Y, std::size_t n);
    void standardize(const double* X, std::size_t n, double* Xs) const;
    void standardize_targets(const double* Y, std::size_t n, double* Ys) const;
    // He/Glorot-uniform weights and zero biases drawn from `rng` in the fixed
    // parameter order.
    void initialize(n4m_rng_pcg64* rng);
    // Mean squared error over `rows` of the standardized Xs/Ys (sum over
    // rows and targets divided by n_rows * n_targets). When `grad` is not
    // null it receives the exact gradient with respect to parameters().
    double loss_gradient(const double* Xs, const double* Ys, const std::size_t* rows,
                         std::size_t n_rows, std::vector<double>* grad) const;
    std::vector<double>& parameters() noexcept { return theta_; }
    const std::vector<double>& parameters() const noexcept { return theta_; }

  private:
    struct Work;
    void forward(const double* xs, Work& w) const;
    void backward(const double* xs, const Work& w, const double* g_out, Work& scratch,
                  std::vector<double>& grad) const;
    Work make_work() const;
    void unstandardized_outputs(const Work& w, double* out) const;

    std::size_t p_ = 0;
    std::size_t q_ = 0;
    Hyper hyper_;
    std::vector<BranchLayout> branches_;
    std::size_t z_size_ = 0;  // concatenated embedding width
    std::size_t u_size_ = 0;  // head output-layer input width
    Slice hid_w_, hid_b_, out_w_, out_b_;
    std::vector<double> theta_;
    // Standardization: per branch, mean/std of its n_scale statistics.
    std::vector<std::vector<double>> x_mean_, x_std_;
    std::vector<double> y_mean_, y_std_;
};

}  // namespace n4m::core::neural
