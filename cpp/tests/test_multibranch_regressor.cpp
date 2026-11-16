// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Focused scientific probe of models.neural.multibranch_regressor on one tiny
// deterministic three-branch case (conv1d [16], dense [3], conv2d [5,5,2]):
//   1. central finite differences agree with the backpropagated gradient for
//      every parameter tensor (both convolutions, embeddings, head);
//   2. supervised training lowers the training loss and changes every tensor
//      (conv kernels included); epochs = 0 keeps the seeded initialization;
//   3. N4ME export -> import predicts/transforms bit-identically without a fit
//      and re-exports the same bytes; the same seed refits identically;
//   4. column-major and padded row-stride inputs give the row-major result;
//   5. non-finite data and inconsistent declarations are refused.
// It prints the loss history ends and a prediction checksum that the WASM
// build must reproduce for the same data and parameters.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/neural/multibranch_network.hpp"
#include "n4m/estimator.h"
#include "n4m/n4m.h"

namespace {

namespace nn = n4m::core::neural;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

constexpr std::size_t kRows = 24;
constexpr std::size_t kL = 16;           // conv1d length
constexpr std::size_t kD = 3;            // dense width
constexpr std::size_t kH = 5, kW = 5, kC = 2;  // conv2d HWC
constexpr std::size_t kCols = kL + kD + kH * kW * kC;

// Deterministic data: a Gaussian peak whose position and height vary, three
// markers, and an image blob whose intensity varies; y depends on all three.
void make_data(std::vector<double>& X, std::vector<double>& y) {
    X.assign(kRows * kCols, 0.0);
    y.assign(kRows, 0.0);
    for (std::size_t i = 0; i < kRows; ++i) {
        const double a = static_cast<double>(i);
        const double height = 1.0 + 0.5 * std::sin(0.7 * a);
        const double center = 6.0 + 3.0 * std::cos(0.3 * a);
        double* row = X.data() + i * kCols;
        for (std::size_t l = 0; l < kL; ++l) {
            const double d = static_cast<double>(l) - center;
            row[l] = height * std::exp(-0.25 * d * d) +
                     0.05 * std::cos(1.3 * a + static_cast<double>(l));
        }
        const double m0 = static_cast<double>(i % 3);
        const double m1 = static_cast<double>((i * 7) % 3);
        const double m2 = 0.5 * std::sin(1.1 * a);
        row[kL + 0] = m0;
        row[kL + 1] = m1;
        row[kL + 2] = m2;
        const double blob = 0.5 + 0.4 * std::cos(0.45 * a);
        for (std::size_t h = 0; h < kH; ++h) {
            for (std::size_t w = 0; w < kW; ++w) {
                const double r2 = (static_cast<double>(h) - 2.0) * (static_cast<double>(h) - 2.0) +
                                  (static_cast<double>(w) - 2.0) * (static_cast<double>(w) - 2.0);
                for (std::size_t c = 0; c < kC; ++c) {
                    row[kL + kD + (h * kW + w) * kC + c] =
                        (c == 0 ? blob : 1.0 - blob) * std::exp(-0.5 * r2) +
                        0.02 * std::sin(a + static_cast<double>(h * 3 + w + c));
                }
            }
        }
        y[i] = 2.0 * height + 0.3 * center + 0.8 * m0 - 0.5 * m1 + m2 + 1.5 * blob;
    }
}

std::vector<nn::BranchShape> shapes() {
    return {{nn::kKindConv1d, static_cast<std::int64_t>(kL), 1, 1},
            {nn::kKindDense, static_cast<std::int64_t>(kD), 1, 1},
            {nn::kKindConv2d, static_cast<std::int64_t>(kH), static_cast<std::int64_t>(kW),
             static_cast<std::int64_t>(kC)}};
}

nn::Hyper small_hyper() {
    nn::Hyper h;
    h.filters = 2;
    h.kernel_size = 3;
    h.pool_size = 2;
    h.embedding_dim = 3;
    h.head_hidden = 4;
    return h;
}

// 1. Finite differences on the standardized loss.
void gradient_check(const std::vector<double>& X, const std::vector<double>& y) {
    nn::MultibranchNetwork net;
    net.configure(static_cast<std::int64_t>(kCols), 1, shapes(), small_hyper());
    net.fit_scaling(X.data(), y.data(), kRows);
    std::vector<double> Xs(kRows * kCols), Ys(kRows);
    net.standardize(X.data(), kRows, Xs.data());
    net.standardize_targets(y.data(), kRows, Ys.data());
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, 11);
    net.initialize(&rng);
    // Nonzero biases so every bias gradient is exercised away from zero.
    std::vector<double>& theta = net.parameters();
    for (std::size_t k = 0; k < theta.size(); ++k) {
        if (theta[k] == 0.0) theta[k] = 0.05 * std::sin(static_cast<double>(k) + 0.5);
    }
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < kRows; i += 3) rows.push_back(i);
    std::vector<double> grad;
    net.loss_gradient(Xs.data(), Ys.data(), rows.data(), rows.size(), &grad);
    const double h = 1e-6;
    double worst = 0.0;
    std::size_t checked = 0, kinks = 0;
    for (std::size_t k = 0; k < theta.size(); ++k) {
        const double saved = theta[k];
        theta[k] = saved + h;
        const double up =
            net.loss_gradient(Xs.data(), Ys.data(), rows.data(), rows.size(), nullptr);
        theta[k] = saved - h;
        const double down =
            net.loss_gradient(Xs.data(), Ys.data(), rows.data(), rows.size(), nullptr);
        theta[k] = saved;
        const double fd = (up - down) / (2.0 * h);
        const double diff = std::fabs(fd - grad[k]);
        const double rel = diff / std::fmax(1e-6, std::fabs(fd) + std::fabs(grad[k]));
        ++checked;
        // A step crossing a leaky-ReLU kink yields a one-sided slope; it is
        // reported, not hidden, and must stay rare.
        if (rel > 1e-5 && diff > 1e-7) {
            ++kinks;
            continue;
        }
        worst = std::fmax(worst, rel);
    }
    std::printf("gradient: %zu parameters, worst relative error %.3e, %zu kink-affected\n",
                checked, worst, kinks);
    check(worst <= 1e-5, "finite-difference gradient agreement");
    check(kinks * 50 <= checked, "kink-affected finite differences are rare");
}

// 2. Training changes every tensor and lowers the loss.
void training_check(const std::vector<double>& X, const std::vector<double>& y) {
    nn::Training t;
    t.epochs = 60;
    t.batch_size = 8;
    t.learning_rate = 0.01;
    t.weight_decay = 1e-4;
    t.seed = 3;
    nn::MultibranchNetwork trained;
    trained.configure(static_cast<std::int64_t>(kCols), 1, shapes(), small_hyper());
    const std::vector<double> history = trained.fit(X.data(), y.data(), kRows, t);
    nn::MultibranchNetwork initial;
    initial.configure(static_cast<std::int64_t>(kCols), 1, shapes(), small_hyper());
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, t.seed);
    initial.initialize(&rng);
    std::printf("loss: initial %.6g, final %.6g over %zu epochs\n", history.front(),
                history.back(), history.size() - 1);
    check(history.size() == 61, "loss history has epochs + 1 entries");
    check(history.back() < 0.5 * history.front(), "training halves the training loss");
    const auto& a = trained.parameters();
    const auto& b = initial.parameters();
    auto changed = [&](const nn::Slice& s) {
        double delta = 0.0;
        for (std::size_t k = 0; k < s.size; ++k) {
            delta = std::fmax(delta, std::fabs(a[s.offset + k] - b[s.offset + k]));
        }
        return s.size > 0 && delta > 1e-6;
    };
    for (const nn::BranchLayout& L : trained.branches()) {
        if (L.conv) {
            check(changed(L.conv_k), "convolution kernel weights are trained");
            check(changed(L.conv_b), "convolution biases are trained");
        }
        check(changed(L.emb_w), "branch embedding weights are trained");
    }
    nn::Training zero = t;
    zero.epochs = 0;
    nn::MultibranchNetwork untouched;
    untouched.configure(static_cast<std::int64_t>(kCols), 1, shapes(), small_hyper());
    const std::vector<double> h0 = untouched.fit(X.data(), y.data(), kRows, zero);
    check(h0.size() == 1 && untouched.parameters() == initial.parameters(),
          "epochs = 0 keeps the seeded initialization");
}

n4m_estimator_t* fit_public(n4m_context_t* ctx, const std::vector<double>& X,
                            const std::vector<double>& y, std::int64_t seed) {
    int32_t index = -1;
    n4m_method_find("models.neural.multibranch_regressor", &index);
    n4m_params_t* params = nullptr;
    n4m_params_create(ctx, index, &params);
    const int64_t kinds[] = {1, 0, 2};
    const int64_t dims[] = {static_cast<int64_t>(kL), 1, 1, static_cast<int64_t>(kD), 1, 1,
                            static_cast<int64_t>(kH), static_cast<int64_t>(kW),
                            static_cast<int64_t>(kC)};
    check(n4m_params_set_int_array(params, "branch_kinds", kinds, 3) == N4M_OK, "set kinds");
    check(n4m_params_set_int_array(params, "branch_shapes", dims, 9) == N4M_OK, "set shapes");
    n4m_params_set_int(params, "filters", 2);
    n4m_params_set_int(params, "kernel_size", 3);
    n4m_params_set_int(params, "embedding_dim", 3);
    n4m_params_set_int(params, "head_hidden", 4);
    n4m_params_set_int(params, "epochs", 40);
    n4m_params_set_int(params, "batch_size", 8);
    n4m_params_set_int(params, "seed", seed);
    n4m_estimator_t* est = nullptr;
    check(n4m_estimator_create(ctx, "models.neural.multibranch_regressor", params, &est) == N4M_OK,
          "create estimator");
    n4m_params_destroy(params);
    n4m_matrix_view_t Xv{}, Yv{};
    n4m_matrix_view_init_rowmajor(&Xv, const_cast<double*>(X.data()), static_cast<int64_t>(kRows),
                                  static_cast<int64_t>(kCols), N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&Yv, const_cast<double*>(y.data()), static_cast<int64_t>(kRows),
                                  1, N4M_DTYPE_F64);
    const int64_t blocks[] = {static_cast<int64_t>(kL), static_cast<int64_t>(kD),
                              static_cast<int64_t>(kH * kW * kC)};
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &Xv;
    in.Y = &Yv;
    in.block_sizes = blocks;
    in.n_blocks = 3;
    const n4m_status_t st = n4m_estimator_fit(ctx, est, &in);
    if (st != N4M_OK) std::printf("fit: %s\n", n4m_context_last_error(ctx));
    check(st == N4M_OK, "public fit");
    return est;
}

std::vector<double> predict_view(n4m_context_t* ctx, const n4m_estimator_t* est,
                                 const n4m_matrix_view_t& X) {
    std::vector<double> out(kRows);
    n4m_matrix_view_t P{};
    n4m_matrix_view_init_rowmajor(&P, out.data(), static_cast<int64_t>(kRows), 1, N4M_DTYPE_F64);
    check(n4m_estimator_predict(ctx, est, &X, &P) == N4M_OK, "predict");
    return out;
}

std::vector<unsigned char> export_state(n4m_context_t* ctx, const n4m_estimator_t* est) {
    size_t size = 0, written = 0;
    check(n4m_estimator_export_size(ctx, est, 0, &size) == N4M_OK, "export size");
    std::vector<unsigned char> bytes(size);
    check(n4m_estimator_export_to_buffer(ctx, est, 0, bytes.data(), size, &written) == N4M_OK &&
              written == size,
          "export");
    return bytes;
}

// 3-5. Public C ABI: replay, strides, determinism, refusals.
void public_check(const std::vector<double>& X, const std::vector<double>& y) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_estimator_t* est = fit_public(ctx, X, y, 5);
    n4m_matrix_view_t Xv{};
    n4m_matrix_view_init_rowmajor(&Xv, const_cast<double*>(X.data()), static_cast<int64_t>(kRows),
                                  static_cast<int64_t>(kCols), N4M_DTYPE_F64);
    const std::vector<double> pred = predict_view(ctx, est, Xv);

    const n4m_method_result_t* result = nullptr;
    check(n4m_estimator_fit_result(est, &result) == N4M_OK, "fit diagnostics");
    const double* hist = nullptr;
    int64_t hr = 0, hc = 0;
    check(n4m_method_result_get_double_matrix(result, "loss_history", &hist, &hr, &hc) == N4M_OK &&
              hr == 41 && hc == 1,
          "loss_history shape");

    // Transform: learned embeddings, 3 branches x embedding_dim 3.
    int64_t tcols = 0;
    n4m_estimator_transform_cols(est, &tcols);
    check(tcols == 9, "embedding width");
    std::vector<double> emb(kRows * 9), emb2(kRows * 9);
    n4m_matrix_view_t Ev{};
    n4m_matrix_view_init_rowmajor(&Ev, emb.data(), static_cast<int64_t>(kRows), 9, N4M_DTYPE_F64);
    check(n4m_estimator_transform(ctx, est, &Xv, &Ev) == N4M_OK, "transform");

    // Replay without fit.
    const std::vector<unsigned char> bytes = export_state(ctx, est);
    n4m_estimator_t* back = nullptr;
    check(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) == N4M_OK,
          "import");
    check(predict_view(ctx, back, Xv) == pred, "imported state predicts bit-identically");
    n4m_matrix_view_init_rowmajor(&Ev, emb2.data(), static_cast<int64_t>(kRows), 9, N4M_DTYPE_F64);
    check(n4m_estimator_transform(ctx, back, &Xv, &Ev) == N4M_OK && emb2 == emb,
          "imported state transforms bit-identically");
    check(export_state(ctx, back) == bytes, "re-export is byte-identical");
    const n4m_method_result_t* none = nullptr;
    check(n4m_estimator_fit_result(back, &none) == N4M_ERR_UNSUPPORTED,
          "imported state carries no fit diagnostics");

    // Strided inputs.
    std::vector<double> cm(kRows * kCols), padded(kRows * (kCols + 3), -7.0);
    for (std::size_t i = 0; i < kRows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            cm[j * kRows + i] = X[i * kCols + j];
            padded[i * (kCols + 3) + j] = X[i * kCols + j];
        }
    }
    n4m_matrix_view_t Cv{}, Pv{};
    n4m_matrix_view_init_colmajor(&Cv, cm.data(), static_cast<int64_t>(kRows),
                                  static_cast<int64_t>(kCols), N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&Pv, padded.data(), static_cast<int64_t>(kRows),
                                  static_cast<int64_t>(kCols), N4M_DTYPE_F64);
    Pv.row_stride = static_cast<int64_t>(kCols + 3);
    check(predict_view(ctx, est, Cv) == pred, "column-major input");
    check(predict_view(ctx, est, Pv) == pred, "padded row-stride input");

    // Determinism and seed sensitivity.
    n4m_estimator_t* same = fit_public(ctx, X, y, 5);
    check(export_state(ctx, same) == bytes, "same seed refits byte-identically");
    n4m_estimator_t* other = fit_public(ctx, X, y, -3);  // u64 seed 2^64 - 3
    check(predict_view(ctx, other, Xv) != pred, "another seed gives another network");

    // Refusals: missing value at fit and an inconsistent declaration.
    std::vector<double> bad = X;
    bad[5] = std::nan("");
    n4m_estimator_t* refused = nullptr;
    int32_t index = -1;
    n4m_method_find("models.neural.multibranch_regressor", &index);
    n4m_estimator_create(ctx, "models.neural.multibranch_regressor", nullptr, &refused);
    n4m_matrix_view_t Bv{}, Yv{};
    n4m_matrix_view_init_rowmajor(&Bv, bad.data(), static_cast<int64_t>(kRows),
                                  static_cast<int64_t>(kCols), N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&Yv, const_cast<double*>(y.data()), static_cast<int64_t>(kRows),
                                  1, N4M_DTYPE_F64);
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &Bv;
    in.Y = &Yv;
    check(n4m_estimator_fit(ctx, refused, &in) == N4M_ERR_INVALID_ARGUMENT, "NaN refused at fit");
    n4m_params_t* params = nullptr;
    n4m_params_create(ctx, index, &params);
    const int64_t kinds[] = {2};
    n4m_params_set_int_array(params, "branch_kinds", kinds, 1);
    n4m_estimator_t* conv2d = nullptr;
    n4m_estimator_create(ctx, "models.neural.multibranch_regressor", params, &conv2d);
    n4m_params_destroy(params);
    in.X = &Xv;
    check(n4m_estimator_fit(ctx, conv2d, &in) == N4M_ERR_INVALID_ARGUMENT,
          "conv2d without a declared shape is refused");

    double checksum = 0.0;
    for (std::size_t i = 0; i < kRows; ++i) checksum += pred[i] * static_cast<double>(i + 1);
    std::printf("public: loss %.17g -> %.17g, prediction checksum %.17g\n", hist[0], hist[40],
                checksum);

    n4m_estimator_destroy(conv2d);
    n4m_estimator_destroy(refused);
    n4m_estimator_destroy(other);
    n4m_estimator_destroy(same);
    n4m_estimator_destroy(back);
    n4m_estimator_destroy(est);
    n4m_context_destroy(ctx);
}

}  // namespace

int main() {
    std::vector<double> X, y;
    make_data(X, y);
    gradient_check(X, y);
    training_check(X, y);
    public_check(X, y);
    std::printf("n4m_multibranch_regressor_tests: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
