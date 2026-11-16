// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Jointly trained multibranch neural regressor (see multibranch_network.hpp).
//
// Determinism: every loop has a fixed order, the training path uses only
// +, -, *, / and sqrt, and no threads are used, so a build without
// floating-point contraction reproduces the same weights on every platform
// (native and WASM included) for the same data, parameters and seed.

#include "core/neural/multibranch_network.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/estimator/state_io.hpp"

namespace n4m::core::neural {

namespace {

constexpr std::int64_t kStateVersion = 1;
// Bound for every declared dimension (also keeps products far from overflow).
constexpr std::int64_t kMaxDim = std::int64_t{1} << 31;

[[noreturn]] void invalid(const std::string& message) { throw std::invalid_argument(message); }

std::string branch_name(std::size_t b) { return "multibranch branch " + std::to_string(b); }

inline double act(double v) noexcept { return v > 0.0 ? v : kLeakySlope * v; }
inline double act_grad(double pre) noexcept { return pre > 0.0 ? 1.0 : kLeakySlope; }

std::size_t to_size(std::int64_t v) { return static_cast<std::size_t>(v); }

// Statistic threshold: a spread this small relative to the level is a
// constant (roundoff only); it is then not rescaled.
double safe_std(double variance, double mean) {
    const double sd = std::sqrt(variance > 0.0 ? variance : 0.0);
    const double level = std::fabs(mean) > 1.0 ? std::fabs(mean) : 1.0;
    return sd > 1e-10 * level ? sd : 1.0;
}

void write_array(n4m_state_writer_t* w, const std::vector<double>& v) {
    n4m_state_write_f64_array(w, v.data(), static_cast<std::int64_t>(v.size()));
}

void write_slice(n4m_state_writer_t* w, const std::vector<double>& theta, const Slice& s) {
    n4m_state_write_f64_array(w, theta.data() + s.offset, static_cast<std::int64_t>(s.size));
}

bool read_array(n4m_state_reader_t* r, std::vector<double>& v) {
    return n4m_state_read_f64_array(r, v.data(), static_cast<std::int64_t>(v.size())) != 0;
}

bool read_slice(n4m_state_reader_t* r, std::vector<double>& theta, const Slice& s) {
    return n4m_state_read_f64_array(r, theta.data() + s.offset,
                                    static_cast<std::int64_t>(s.size)) != 0;
}

bool all_finite(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

}  // namespace

// Per-sample activations (forward) or gradients with respect to them
// (backward scratch), sized from the architecture.
struct MultibranchNetwork::Work {
    std::vector<std::vector<double>> conv_pre;  // per branch: filters * out_h * out_w
    std::vector<std::vector<double>> pooled;    // per branch: embed_in (conv only)
    std::vector<std::vector<double>> emb_pre;   // per branch: embedding_dim
    std::vector<double> z;                      // concatenated embeddings
    std::vector<double> hid_pre;                // head hidden pre-activations
    std::vector<double> hid;                    // head hidden activations
    std::vector<double> out;                    // standardized outputs
};

void MultibranchNetwork::configure(std::int64_t n_cols, std::int64_t n_out,
                                   const std::vector<BranchShape>& shapes,
                                   const Hyper& h) {
    if (n_cols <= 0 || n_cols > kMaxDim) invalid("multibranch: invalid feature count");
    if (n_out <= 0 || n_out > (std::int64_t{1} << 20)) {
        invalid("multibranch: invalid target count");
    }
    if (shapes.empty()) invalid("multibranch: at least one branch is required");
    if (h.filters < 1 || h.filters > 64 || h.kernel_size < 1 ||
        h.kernel_size > 31 || h.pool_size < 1 || h.pool_size > 16 ||
        h.embedding_dim < 1 || h.embedding_dim > 256 || h.head_hidden < 0 ||
        h.head_hidden > 256) {
        invalid("multibranch: architecture parameter out of range");
    }
    p_ = to_size(n_cols);
    q_ = to_size(n_out);
    hyper_ = h;
    branches_.clear();
    const std::size_t F = to_size(h.filters);
    const std::size_t K = to_size(h.kernel_size);
    const std::size_t P = to_size(h.pool_size);
    const std::size_t E = to_size(h.embedding_dim);
    std::size_t offset = 0;
    std::size_t n_params = 0;
    auto take = [&n_params](std::size_t size, bool decay) {
        Slice s;
        s.offset = n_params;
        s.size = size;
        s.decay = decay;
        n_params += size;
        if (n_params > kMaxParameters) {
            invalid("multibranch: the architecture exceeds the parameter budget (" +
                    std::to_string(kMaxParameters) + ")");
        }
        return s;
    };
    for (std::size_t b = 0; b < shapes.size(); ++b) {
        const BranchShape& s = shapes[b];
        const std::string name = branch_name(b);
        if (s.d0 < 1 || s.d1 < 1 || s.d2 < 1 || s.d0 > kMaxDim || s.d1 > kMaxDim ||
            s.d2 > kMaxDim) {
            invalid(name + ": shape dimensions must be positive");
        }
        BranchLayout L;
        L.shape = s;
        if (s.kind == kKindDense || s.kind == kKindConv1d) {
            if (s.d1 != 1 || s.d2 != 1) {
                invalid(name + ": a dense or conv1d shape is [width, 1, 1]");
            }
            L.width = to_size(s.d0);
        } else if (s.kind == kKindConv2d) {
            // 64-bit products: size_t is 32 bits on wasm32.
            const auto hw = static_cast<std::uint64_t>(s.d0) * static_cast<std::uint64_t>(s.d1);
            if (hw > p_ || hw * static_cast<std::uint64_t>(s.d2) > p_) {
                invalid(name + ": shape H x W x C exceeds the input width");
            }
            L.width = static_cast<std::size_t>(hw * static_cast<std::uint64_t>(s.d2));
        } else {
            invalid(name + ": kind must be 0 (dense), 1 (conv1d) or 2 (conv2d)");
        }
        if (L.width > p_ - offset) invalid(name + ": shape product exceeds the remaining columns");
        L.offset = offset;
        offset += L.width;
        L.conv = s.kind != kKindDense;
        if (L.conv) {
            L.in_h = to_size(s.d0);
            L.in_w = s.kind == kKindConv2d ? to_size(s.d1) : 1;
            L.in_c = s.kind == kKindConv2d ? to_size(s.d2) : 1;
            L.k_h = K;
            L.k_w = s.kind == kKindConv2d ? K : 1;
            L.p_h = P;
            L.p_w = s.kind == kKindConv2d ? P : 1;
            if (L.in_h < L.k_h || L.in_w < L.k_w) {
                invalid(name + ": kernel_size is larger than the branch input");
            }
            L.out_h = L.in_h - L.k_h + 1;
            L.out_w = L.in_w - L.k_w + 1;
            if (L.out_h < L.p_h || L.out_w < L.p_w) {
                invalid(name + ": pool_size is larger than the convolution output");
            }
            L.pool_h = L.out_h / L.p_h;
            L.pool_w = L.out_w / L.p_w;
            // out_h * out_w <= width <= p_; bound before multiplying by F.
            if (L.out_h * L.out_w > kMaxActivations / F ||
                L.in_c > kMaxParameters / (F * L.k_h * L.k_w)) {
                invalid(name + ": convolution exceeds the activation or parameter budget");
            }
            L.conv_size = F * L.out_h * L.out_w;
            if (L.conv_size > kMaxActivations) {
                invalid(name + ": convolution output exceeds the activation budget (" +
                        std::to_string(kMaxActivations) + ")");
            }
            L.embed_in = F * L.pool_h * L.pool_w;
            L.n_scale = L.in_c;
        } else {
            L.embed_in = L.width;
            L.n_scale = L.width;
        }
        branches_.push_back(L);
    }
    if (offset != p_) invalid("multibranch: branch shapes do not cover the input columns");
    // Parameter layout: per branch, conv kernel and bias, embedding weights and
    // bias; then head hidden weights and bias, output weights and bias. This is
    // also the initialization and serialization order.
    std::size_t activations = 0;
    for (BranchLayout& L : branches_) {
        if (L.conv) {
            L.conv_k = take(F * L.in_c * L.k_h * L.k_w, true);
            L.conv_b = take(F, false);
            activations += 2 * L.conv_size + L.embed_in;
        }
        if (L.embed_in > kMaxParameters / E) invalid("multibranch: embedding exceeds the budget");
        L.emb_w = take(E * L.embed_in, true);
        L.emb_b = take(E, false);
        activations += E;
    }
    z_size_ = E * branches_.size();
    const std::size_t H = to_size(h.head_hidden);
    u_size_ = H > 0 ? H : z_size_;
    if (H > 0) {
        hid_w_ = take(H * z_size_, true);
        hid_b_ = take(H, false);
    } else {
        hid_w_ = Slice{};
        hid_b_ = Slice{};
    }
    if (q_ > kMaxParameters / u_size_) invalid("multibranch: the output layer exceeds the budget");
    out_w_ = take(q_ * u_size_, true);
    out_b_ = take(q_, false);
    activations += z_size_ + 2 * H + q_;
    if (activations > kMaxActivations) {
        invalid("multibranch: per-sample activations exceed the budget (" +
                std::to_string(kMaxActivations) + ")");
    }
    theta_.assign(n_params, 0.0);
    x_mean_.assign(branches_.size(), {});
    x_std_.assign(branches_.size(), {});
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        x_mean_[b].assign(branches_[b].n_scale, 0.0);
        x_std_[b].assign(branches_[b].n_scale, 1.0);
    }
    y_mean_.assign(q_, 0.0);
    y_std_.assign(q_, 1.0);
}

void MultibranchNetwork::fit_scaling(const double* X, const double* Y, std::size_t n) {
    const double inv_n = 1.0 / static_cast<double>(n);
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        const BranchLayout& L = branches_[b];
        std::vector<double>& mean = x_mean_[b];
        std::vector<double>& sd = x_std_[b];
        if (!L.conv) {
            // Dense: per column.
            for (std::size_t j = 0; j < L.width; ++j) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) s += X[i * p_ + L.offset + j];
                const double m = s * inv_n;
                double v = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double d = X[i * p_ + L.offset + j] - m;
                    v += d * d;
                }
                mean[j] = m;
                sd[j] = safe_std(v * inv_n, m);
            }
            continue;
        }
        // conv1d: one statistic for the block; conv2d: one per channel (HWC).
        const std::size_t C = L.in_c;
        const std::size_t per_channel = L.width / C;
        const double inv_count = 1.0 / (static_cast<double>(n) * static_cast<double>(per_channel));
        for (std::size_t c = 0; c < C; ++c) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double* row = X + i * p_ + L.offset;
                for (std::size_t k = 0; k < per_channel; ++k) s += row[k * C + c];
            }
            const double m = s * inv_count;
            double v = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double* row = X + i * p_ + L.offset;
                for (std::size_t k = 0; k < per_channel; ++k) {
                    const double d = row[k * C + c] - m;
                    v += d * d;
                }
            }
            mean[c] = m;
            sd[c] = safe_std(v * inv_count, m);
        }
    }
    for (std::size_t t = 0; t < q_; ++t) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += Y[i * q_ + t];
        const double m = s * inv_n;
        double v = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = Y[i * q_ + t] - m;
            v += d * d;
        }
        y_mean_[t] = m;
        y_std_[t] = safe_std(v * inv_n, m);
    }
}

void MultibranchNetwork::standardize(const double* X, std::size_t n, double* Xs) const {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t b = 0; b < branches_.size(); ++b) {
            const BranchLayout& L = branches_[b];
            const double* src = X + i * p_ + L.offset;
            double* dst = Xs + i * p_ + L.offset;
            for (std::size_t j = 0; j < L.width; ++j) {
                const std::size_t s = L.conv ? j % L.in_c : j;
                dst[j] = (src[j] - x_mean_[b][s]) / x_std_[b][s];
            }
        }
    }
}

void MultibranchNetwork::standardize_targets(const double* Y, std::size_t n, double* Ys) const {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < q_; ++t) {
            Ys[i * q_ + t] = (Y[i * q_ + t] - y_mean_[t]) / y_std_[t];
        }
    }
}

void MultibranchNetwork::initialize(n4m_rng_pcg64* rng) {
    std::fill(theta_.begin(), theta_.end(), 0.0);
    auto uniform = [this, rng](const Slice& s, double bound) {
        for (std::size_t k = 0; k < s.size; ++k) {
            const double u = n4m_pcg64_engine_next_double(rng);
            theta_[s.offset + k] = bound * (2.0 * u - 1.0);
        }
    };
    // He-uniform for layers followed by the leaky ReLU, Glorot-uniform for the
    // linear output layer; biases stay zero.
    for (const BranchLayout& L : branches_) {
        if (L.conv) {
            const double fan_in = static_cast<double>(L.in_c * L.k_h * L.k_w);
            uniform(L.conv_k, std::sqrt(6.0 / fan_in));
        }
        uniform(L.emb_w, std::sqrt(6.0 / static_cast<double>(L.embed_in)));
    }
    if (hid_w_.size > 0) uniform(hid_w_, std::sqrt(6.0 / static_cast<double>(z_size_)));
    uniform(out_w_, std::sqrt(6.0 / static_cast<double>(u_size_ + q_)));
}

MultibranchNetwork::Work MultibranchNetwork::make_work() const {
    Work w;
    w.conv_pre.resize(branches_.size());
    w.pooled.resize(branches_.size());
    w.emb_pre.resize(branches_.size());
    const std::size_t E = to_size(hyper_.embedding_dim);
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        const BranchLayout& L = branches_[b];
        if (L.conv) {
            w.conv_pre[b].assign(L.conv_size, 0.0);
            w.pooled[b].assign(L.embed_in, 0.0);
        }
        w.emb_pre[b].assign(E, 0.0);
    }
    w.z.assign(z_size_, 0.0);
    w.hid_pre.assign(to_size(hyper_.head_hidden), 0.0);
    w.hid.assign(to_size(hyper_.head_hidden), 0.0);
    w.out.assign(q_, 0.0);
    return w;
}

void MultibranchNetwork::forward(const double* xs, Work& w) const {
    const std::size_t F = to_size(hyper_.filters);
    const std::size_t E = to_size(hyper_.embedding_dim);
    const double* th = theta_.data();
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        const BranchLayout& L = branches_[b];
        const double* x = xs + L.offset;
        const double* h = x;
        if (L.conv) {
            const double* kern = th + L.conv_k.offset;
            const double* kbias = th + L.conv_b.offset;
            double* pre = w.conv_pre[b].data();
            for (std::size_t f = 0; f < F; ++f) {
                for (std::size_t i = 0; i < L.out_h; ++i) {
                    for (std::size_t j = 0; j < L.out_w; ++j) {
                        double s = kbias[f];
                        for (std::size_t c = 0; c < L.in_c; ++c) {
                            const double* kf = kern + ((f * L.in_c + c) * L.k_h) * L.k_w;
                            for (std::size_t a = 0; a < L.k_h; ++a) {
                                for (std::size_t d = 0; d < L.k_w; ++d) {
                                    s += kf[a * L.k_w + d] *
                                         x[((i + a) * L.in_w + (j + d)) * L.in_c + c];
                                }
                            }
                        }
                        pre[(f * L.out_h + i) * L.out_w + j] = s;
                    }
                }
            }
            // Average pooling of the activations (trailing positions that do
            // not fill a whole window are dropped).
            double* pooled = w.pooled[b].data();
            const double area = static_cast<double>(L.p_h * L.p_w);
            for (std::size_t f = 0; f < F; ++f) {
                for (std::size_t i = 0; i < L.pool_h; ++i) {
                    for (std::size_t j = 0; j < L.pool_w; ++j) {
                        double s = 0.0;
                        for (std::size_t a = 0; a < L.p_h; ++a) {
                            for (std::size_t d = 0; d < L.p_w; ++d) {
                                s += act(pre[(f * L.out_h + i * L.p_h + a) * L.out_w +
                                             j * L.p_w + d]);
                            }
                        }
                        pooled[(f * L.pool_h + i) * L.pool_w + j] = s / area;
                    }
                }
            }
            h = pooled;
        }
        const double* W = th + L.emb_w.offset;
        const double* bias = th + L.emb_b.offset;
        double* epre = w.emb_pre[b].data();
        for (std::size_t e = 0; e < E; ++e) {
            double s = bias[e];
            const double* We = W + e * L.embed_in;
            for (std::size_t k = 0; k < L.embed_in; ++k) s += We[k] * h[k];
            epre[e] = s;
            w.z[b * E + e] = act(s);
        }
    }
    const double* u = w.z.data();
    if (hid_w_.size > 0) {
        const std::size_t H = to_size(hyper_.head_hidden);
        const double* V = th + hid_w_.offset;
        const double* vb = th + hid_b_.offset;
        for (std::size_t k = 0; k < H; ++k) {
            double s = vb[k];
            const double* Vk = V + k * z_size_;
            for (std::size_t j = 0; j < z_size_; ++j) s += Vk[j] * w.z[j];
            w.hid_pre[k] = s;
            w.hid[k] = act(s);
        }
        u = w.hid.data();
    }
    const double* O = th + out_w_.offset;
    const double* ob = th + out_b_.offset;
    for (std::size_t t = 0; t < q_; ++t) {
        double s = ob[t];
        const double* Ot = O + t * u_size_;
        for (std::size_t j = 0; j < u_size_; ++j) s += Ot[j] * u[j];
        w.out[t] = s;
    }
}

// Accumulates into `grad` the gradient of sum_t g_out[t] * out[t] for one
// sample whose forward pass is `w`. `scratch` holds activation gradients.
void MultibranchNetwork::backward(const double* xs, const Work& w, const double* g_out,
                                  Work& scratch, std::vector<double>& grad) const {
    const std::size_t F = to_size(hyper_.filters);
    const std::size_t E = to_size(hyper_.embedding_dim);
    const double* th = theta_.data();
    double* g = grad.data();

    // Output layer.
    const double* u = hid_w_.size > 0 ? w.hid.data() : w.z.data();
    std::vector<double>& du = hid_w_.size > 0 ? scratch.hid : scratch.z;
    std::fill(du.begin(), du.end(), 0.0);
    {
        const double* O = th + out_w_.offset;
        double* gO = g + out_w_.offset;
        double* gob = g + out_b_.offset;
        for (std::size_t t = 0; t < q_; ++t) {
            const double gt = g_out[t];
            gob[t] += gt;
            for (std::size_t j = 0; j < u_size_; ++j) {
                gO[t * u_size_ + j] += gt * u[j];
                du[j] += O[t * u_size_ + j] * gt;
            }
        }
    }
    // Head hidden layer: du (= scratch.hid) -> dz (= scratch.z).
    std::vector<double>& dz = scratch.z;
    if (hid_w_.size > 0) {
        const std::size_t H = to_size(hyper_.head_hidden);
        std::fill(dz.begin(), dz.end(), 0.0);
        const double* V = th + hid_w_.offset;
        double* gV = g + hid_w_.offset;
        double* gvb = g + hid_b_.offset;
        for (std::size_t k = 0; k < H; ++k) {
            const double d = du[k] * act_grad(w.hid_pre[k]);
            gvb[k] += d;
            for (std::size_t j = 0; j < z_size_; ++j) {
                gV[k * z_size_ + j] += d * w.z[j];
                dz[j] += V[k * z_size_ + j] * d;
            }
        }
    }
    // Branches.
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        const BranchLayout& L = branches_[b];
        const double* x = xs + L.offset;
        const double* h = L.conv ? w.pooled[b].data() : x;
        const double* W = th + L.emb_w.offset;
        double* gW = g + L.emb_w.offset;
        double* geb = g + L.emb_b.offset;
        std::vector<double>& dh = scratch.pooled[b];
        if (L.conv) std::fill(dh.begin(), dh.end(), 0.0);
        for (std::size_t e = 0; e < E; ++e) {
            const double d = dz[b * E + e] * act_grad(w.emb_pre[b][e]);
            geb[e] += d;
            double* gWe = gW + e * L.embed_in;
            for (std::size_t k = 0; k < L.embed_in; ++k) gWe[k] += d * h[k];
            if (L.conv) {
                const double* We = W + e * L.embed_in;
                for (std::size_t k = 0; k < L.embed_in; ++k) dh[k] += We[k] * d;
            }
        }
        if (!L.conv) continue;
        // Unpool (non-overlapping average windows), through the activation,
        // into the convolution parameters.
        const double* pre = w.conv_pre[b].data();
        double* dpre = scratch.conv_pre[b].data();
        std::fill(scratch.conv_pre[b].begin(), scratch.conv_pre[b].end(), 0.0);
        const double area = static_cast<double>(L.p_h * L.p_w);
        for (std::size_t f = 0; f < F; ++f) {
            for (std::size_t i = 0; i < L.pool_h; ++i) {
                for (std::size_t j = 0; j < L.pool_w; ++j) {
                    const double gp = dh[(f * L.pool_h + i) * L.pool_w + j] / area;
                    for (std::size_t a = 0; a < L.p_h; ++a) {
                        for (std::size_t d = 0; d < L.p_w; ++d) {
                            const std::size_t at =
                                (f * L.out_h + i * L.p_h + a) * L.out_w + j * L.p_w + d;
                            dpre[at] = gp * act_grad(pre[at]);
                        }
                    }
                }
            }
        }
        double* gK = g + L.conv_k.offset;
        double* gkb = g + L.conv_b.offset;
        for (std::size_t f = 0; f < F; ++f) {
            for (std::size_t i = 0; i < L.out_h; ++i) {
                for (std::size_t j = 0; j < L.out_w; ++j) {
                    const double d = dpre[(f * L.out_h + i) * L.out_w + j];
                    if (d == 0.0) continue;
                    gkb[f] += d;
                    for (std::size_t c = 0; c < L.in_c; ++c) {
                        double* gkf = gK + ((f * L.in_c + c) * L.k_h) * L.k_w;
                        for (std::size_t a = 0; a < L.k_h; ++a) {
                            for (std::size_t e2 = 0; e2 < L.k_w; ++e2) {
                                gkf[a * L.k_w + e2] +=
                                    d * x[((i + a) * L.in_w + (j + e2)) * L.in_c + c];
                            }
                        }
                    }
                }
            }
        }
    }
}

double MultibranchNetwork::loss_gradient(const double* Xs, const double* Ys,
                                         const std::size_t* rows, std::size_t n_rows,
                                         std::vector<double>* grad) const {
    Work w = make_work();
    Work scratch;
    if (grad != nullptr) {
        grad->assign(theta_.size(), 0.0);
        scratch = make_work();
    }
    const double scale = 1.0 / (static_cast<double>(n_rows) * static_cast<double>(q_));
    std::vector<double> g_out(q_, 0.0);
    double loss = 0.0;
    for (std::size_t r = 0; r < n_rows; ++r) {
        const std::size_t i = rows[r];
        const double* xs = Xs + i * p_;
        const double* ys = Ys + i * q_;
        forward(xs, w);
        for (std::size_t t = 0; t < q_; ++t) {
            const double err = w.out[t] - ys[t];
            loss += err * err;
            g_out[t] = 2.0 * err * scale;
        }
        if (grad != nullptr) backward(xs, w, g_out.data(), scratch, *grad);
    }
    return loss * scale;
}

std::vector<double> MultibranchNetwork::fit(const double* X, const double* Y, std::size_t n,
                                            const Training& t) {
    if (n == 0) invalid("multibranch: no training rows");
    if (t.epochs < 0 || t.batch_size < 1 || !(t.learning_rate >= 0.0) ||
        !std::isfinite(t.learning_rate) || !(t.weight_decay >= 0.0) ||
        !std::isfinite(t.weight_decay)) {
        invalid("multibranch: invalid training settings");
    }
    if (n > std::numeric_limits<std::size_t>::max() / p_) invalid("multibranch: X is too large");
    fit_scaling(X, Y, n);
    std::vector<double> Xs(n * p_), Ys(n * q_);
    standardize(X, n, Xs.data());
    standardize_targets(Y, n, Ys.data());

    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, t.seed);
    initialize(&rng);

    const auto epochs = to_size(t.epochs);
    const std::size_t batch = to_size(t.batch_size) < n ? to_size(t.batch_size) : n;
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i) all[i] = i;
    std::vector<double> history;
    history.reserve(epochs + 1);
    history.push_back(loss_gradient(Xs.data(), Ys.data(), all.data(), n, nullptr));

    std::vector<double> grad, m(theta_.size(), 0.0), v(theta_.size(), 0.0);
    std::vector<unsigned char> decay(theta_.size(), 0);
    auto mark = [&decay](const Slice& s) {
        if (!s.decay) return;
        for (std::size_t k = 0; k < s.size; ++k) decay[s.offset + k] = 1;
    };
    for (const BranchLayout& L : branches_) {
        mark(L.conv_k);
        mark(L.emb_w);
    }
    mark(hid_w_);
    mark(out_w_);

    std::vector<std::int64_t> perm(n);
    std::vector<std::size_t> order(n);
    double b1t = 1.0, b2t = 1.0;
    const double lr = t.learning_rate;
    const double wd = t.weight_decay;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        n4m_pcg64_engine_permutation(&rng, perm.data(), static_cast<std::int64_t>(n));
        for (std::size_t i = 0; i < n; ++i) order[i] = to_size(perm[i]);
        for (std::size_t start = 0; start < n; start += batch) {
            const std::size_t nb = start + batch <= n ? batch : n - start;
            loss_gradient(Xs.data(), Ys.data(), order.data() + start, nb, &grad);
            b1t *= kAdamBeta1;
            b2t *= kAdamBeta2;
            const double c1 = 1.0 - b1t;
            const double c2 = 1.0 - b2t;
            for (std::size_t k = 0; k < theta_.size(); ++k) {
                const double gk = grad[k];
                m[k] = kAdamBeta1 * m[k] + (1.0 - kAdamBeta1) * gk;
                v[k] = kAdamBeta2 * v[k] + (1.0 - kAdamBeta2) * (gk * gk);
                const double mhat = m[k] / c1;
                const double vhat = v[k] / c2;
                double step = mhat / (std::sqrt(vhat) + kAdamEpsilon);
                if (decay[k] != 0) step += wd * theta_[k];
                theta_[k] -= lr * step;
            }
        }
        history.push_back(loss_gradient(Xs.data(), Ys.data(), all.data(), n, nullptr));
    }
    if (!all_finite(theta_)) {
        throw std::domain_error("multibranch: training diverged (non-finite weights); "
                                "reduce learning_rate");
    }
    return history;
}

void MultibranchNetwork::unstandardized_outputs(const Work& w, double* out) const {
    for (std::size_t t = 0; t < q_; ++t) out[t] = w.out[t] * y_std_[t] + y_mean_[t];
}

void MultibranchNetwork::predict(const double* X, std::size_t n, double* out) const {
    std::vector<double> xs(p_);
    Work w = make_work();
    for (std::size_t i = 0; i < n; ++i) {
        standardize(X + i * p_, 1, xs.data());
        forward(xs.data(), w);
        unstandardized_outputs(w, out + i * q_);
    }
}

void MultibranchNetwork::transform(const double* X, std::size_t n, double* out) const {
    std::vector<double> xs(p_);
    Work w = make_work();
    for (std::size_t i = 0; i < n; ++i) {
        standardize(X + i * p_, 1, xs.data());
        forward(xs.data(), w);
        std::memcpy(out + i * z_size_, w.z.data(), sizeof(double) * z_size_);
    }
}

void MultibranchNetwork::save(n4m_state_writer_t* w) const {
    n4m_state_write_i64(w, kStateVersion);
    n4m_state_write_i64(w, static_cast<std::int64_t>(p_));
    n4m_state_write_i64(w, static_cast<std::int64_t>(q_));
    n4m_state_write_i64(w, static_cast<std::int64_t>(branches_.size()));
    n4m_state_write_i64(w, hyper_.filters);
    n4m_state_write_i64(w, hyper_.kernel_size);
    n4m_state_write_i64(w, hyper_.pool_size);
    n4m_state_write_i64(w, hyper_.embedding_dim);
    n4m_state_write_i64(w, hyper_.head_hidden);
    for (const BranchLayout& L : branches_) {
        n4m_state_write_i64(w, L.shape.kind);
        n4m_state_write_i64(w, L.shape.d0);
        n4m_state_write_i64(w, L.shape.d1);
        n4m_state_write_i64(w, L.shape.d2);
    }
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        write_array(w, x_mean_[b]);
        write_array(w, x_std_[b]);
    }
    write_array(w, y_mean_);
    write_array(w, y_std_);
    for (const BranchLayout& L : branches_) {
        write_slice(w, theta_, L.conv_k);
        write_slice(w, theta_, L.conv_b);
        write_slice(w, theta_, L.emb_w);
        write_slice(w, theta_, L.emb_b);
    }
    write_slice(w, theta_, hid_w_);
    write_slice(w, theta_, hid_b_);
    write_slice(w, theta_, out_w_);
    write_slice(w, theta_, out_b_);
}

bool MultibranchNetwork::load(n4m_state_reader_t* r) {
    std::int64_t version = 0, p = 0, q = 0, nb = 0;
    Hyper h;
    if (!n4m_state_read_i64(r, &version) || version != kStateVersion ||
        !n4m_state_read_i64(r, &p) || !n4m_state_read_i64(r, &q) ||
        !n4m_state_read_i64(r, &nb) || !n4m_state_read_i64(r, &h.filters) ||
        !n4m_state_read_i64(r, &h.kernel_size) || !n4m_state_read_i64(r, &h.pool_size) ||
        !n4m_state_read_i64(r, &h.embedding_dim) || !n4m_state_read_i64(r, &h.head_hidden)) {
        return false;
    }
    // Each branch needs four i64 values: refuse a count the payload cannot hold
    // before allocating.
    if (nb <= 0 || static_cast<std::uint64_t>(nb) > r->remaining() / 32) return false;
    std::vector<BranchShape> shapes(to_size(nb));
    for (BranchShape& s : shapes) {
        if (!n4m_state_read_i64(r, &s.kind) || !n4m_state_read_i64(r, &s.d0) ||
            !n4m_state_read_i64(r, &s.d1) || !n4m_state_read_i64(r, &s.d2)) {
            return false;
        }
    }
    try {
        configure(p, q, shapes, h);
    } catch (const std::invalid_argument&) {
        return false;
    }
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        if (!read_array(r, x_mean_[b]) || !read_array(r, x_std_[b])) return false;
    }
    if (!read_array(r, y_mean_) || !read_array(r, y_std_)) return false;
    for (const BranchLayout& L : branches_) {
        if (!read_slice(r, theta_, L.conv_k) || !read_slice(r, theta_, L.conv_b) ||
            !read_slice(r, theta_, L.emb_w) || !read_slice(r, theta_, L.emb_b)) {
            return false;
        }
    }
    if (!read_slice(r, theta_, hid_w_) || !read_slice(r, theta_, hid_b_) ||
        !read_slice(r, theta_, out_w_) || !read_slice(r, theta_, out_b_)) {
        return false;
    }
    for (std::size_t b = 0; b < branches_.size(); ++b) {
        if (!all_finite(x_mean_[b]) || !all_finite(x_std_[b])) return false;
        for (double s : x_std_[b]) {
            if (!(s > 0.0)) return false;
        }
    }
    for (double s : y_std_) {
        if (!(s > 0.0)) return false;
    }
    return all_finite(y_mean_) && all_finite(y_std_) && all_finite(theta_);
}

}  // namespace n4m::core::neural
