// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Model-intrinsic importances (ABI 2.24) over the public ABI: affine maps of
// fitted estimators (also after N4ME import), the PLS latent structure
// against direct formulas, VIP / target-projection selectivity ratio /
// standardized coefficients against direct formulas, the raw-input affine
// map of affine role pipelines against their predictions, refusal of
// non-affine steps, an exact linearity sweep over every transformer that
// declares N4M_CAP_AFFINE_TRANSFORM, the GRADIENT explainer against linear
// SHAP, perturbed coalition rows, and profile aggregation / comparison
// (Spearman pinned to scipy.stats.spearmanr 1.15.3 on tied data).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "harness.hpp"
#include "n4m/n4m.h"

namespace {

using Values = std::vector<double>;

std::size_t sz(int64_t v) { return static_cast<std::size_t>(v); }

struct Ctx {
    n4m_context_t* p = nullptr;
    Ctx() { N4M_TEST_REQUIRE(n4m_context_create(&p) == N4M_OK); }
    ~Ctx() { n4m_context_destroy(p); }
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;
};

n4m_matrix_view_t view(const double* x, int64_t rows, int64_t cols) {
    return {const_cast<double*>(x), rows, cols, cols, 1, N4M_DTYPE_F64, 0};
}

double max_abs_diff(const Values& a, const Values& b) {
    N4M_TEST_REQUIRE(a.size() == b.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

double max_abs(const Values& a) {
    double m = 0.0;
    for (double v : a) m = std::max(m, std::fabs(v));
    return m;
}

std::string last_error(n4m_context_t* ctx) {
    const char* m = n4m_context_last_error(ctx);
    return m != nullptr ? m : "";
}

// Smooth synthetic spectra (rows x cols) with two targets.
struct Spectra {
    int64_t n, p;
    Values x, y;  // y: n x 2
    Spectra(int64_t rows, int64_t cols, double offset = 1.0) : n(rows), p(cols) {
        for (int64_t i = 0; i < n; ++i) {
            const double a = std::sin(0.37 * static_cast<double>(i) + 0.11);
            const double b = std::cos(0.23 * static_cast<double>(i * i % 17));
            double t0 = 0.0, t1 = 0.0;
            for (int64_t j = 0; j < p; ++j) {
                const double u = static_cast<double>(j) / static_cast<double>(p);
                const double v = offset + a * std::exp(-8.0 * (u - 0.3) * (u - 0.3)) +
                                 b * std::exp(-10.0 * (u - 0.7) * (u - 0.7)) +
                                 0.05 * std::sin(static_cast<double>(i) * 1.7 +
                                                 static_cast<double>(j));
                x.push_back(v);
                t0 += (j % 3 == 0 ? 1.5 : -0.4) * v;
                t1 += std::cos(static_cast<double>(j)) * v;
            }
            y.push_back(t0 + 0.01 * std::sin(13.0 * static_cast<double>(i)));
            y.push_back(t1);
        }
    }
};

n4m_params_t* params_for(n4m_context_t* ctx, const char* id) {
    int32_t index = -1;
    N4M_TEST_REQUIRE(n4m_method_find(id, &index) == N4M_OK);
    n4m_params_t* p = nullptr;
    N4M_TEST_REQUIRE(n4m_params_create(ctx, index, &p) == N4M_OK);
    return p;
}

n4m_estimator_t* fit_estimator(n4m_context_t* ctx, const char* id, const n4m_params_t* params,
                               const Spectra& d, int64_t q) {
    n4m_estimator_t* est = nullptr;
    N4M_TEST_REQUIRE(n4m_estimator_create(ctx, id, params, &est) == N4M_OK);
    Values y(sz(d.n * q));
    for (int64_t i = 0; i < d.n; ++i)
        for (int64_t t = 0; t < q; ++t) y[sz(i * q + t)] = d.y[sz(i * 2 + t)];
    const auto X = view(d.x.data(), d.n, d.p);
    const auto Y = view(y.data(), d.n, q);
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &X;
    in.Y = &Y;
    N4M_TEST_REQUIRE(n4m_estimator_fit(ctx, est, &in) == N4M_OK);
    return est;
}

n4m_estimator_t* roundtrip(n4m_context_t* ctx, const n4m_estimator_t* est) {
    size_t size = 0;
    N4M_TEST_REQUIRE(n4m_estimator_export_size(ctx, est, 0, &size) == N4M_OK);
    std::vector<unsigned char> bytes(size);
    size_t written = 0;
    N4M_TEST_REQUIRE(n4m_estimator_export_to_buffer(ctx, est, 0, bytes.data(), size, &written) ==
                     N4M_OK);
    n4m_estimator_t* out = nullptr;
    N4M_TEST_REQUIRE(n4m_estimator_import_from_buffer(ctx, bytes.data(), written, &out) == N4M_OK);
    return out;
}

Values predict(n4m_context_t* ctx, const n4m_estimator_t* est, const Values& x, int64_t n,
               int64_t p) {
    int64_t q = 0;
    N4M_TEST_REQUIRE(n4m_estimator_n_outputs(est, &q) == N4M_OK);
    Values out(sz(n * q));
    const auto X = view(x.data(), n, p);
    auto O = view(out.data(), n, q);
    N4M_TEST_REQUIRE(n4m_estimator_predict(ctx, est, &X, &O) == N4M_OK);
    return out;
}

struct Affine {
    Values B, b;
};

Affine affine_of(n4m_context_t* ctx, const n4m_estimator_t* est, int64_t p, int64_t q) {
    Affine a{Values(sz(p * q)), Values(sz(q))};
    N4M_TEST_REQUIRE(n4m_estimator_affine_coefficients_v1(ctx, est, a.B.data(), p * q,
                                                          a.b.data(), q) == N4M_OK);
    return a;
}

Values apply_affine(const Affine& a, const Values& x, int64_t n, int64_t p, int64_t q) {
    Values out(sz(n * q));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t t = 0; t < q; ++t) {
            double s = a.b[sz(t)];
            for (int64_t j = 0; j < p; ++j) s += x[sz(i * p + j)] * a.B[sz(j * q + t)];
            out[sz(i * q + t)] = s;
        }
    return out;
}

// ---------------------------------------------------------------------------

void affine_coefficients_pls_and_ridge() {
    Ctx ctx;
    const Spectra d(36, 14);
    const Spectra probe(9, 14, 1.3);
    for (const char* id : {"models.pls.pls_regression", "models.regularized.ridge"}) {
        n4m_params_t* params = params_for(ctx.p, id);
        if (std::string(id) == "models.pls.pls_regression") {
            N4M_TEST_REQUIRE(n4m_params_set_int(params, "n_components", 4) == N4M_OK);
        }
        n4m_estimator_t* est = fit_estimator(ctx.p, id, params, d, 2);
        n4m_params_destroy(params);
        n4m_estimator_t* imported = roundtrip(ctx.p, est);
        for (const n4m_estimator_t* e : {static_cast<const n4m_estimator_t*>(est),
                                         static_cast<const n4m_estimator_t*>(imported)}) {
            const Affine a = affine_of(ctx.p, e, 14, 2);
            const Values pred = predict(ctx.p, e, probe.x, 9, 14);
            N4M_TEST_REQUIRE(max_abs_diff(apply_affine(a, probe.x, 9, 14, 2), pred) <=
                             1e-10 * (1.0 + max_abs(pred)));
        }
        // Exactly the same map before and after N4ME import.
        const Affine a0 = affine_of(ctx.p, est, 14, 2), a1 = affine_of(ctx.p, imported, 14, 2);
        N4M_TEST_REQUIRE(a0.B == a1.B && a0.b == a1.b);
        // Too small buffers are refused.
        N4M_TEST_REQUIRE(n4m_estimator_affine_coefficients_v1(ctx.p, est, Values(27).data(), 27,
                                                              Values(2).data(), 2) ==
                         N4M_ERR_SHAPE_MISMATCH);
        n4m_estimator_destroy(imported);
        n4m_estimator_destroy(est);
    }
    // A non-affine predictor is refused, typed.
    n4m_estimator_t* snv = nullptr;
    N4M_TEST_REQUIRE(n4m_estimator_create(ctx.p, "preprocessing.scatter.snv", nullptr, &snv) ==
                     N4M_OK);
    const auto X = view(d.x.data(), d.n, d.p);
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &X;
    N4M_TEST_REQUIRE(n4m_estimator_fit(ctx.p, snv, &in) == N4M_OK);
    Values B(14), b(1);
    N4M_TEST_REQUIRE(n4m_estimator_affine_coefficients_v1(ctx.p, snv, B.data(), 14, b.data(),
                                                          1) == N4M_ERR_UNSUPPORTED);
    n4m_estimator_destroy(snv);
}

struct Latent {
    int64_t a = 0;
    Values W, R, P, Q, mean, scale;
};

Latent latent_of(n4m_context_t* ctx, const n4m_estimator_t* est, int64_t p, int64_t q) {
    Latent L;
    N4M_TEST_REQUIRE(n4m_estimator_latent_structure_v1(ctx, est, &L.a, nullptr, nullptr, nullptr,
                                                       0, nullptr, 0, nullptr, nullptr, 0) ==
                     N4M_OK);
    L.W.resize(sz(p * L.a));
    L.R.resize(sz(p * L.a));
    L.P.resize(sz(p * L.a));
    L.Q.resize(sz(q * L.a));
    L.mean.resize(sz(p));
    L.scale.resize(sz(p));
    int64_t a = 0;
    N4M_TEST_REQUIRE(n4m_estimator_latent_structure_v1(
                         ctx, est, &a, L.W.data(), L.R.data(), L.P.data(), p * L.a, L.Q.data(),
                         q * L.a, L.mean.data(), L.scale.data(), p) == N4M_OK);
    N4M_TEST_REQUIRE(a == L.a);
    return L;
}

void latent_structure_and_importances() {
    Ctx ctx;
    const int64_t n = 40, p = 12, q = 2, A = 3;
    const Spectra d(n, p);
    n4m_params_t* params = params_for(ctx.p, "models.pls.pls_regression");
    N4M_TEST_REQUIRE(n4m_params_set_int(params, "n_components", A) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_params_set_bool(params, "scale_x", 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_params_set_bool(params, "scale_y", 0) == N4M_OK);
    n4m_estimator_t* fitted = fit_estimator(ctx.p, "models.pls.pls_regression", params, d, q);
    n4m_params_destroy(params);
    n4m_estimator_t* est = roundtrip(ctx.p, fitted);  // the browser always re-imports
    const Latent L = latent_of(ctx.p, est, p, q);
    N4M_TEST_REQUIRE(L.a == A);
    for (double s : L.scale) N4M_TEST_REQUIRE(s == 1.0);

    // Centred, unscaled X and Y: B = R Q', and T = (X - mean) R is the
    // latent transform of the estimator.
    const Affine aff = affine_of(ctx.p, est, p, q);
    Values B(sz(p * q), 0.0);
    for (int64_t j = 0; j < p; ++j)
        for (int64_t t = 0; t < q; ++t)
            for (int64_t c = 0; c < A; ++c)
                B[sz(j * q + t)] += L.R[sz(j * A + c)] * L.Q[sz(t * A + c)];
    N4M_TEST_REQUIRE(max_abs_diff(B, aff.B) <= 1e-9 * (1.0 + max_abs(aff.B)));
    Values T(sz(n * A), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < p; ++j)
            for (int64_t c = 0; c < A; ++c)
                T[sz(i * A + c)] += (d.x[sz(i * p + j)] - L.mean[sz(j)]) * L.R[sz(j * A + c)];
    Values Tlib(sz(n * A));
    {
        const auto X = view(d.x.data(), n, p);
        auto O = view(Tlib.data(), n, A);
        N4M_TEST_REQUIRE(n4m_estimator_transform(ctx.p, est, &X, &O) == N4M_OK);
    }
    N4M_TEST_REQUIRE(max_abs_diff(T, Tlib) <= 1e-10 * (1.0 + max_abs(T)));

    const auto X = view(d.x.data(), n, p);
    // VIP: sqrt(p sum_a SSY_a (w_ja / ||w_a||)^2 / sum_a SSY_a), SSY_a = ||t_a||^2 ||q_a||^2.
    Values vip_oracle(sz(p), 0.0);
    {
        Values ssy(sz(A), 0.0), wn(sz(A), 0.0);
        double total = 0.0;
        for (int64_t c = 0; c < A; ++c) {
            double tt = 0.0, qq = 0.0, ww = 0.0;
            for (int64_t i = 0; i < n; ++i) tt += T[sz(i * A + c)] * T[sz(i * A + c)];
            for (int64_t t = 0; t < q; ++t) qq += L.Q[sz(t * A + c)] * L.Q[sz(t * A + c)];
            for (int64_t j = 0; j < p; ++j) ww += L.W[sz(j * A + c)] * L.W[sz(j * A + c)];
            ssy[sz(c)] = tt * qq;
            wn[sz(c)] = std::sqrt(ww);
            total += ssy[sz(c)];
        }
        for (int64_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (int64_t c = 0; c < A; ++c) {
                const double w = L.W[sz(j * A + c)] / wn[sz(c)];
                s += ssy[sz(c)] * w * w;
            }
            vip_oracle[sz(j)] = std::sqrt(static_cast<double>(p) * s / total);
        }
    }
    Values vip(sz(p));
    N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, est, &X, N4M_IMPORTANCE_VIP, 0,
                                                 vip.data(), p) == N4M_OK);
    N4M_TEST_REQUIRE(max_abs_diff(vip, vip_oracle) <= 1e-10);
    double vip_ss = 0.0;
    for (double v : vip) vip_ss += v * v;
    N4M_TEST_REQUIRE(std::fabs(vip_ss / static_cast<double>(p) - 1.0) <= 1e-10);

    // Target-projection selectivity ratio of output 1 (Rajalahti et al. 2009).
    for (int64_t o = 0; o < q; ++o) {
        Values mean(sz(p), 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < p; ++j) mean[sz(j)] += d.x[sz(i * p + j)] / n;
        Values w(sz(p));
        double norm = 0.0;
        for (int64_t j = 0; j < p; ++j) {
            w[sz(j)] = aff.B[sz(j * q + o)];
            norm += w[sz(j)] * w[sz(j)];
        }
        norm = std::sqrt(norm);
        Values t(sz(n), 0.0);
        double tt = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < p; ++j)
                t[sz(i)] += (d.x[sz(i * p + j)] - mean[sz(j)]) * w[sz(j)] / norm;
            tt += t[sz(i)] * t[sz(i)];
        }
        Values sr_oracle(sz(p));
        for (int64_t j = 0; j < p; ++j) {
            double zt = 0.0;
            for (int64_t i = 0; i < n; ++i) zt += (d.x[sz(i * p + j)] - mean[sz(j)]) * t[sz(i)];
            const double pj = zt / tt;
            double expl = 0.0, res = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double fit = t[sz(i)] * pj;
                const double r = d.x[sz(i * p + j)] - mean[sz(j)] - fit;
                expl += fit * fit;
                res += r * r;
            }
            sr_oracle[sz(j)] = expl / res;
        }
        Values sr(sz(p));
        N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, est, &X,
                                                     N4M_IMPORTANCE_SELECTIVITY_RATIO, o,
                                                     sr.data(), p) == N4M_OK);
        for (int64_t j = 0; j < p; ++j)
            N4M_TEST_REQUIRE(std::fabs(sr[sz(j)] - sr_oracle[sz(j)]) <=
                             1e-9 * (1.0 + sr_oracle[sz(j)]));
        // Standardized coefficients: B[j][o] * sd_j (ddof = 1).
        Values sc(sz(p));
        N4M_TEST_REQUIRE(n4m_estimator_importance_v1(
                             ctx.p, est, &X, N4M_IMPORTANCE_STANDARDIZED_COEFFICIENTS, o,
                             sc.data(), p) == N4M_OK);
        for (int64_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double dv = d.x[sz(i * p + j)] - mean[sz(j)];
                ss += dv * dv;
            }
            const double expect = aff.B[sz(j * q + o)] * std::sqrt(ss / static_cast<double>(n - 1));
            N4M_TEST_REQUIRE(std::fabs(sc[sz(j)] - expect) <= 1e-12 * (1.0 + std::fabs(expect)));
        }
    }
    // Refusals: output index, one row for standardized coefficients, kind.
    Values out(sz(p));
    N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, est, &X, N4M_IMPORTANCE_SELECTIVITY_RATIO,
                                                 2, out.data(), p) == N4M_ERR_INVALID_ARGUMENT);
    const auto one = view(d.x.data(), 1, p);
    N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, est, &one,
                                                 N4M_IMPORTANCE_STANDARDIZED_COEFFICIENTS, 0,
                                                 out.data(), p) == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, est, &X, 7, 0, out.data(), p) ==
                     N4M_ERR_INVALID_ARGUMENT);
    n4m_estimator_destroy(est);
    n4m_estimator_destroy(fitted);

    // Ridge: affine (SR, standardized coefficients) but no latent structure.
    n4m_estimator_t* ridge = fit_estimator(ctx.p, "models.regularized.ridge", nullptr, d, q);
    int64_t a = -1;
    N4M_TEST_REQUIRE(n4m_estimator_latent_structure_v1(ctx.p, ridge, &a, nullptr, nullptr,
                                                       nullptr, 0, nullptr, 0, nullptr, nullptr,
                                                       0) == N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(a == 0);
    N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, ridge, &X, N4M_IMPORTANCE_VIP, 0,
                                                 out.data(), p) == N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(n4m_estimator_importance_v1(ctx.p, ridge, &X,
                                                 N4M_IMPORTANCE_SELECTIVITY_RATIO, 0, out.data(),
                                                 p) == N4M_OK);
    for (double v : out) N4M_TEST_REQUIRE(v >= 0.0);
    n4m_estimator_destroy(ridge);
}

// ---------------------------------------------------------------------------
// Role pipelines.

struct StepSpec {
    const char* id;
    std::vector<std::pair<const char*, int64_t>> ints;
};

n4m_role_pipeline_t* fit_pipeline(n4m_context_t* ctx, const std::vector<StepSpec>& steps,
                                  const Spectra& d, int64_t q) {
    std::vector<const char*> ids;
    std::vector<n4m_params_t*> owned;
    std::vector<const n4m_params_t*> params;
    for (const StepSpec& s : steps) {
        ids.push_back(s.id);
        n4m_params_t* p = params_for(ctx, s.id);
        for (const auto& kv : s.ints) {
            N4M_TEST_REQUIRE(n4m_params_set_int(p, kv.first, kv.second) == N4M_OK);
        }
        owned.push_back(p);
        params.push_back(p);
    }
    n4m_role_pipeline_t* pipe = nullptr;
    N4M_TEST_REQUIRE(n4m_role_pipeline_create(ctx, static_cast<int32_t>(ids.size()), ids.data(),
                                              params.data(), &pipe) == N4M_OK);
    for (n4m_params_t* p : owned) n4m_params_destroy(p);
    Values y(sz(d.n * q));
    for (int64_t i = 0; i < d.n; ++i)
        for (int64_t t = 0; t < q; ++t) y[sz(i * q + t)] = d.y[sz(i * 2 + t)];
    const auto X = view(d.x.data(), d.n, d.p);
    const auto Y = view(y.data(), d.n, q);
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &X;
    in.Y = &Y;
    N4M_TEST_REQUIRE(n4m_role_pipeline_fit(ctx, pipe, &in) == N4M_OK);
    return pipe;
}

n4m_role_pipeline_t* reimport(n4m_context_t* ctx, const std::vector<StepSpec>& steps,
                              const n4m_role_pipeline_t* fitted) {
    std::vector<const char*> ids;
    std::vector<n4m_params_t*> owned;
    std::vector<const n4m_params_t*> params;
    for (const StepSpec& s : steps) {
        ids.push_back(s.id);
        n4m_params_t* p = params_for(ctx, s.id);
        for (const auto& kv : s.ints) {
            N4M_TEST_REQUIRE(n4m_params_set_int(p, kv.first, kv.second) == N4M_OK);
        }
        owned.push_back(p);
        params.push_back(p);
    }
    n4m_role_pipeline_t* pipe = nullptr;
    N4M_TEST_REQUIRE(n4m_role_pipeline_create(ctx, static_cast<int32_t>(ids.size()), ids.data(),
                                              params.data(), &pipe) == N4M_OK);
    for (n4m_params_t* p : owned) n4m_params_destroy(p);
    int32_t n_states = 0;
    N4M_TEST_REQUIRE(n4m_role_pipeline_n_states(fitted, &n_states) == N4M_OK);
    std::vector<std::vector<unsigned char>> bytes(sz(n_states));
    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    for (int32_t k = 0; k < n_states; ++k) {
        size_t size = 0, written = 0;
        N4M_TEST_REQUIRE(n4m_role_pipeline_export_state_size(ctx, fitted, k, 0, &size) == N4M_OK);
        bytes[sz(k)].resize(size);
        N4M_TEST_REQUIRE(n4m_role_pipeline_export_state_to_buffer(ctx, fitted, k, 0,
                                                                  bytes[sz(k)].data(), size,
                                                                  &written) == N4M_OK);
        ptrs.push_back(bytes[sz(k)].data());
        sizes.push_back(written);
    }
    N4M_TEST_REQUIRE(n4m_role_pipeline_import_states(ctx, pipe, n_states, ptrs.data(),
                                                     sizes.data()) == N4M_OK);
    return pipe;
}

Values pipeline_predict(n4m_context_t* ctx, const n4m_role_pipeline_t* pipe, const Values& x,
                        int64_t n, int64_t p, int64_t q) {
    Values out(sz(n * q));
    const auto X = view(x.data(), n, p);
    auto O = view(out.data(), n, q);
    N4M_TEST_REQUIRE(n4m_role_pipeline_predict(ctx, pipe, &X, &O) == N4M_OK);
    return out;
}

void pipeline_input_affine_exact() {
    Ctx ctx;
    const int64_t p = 30, q = 2;
    const Spectra d(40, p);
    const Spectra probe(11, p, 0.7);
    // Note: a mirror-mode Savitzky-Golay derivative has an exactly zero first
    // column (computed as +-1e-17); autoscaling such a column amplifies the
    // rounding, so the derivative recipes here do not autoscale it.
    const std::vector<std::vector<StepSpec>> recipes = {
        {{"preprocessing.derivatives.savitzky_golay",
          {{"window_length", 7}, {"polyorder", 2}, {"deriv", 0}}},
         {"preprocessing.scaling.standard_scale", {}},
         {"models.pls.pls_regression", {{"n_components", 4}}}},
        {{"preprocessing.derivatives.first_derivative", {}},
         {"preprocessing.smoothing.gaussian", {}},
         {"preprocessing.scaling.baseline", {}},
         {"models.pls.pls_regression", {{"n_components", 3}}}},
        {{"preprocessing.resampling.crop", {{"start", 3}, {"end", 27}}},
         {"preprocessing.baselines.detrend", {}},
         {"models.regularized.ridge", {}}},
        {{"models.regularized.ridge", {}}},
    };
    for (const auto& recipe : recipes) {
        n4m_role_pipeline_t* fitted = fit_pipeline(ctx.p, recipe, d, q);
        n4m_role_pipeline_t* pipe = reimport(ctx.p, recipe, fitted);
        Affine raw{Values(sz(p * q)), Values(sz(q))};
        N4M_TEST_REQUIRE(n4m_role_pipeline_input_affine_coefficients_v1(
                             ctx.p, pipe, raw.B.data(), p * q, raw.b.data(), q) == N4M_OK);
        const Values pred = pipeline_predict(ctx.p, pipe, probe.x, probe.n, p, q);
        const Values lin = apply_affine(raw, probe.x, probe.n, p, q);
        N4M_TEST_REQUIRE(max_abs_diff(lin, pred) <= 1e-10 * (1.0 + max_abs(pred)));
        // Every step is affine; column alignment follows the transformer traits.
        int32_t n_steps = 0;
        N4M_TEST_REQUIRE(n4m_role_pipeline_n_steps(pipe, &n_steps) == N4M_OK);
        for (int32_t s = 0; s < n_steps; ++s) {
            n4m_role_pipeline_step_info_v1_t info{};
            info.struct_size = sizeof(info);
            N4M_TEST_REQUIRE(n4m_role_pipeline_step_info_v1(pipe, s, &info) == N4M_OK);
            N4M_TEST_REQUIRE(info.affine == 1);
            const std::string id = info.method_id;
            const bool aligned = id == "preprocessing.derivatives.savitzky_golay" ||
                                 id == "preprocessing.derivatives.first_derivative" ||
                                 id == "preprocessing.smoothing.gaussian" ||
                                 id == "preprocessing.scaling.standard_scale" ||
                                 id == "preprocessing.baselines.detrend" ||
                                 id == "preprocessing.scaling.baseline";
            N4M_TEST_REQUIRE(info.column_aligned == (aligned ? 1 : 0));
        }
        n4m_role_pipeline_destroy(pipe);
        n4m_role_pipeline_destroy(fitted);
    }
    // A 2.14 caller (shorter descriptor) still reads the prefix.
    {
        n4m_role_pipeline_t* pipe = fit_pipeline(ctx.p, recipes[3], d, q);
        n4m_role_pipeline_step_info_v1_t info{};
        info.struct_size =
            static_cast<uint32_t>(offsetof(n4m_role_pipeline_step_info_v1_t, affine));
        info.affine = 7;
        N4M_TEST_REQUIRE(n4m_role_pipeline_step_info_v1(pipe, 0, &info) == N4M_OK);
        N4M_TEST_REQUIRE(info.affine == 7 && info.n_features_in == p);
        n4m_role_pipeline_destroy(pipe);
    }
    // SNV is not affine: typed refusal naming the step.
    const std::vector<StepSpec> snv = {{"preprocessing.derivatives.savitzky_golay", {}},
                                       {"preprocessing.scatter.snv", {}},
                                       {"models.pls.pls_regression", {{"n_components", 2}}}};
    n4m_role_pipeline_t* pipe = fit_pipeline(ctx.p, snv, d, q);
    Affine raw{Values(sz(p * q)), Values(sz(q))};
    N4M_TEST_REQUIRE(n4m_role_pipeline_input_affine_coefficients_v1(
                         ctx.p, pipe, raw.B.data(), p * q, raw.b.data(), q) ==
                     N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(last_error(ctx.p).find("step 1 (preprocessing.scatter.snv)") !=
                     std::string::npos);
    n4m_role_pipeline_step_info_v1_t info{};
    info.struct_size = sizeof(info);
    N4M_TEST_REQUIRE(n4m_role_pipeline_step_info_v1(pipe, 1, &info) == N4M_OK);
    N4M_TEST_REQUIRE(info.affine == 0 && info.column_aligned == 1);
    n4m_role_pipeline_destroy(pipe);
}

// Every transformer declaring N4M_CAP_AFFINE_TRANSFORM satisfies
// T(a x + (1 - a) z) = a T(x) + (1 - a) T(z) on fitted states (default
// parameters; methods whose defaults cannot fit this data are skipped).
void affine_transform_traits_are_exact() {
    Ctx ctx;
    const int64_t n = 32, p = 24;
    const Spectra d(n, p);
    const Spectra other(n, p, 0.4);
    Values axis(sz(p)), xt(d.x);
    for (int64_t j = 0; j < p; ++j) axis[sz(j)] = 1100.0 + 10.0 * static_cast<double>(j);
    for (double& v : xt) v *= 1.05;
    Values y(sz(n)), w(sz(n), 1.0);
    std::vector<int64_t> groups(sz(n)), folds(sz(n)), labels(sz(n));
    for (int64_t i = 0; i < n; ++i) {
        y[sz(i)] = d.y[sz(2 * i)];
        groups[sz(i)] = i % 5;
        folds[sz(i)] = i % 4;
        labels[sz(i)] = i % 2;
    }
    std::vector<int64_t> fgroups(sz(p)), blocks = {p / 2, p - p / 2};
    for (int64_t j = 0; j < p; ++j) fgroups[sz(j)] = j / 6;
    int32_t count = 0;
    N4M_TEST_REQUIRE(n4m_method_count(&count) == N4M_OK);
    int checked = 0, skipped = 0;
    for (int32_t m = 0; m < count; ++m) {
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        N4M_TEST_REQUIRE(n4m_method_info_v1(m, &info) == N4M_OK);
        if ((info.capabilities & N4M_CAP_AFFINE_TRANSFORM) == 0) continue;
        N4M_TEST_REQUIRE((info.capabilities & N4M_CAP_TRANSFORM) != 0);
        const auto X = view(d.x.data(), n, p);
        const auto Y = view(y.data(), n, 1);
        const auto XT = view(xt.data(), n, p);
        n4m_fit_inputs_v1_t in{};
        in.struct_size = sizeof(in);
        in.X = &X;
        const auto req = [&](n4m_fit_input_t k) { return info.inputs[k] == N4M_INPUT_REQUIRED; };
        if (req(N4M_FIT_INPUT_Y)) in.Y = &Y;
        if (req(N4M_FIT_INPUT_LABELS)) { in.labels = labels.data(); in.n_labels = n; }
        if (req(N4M_FIT_INPUT_SAMPLE_WEIGHT)) {
            in.sample_weight = w.data();
            in.n_sample_weight = n;
        }
        if (req(N4M_FIT_INPUT_GROUPS)) { in.groups = groups.data(); in.n_groups = n; }
        if (req(N4M_FIT_INPUT_FEATURE_GROUPS)) {
            in.feature_groups = fgroups.data();
            in.n_feature_groups = p;
        }
        if (req(N4M_FIT_INPUT_BLOCKS)) { in.block_sizes = blocks.data(); in.n_blocks = 2; }
        if (info.inputs[N4M_FIT_INPUT_AXIS] != N4M_INPUT_NONE) {
            in.axis = axis.data();
            in.n_axis = p;
        }
        if (req(N4M_FIT_INPUT_TARGET_DOMAIN)) in.X_target = &XT;
        if (req(N4M_FIT_INPUT_FOLD_IDS)) { in.fold_ids = folds.data(); in.n_fold_ids = n; }
        n4m_estimator_t* est = nullptr;
        N4M_TEST_REQUIRE(n4m_estimator_create(ctx.p, info.method_id, nullptr, &est) == N4M_OK ||
                         true);
        if (est == nullptr || n4m_estimator_fit(ctx.p, est, &in) != N4M_OK) {
            std::printf("\n    skipped (default fit failed): %s", info.method_id);
            ++skipped;
            n4m_estimator_destroy(est);
            continue;
        }
        int64_t cols = 0;
        N4M_TEST_REQUIRE(n4m_estimator_transform_cols(est, &cols) == N4M_OK);
        const double alpha = 0.3;
        Values mix(sz(n * p));
        for (std::size_t k = 0; k < mix.size(); ++k)
            mix[k] = alpha * d.x[k] + (1.0 - alpha) * other.x[k];
        const auto run = [&](const Values& rows) {
            Values out(sz(n * cols));
            const auto V = view(rows.data(), n, p);
            auto O = view(out.data(), n, cols);
            N4M_TEST_REQUIRE(n4m_estimator_transform(ctx.p, est, &V, &O) == N4M_OK);
            return out;
        };
        const Values tx = run(d.x), tz = run(other.x), tm = run(mix);
        Values combo(tx.size());
        for (std::size_t k = 0; k < combo.size(); ++k)
            combo[k] = alpha * tx[k] + (1.0 - alpha) * tz[k];
        const double err = max_abs_diff(tm, combo);
        if (!(err <= 1e-9 * (1.0 + max_abs(tm)))) {
            std::printf("\n    NOT affine: %s (error %g)", info.method_id, err);
            N4M_TEST_REQUIRE(false);
        }
        ++checked;
        n4m_estimator_destroy(est);
    }
    std::printf("\n    %d affine transformers checked, %d skipped ... ", checked, skipped);
    N4M_TEST_REQUIRE(checked >= 20);
}

// ---------------------------------------------------------------------------
// GRADIENT explainer and coalition rows.

struct Session {
    Values attr, expected, pred, grads, mean_grad, steps;
    int64_t total = 0;
};

template <typename Model>
Session run_gradient(n4m_context_t* ctx, const Values& X, int64_t R, int64_t F, const Values& bg,
                     int64_t B, const std::vector<int64_t>& groups, int64_t G, int64_t O,
                     int32_t baseline, Model model, double step = 0.0) {
    n4m_explain_options_v1_t opt;
    N4M_TEST_REQUIRE(n4m_explain_options_init_v1(&opt) == N4M_OK);
    opt.method = N4M_EXPLAIN_GRADIENT;
    opt.baseline = baseline;
    opt.n_outputs = O;
    opt.max_batch_rows = 13;
    const auto xv = view(X.data(), R, F);
    const auto bv = view(bg.data(), B, F);
    n4m_explain_session_t* s = nullptr;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx, &xv, &bv,
                                                   groups.empty() ? nullptr : groups.data(), G,
                                                   &opt, &s) == N4M_OK);
    if (step > 0.0) {
        N4M_TEST_REQUIRE(n4m_explain_session_set_gradient_step_v1(ctx, s, step) == N4M_OK);
    }
    Session out;
    N4M_TEST_REQUIRE(n4m_explain_session_info_v1(s, nullptr, nullptr, nullptr, nullptr,
                                                 &out.total) == N4M_OK);
    N4M_TEST_REQUIRE(out.total == R * (2 * F + 1) + 1);
    Values rows(sz(13 * F)), preds(sz(13 * O));
    int64_t evaluated = 0;
    for (;;) {
        int64_t written = 0;
        N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx, s, rows.data(), 13, &written) ==
                         N4M_OK);
        if (written == 0) break;
        if (evaluated == 0) {  // the step is frozen once the plan started
            N4M_TEST_REQUIRE(n4m_explain_session_set_gradient_step_v1(ctx, s, 1e-2) ==
                             N4M_ERR_INVALID_ARGUMENT);
        }
        model(rows.data(), written, preds.data());
        evaluated += written;
        N4M_TEST_REQUIRE(n4m_explain_session_submit_v1(ctx, s, preds.data(), written) == N4M_OK);
    }
    N4M_TEST_REQUIRE(evaluated == out.total);
    out.attr.resize(sz(R * G * O));
    out.expected.resize(sz(R * O));
    out.pred.resize(sz(R * O));
    N4M_TEST_REQUIRE(n4m_explain_session_result_v1(
                         ctx, s, out.attr.data(), R * G * O, out.expected.data(), R * O,
                         out.pred.data(), R * O, nullptr, 0) == N4M_OK);
    out.grads.resize(sz(R * F * O));
    out.mean_grad.resize(sz(F * O));
    out.steps.resize(sz(F));
    N4M_TEST_REQUIRE(n4m_explain_session_gradients_v1(ctx, s, out.grads.data(), R * F * O,
                                                      out.mean_grad.data(), F * O,
                                                      out.steps.data(), F) == N4M_OK);
    n4m_explain_session_destroy(s);
    return out;
}

void gradient_equals_linear_shap_on_affine_models() {
    Ctx ctx;
    const int64_t R = 5, F = 9, B = 7, O = 2, G = 3;
    Values X, bg, coef(sz(F * O)), icpt = {0.7, -1.9};
    for (int64_t i = 0; i < R * F; ++i) X.push_back(std::sin(0.7 * static_cast<double>(i)) + 2.0);
    for (int64_t i = 0; i < B * F; ++i) {
        bg.push_back(std::cos(0.3 * static_cast<double>(i)) * 0.4 + 2.0);
    }
    for (int64_t k = 0; k < F * O; ++k) coef[sz(k)] = std::sin(1.3 * static_cast<double>(k)) * 3.0;
    const std::vector<int64_t> groups = {0, 0, 0, 1, 1, 2, 2, 2, 2};
    const auto affine = [&](const double* rows, int64_t n, double* out) {
        for (int64_t i = 0; i < n; ++i)
            for (int64_t o = 0; o < O; ++o) {
                double s = icpt[sz(o)];
                for (int64_t j = 0; j < F; ++j) s += rows[i * F + j] * coef[sz(j * O + o)];
                out[i * O + o] = s;
            }
    };
    for (int32_t baseline : {N4M_EXPLAIN_BASELINE_BACKGROUND, N4M_EXPLAIN_BASELINE_MEAN}) {
        const Session g = run_gradient(ctx.p, X, R, F, bg, B, groups, G, O, baseline, affine);
        // Linear SHAP with the background mean reference.
        Values ref(sz(F), 0.0);
        for (int64_t b = 0; b < B; ++b)
            for (int64_t j = 0; j < F; ++j) ref[sz(j)] += bg[sz(b * F + j)] / B;
        Values attr(sz(R * G * O)), expected(sz(R * O)), pred(sz(R * O));
        const auto xv = view(X.data(), R, F);
        const auto cv = view(coef.data(), F, O);
        N4M_TEST_REQUIRE(n4m_explain_linear_v1(ctx.p, &xv, &cv, icpt.data(), ref.data(),
                                               groups.data(), G, attr.data(), R * G * O,
                                               expected.data(), R * O, pred.data(), R * O) ==
                         N4M_OK);
        N4M_TEST_REQUIRE(max_abs_diff(g.attr, attr) <= 1e-8);
        N4M_TEST_REQUIRE(max_abs_diff(g.expected, expected) <= 1e-12);
        N4M_TEST_REQUIRE(max_abs_diff(g.pred, pred) <= 1e-12);
        for (int64_t r = 0; r < R; ++r)
            for (int64_t j = 0; j < F; ++j)
                for (int64_t o = 0; o < O; ++o)
                    N4M_TEST_REQUIRE(std::fabs(g.grads[sz((r * F + j) * O + o)] -
                                               coef[sz(j * O + o)]) <= 1e-8);
        N4M_TEST_REQUIRE(max_abs_diff(g.mean_grad, coef) <= 1e-8);
        for (double h : g.steps) N4M_TEST_REQUIRE(h > 0.0 && h < 1e-2);
    }
    // A quadratic: central differences are exact up to rounding (g = 2 x).
    const auto quad = [&](const double* rows, int64_t n, double* out) {
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < F; ++j) s += rows[i * F + j] * rows[i * F + j];
            out[i] = s;
        }
    };
    const Session g = run_gradient(ctx.p, X, R, F, bg, B, {}, F, 1,
                                   N4M_EXPLAIN_BASELINE_MEAN, quad, 1e-2);
    for (int64_t r = 0; r < R; ++r)
        for (int64_t j = 0; j < F; ++j)
            N4M_TEST_REQUIRE(std::fabs(g.grads[sz(r * F + j)] - 2.0 * X[sz(r * F + j)]) <= 1e-7);

    // INTERPOLATE is refused for GRADIENT; the step setter is GRADIENT-only.
    n4m_explain_options_v1_t opt;
    N4M_TEST_REQUIRE(n4m_explain_options_init_v1(&opt) == N4M_OK);
    opt.method = N4M_EXPLAIN_GRADIENT;
    opt.baseline = N4M_EXPLAIN_BASELINE_INTERPOLATE;
    const auto xv = view(X.data(), R, F);
    const auto bv = view(bg.data(), B, F);
    n4m_explain_session_t* s = nullptr;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, F, &opt, &s) ==
                     N4M_ERR_INVALID_ARGUMENT);
    opt.method = N4M_EXPLAIN_OCCLUSION;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, F, &opt, &s) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(n4m_explain_session_set_gradient_step_v1(ctx.p, s, 1e-3) ==
                     N4M_ERR_INVALID_ARGUMENT);
    Values gbuf(sz(R * F));
    N4M_TEST_REQUIRE(n4m_explain_session_gradients_v1(ctx.p, s, gbuf.data(), R * F, nullptr, 0,
                                                      nullptr, 0) == N4M_ERR_INVALID_ARGUMENT);
    n4m_explain_session_destroy(s);
}

void perturbed_rows() {
    Ctx ctx;
    const int64_t R = 2, F = 6, B = 3, G = 3;
    const Values X = {1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60};
    const Values bg = {0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, 3};
    const std::vector<int64_t> groups = {0, 0, 1, 1, 2, 2};
    const std::vector<int64_t> absent = {1};
    const auto xv = view(X.data(), R, F);
    const auto bv = view(bg.data(), B, F);
    int64_t rows = 0;
    // Query, then BACKGROUND: one row per background row.
    N4M_TEST_REQUIRE(n4m_explain_perturb_rows_v1(ctx.p, &xv, &bv, groups.data(), G,
                                                 N4M_EXPLAIN_BASELINE_BACKGROUND, absent.data(), 1,
                                                 nullptr, 0, &rows) == N4M_OK);
    N4M_TEST_REQUIRE(rows == R * B);
    Values out(sz(rows * F));
    N4M_TEST_REQUIRE(n4m_explain_perturb_rows_v1(ctx.p, &xv, &bv, groups.data(), G,
                                                 N4M_EXPLAIN_BASELINE_BACKGROUND, absent.data(), 1,
                                                 out.data(), rows, &rows) == N4M_OK);
    for (int64_t i = 0; i < R; ++i)
        for (int64_t b = 0; b < B; ++b)
            for (int64_t j = 0; j < F; ++j) {
                const double expect = groups[sz(j)] == 1 ? bg[sz(b * F + j)] : X[sz(i * F + j)];
                N4M_TEST_REQUIRE(out[sz((i * B + b) * F + j)] == expect);
            }
    // MEAN: the background column mean (2/3).
    N4M_TEST_REQUIRE(n4m_explain_perturb_rows_v1(ctx.p, &xv, &bv, groups.data(), G,
                                                 N4M_EXPLAIN_BASELINE_MEAN, absent.data(), 1,
                                                 out.data(), static_cast<int64_t>(out.size()),
                                                 &rows) == N4M_OK);
    N4M_TEST_REQUIRE(rows == R && std::fabs(out[2] - 2.0 / 3.0) <= 1e-15 && out[1] == 2.0);
    // INTERPOLATE: features 2..3 bridged linearly between x1 and x4.
    N4M_TEST_REQUIRE(n4m_explain_perturb_rows_v1(ctx.p, &xv, nullptr, groups.data(), G,
                                                 N4M_EXPLAIN_BASELINE_INTERPOLATE, absent.data(),
                                                 1, out.data(), static_cast<int64_t>(out.size()),
                                                 &rows) == N4M_OK);
    N4M_TEST_REQUIRE(rows == R);
    N4M_TEST_REQUIRE(std::fabs(out[2] - 3.0) <= 1e-14 && std::fabs(out[3] - 4.0) <= 1e-14);
    N4M_TEST_REQUIRE(std::fabs(out[F + 2] - 30.0) <= 1e-13);
    // Refusals: repeated / out-of-range groups, small buffers.
    const std::vector<int64_t> bad = {1, 1};
    N4M_TEST_REQUIRE(n4m_explain_perturb_rows_v1(ctx.p, &xv, &bv, groups.data(), G,
                                                 N4M_EXPLAIN_BASELINE_MEAN, bad.data(), 2,
                                                 out.data(), 100, &rows) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_perturb_rows_v1(ctx.p, &xv, &bv, groups.data(), G,
                                                 N4M_EXPLAIN_BASELINE_BACKGROUND, absent.data(), 1,
                                                 out.data(), 5, &rows) == N4M_ERR_SHAPE_MISMATCH);
}

void profile_helpers() {
    Ctx ctx;
    // Monte-Carlo standard error of the mean signed attribution.
    const Values se = {3, 0, 4, 1, 0, 2, 0, 2};  // [row][group][output], 2 x 2 x 2
    Values mse(2);
    N4M_TEST_REQUIRE(n4m_explain_summarize_std_error_v1(se.data(), 2, 2, 2, 0, mse.data(), 2) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(mse[0] - 1.5) <= 1e-15 && std::fabs(mse[1] - 2.0) <= 1e-15);

    // Aggregation rules.
    const Values v = {1.0, -3.0, 2.0, -0.5, 4.0};
    const std::vector<int64_t> groups = {0, 0, 1, 1, 1};
    const double expect[5][2] = {{-2.0, 5.5}, {-1.0, 5.5 / 3.0}, {4.0, 6.5},
                                 {2.0, 6.5 / 3.0}, {3.0, 4.0}};
    for (int32_t rule = 0; rule < 5; ++rule) {
        Values out(2);
        N4M_TEST_REQUIRE(n4m_explain_aggregate_profile_v1(ctx.p, v.data(), 5, groups.data(), 2,
                                                          rule, out.data(), 2) == N4M_OK);
        N4M_TEST_REQUIRE(std::fabs(out[0] - expect[rule][0]) <= 1e-15 &&
                         std::fabs(out[1] - expect[rule][1]) <= 1e-15);
    }
    Values out(2);
    N4M_TEST_REQUIRE(n4m_explain_aggregate_profile_v1(ctx.p, v.data(), 5, groups.data(), 2, 5,
                                                      out.data(), 2) == N4M_ERR_INVALID_ARGUMENT);

    // Spearman (scipy.stats.spearmanr 1.15.3, ties) and top-4 Jaccard.
    const Values P = {3, 1, 2, 2, 5, 0.5, 4, 4,  1, 2, 2, 3, 4, 0.2, 5, 6,
                      0, 0, 1, 1, 2, 2,   3, 3,  5, 4, 3, 2, 1, 0,   -1, -2};
    const double scipy[4][4] = {
        {1.0, 0.7575896711961079, 0.4938647983247948, -0.36147201953390995},
        {0.7575896711961079, 1.0, 0.6626889364596515, -0.6347419188140273},
        {0.4938647983247948, 0.6626889364596515, 1.0, -0.9759000729485332},
        {-0.36147201953390995, -0.6347419188140273, -0.9759000729485332, 1.0}};
    const double jac[4][4] = {{1.0, 0.6, 0.6, 1.0 / 7.0},
                              {0.6, 1.0, 0.6, 1.0 / 7.0},
                              {0.6, 0.6, 1.0, 0.0},
                              {1.0 / 7.0, 1.0 / 7.0, 0.0, 1.0}};
    const auto pv = view(P.data(), 4, 8);
    Values rho(16), J(16);
    N4M_TEST_REQUIRE(n4m_explain_compare_profiles_v1(ctx.p, &pv, 4, rho.data(), 16, J.data(),
                                                     16) == N4M_OK);
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) {
            N4M_TEST_REQUIRE(std::fabs(rho[sz(i * 4 + k)] - scipy[i][k]) <= 1e-12);
            N4M_TEST_REQUIRE(std::fabs(J[sz(i * 4 + k)] - jac[i][k]) <= 1e-15);
        }
    // Infinite values rank above every finite one; a constant profile has NaN rho.
    const double inf = std::numeric_limits<double>::infinity();
    const Values Q = {1, 2, inf, 3, 1, 2, 4, 3, 7, 7, 7, 7};
    const auto qv = view(Q.data(), 3, 4);
    Values r2(9);
    N4M_TEST_REQUIRE(n4m_explain_compare_profiles_v1(ctx.p, &qv, 2, r2.data(), 9, nullptr, 0) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(r2[1] - 1.0) <= 1e-12 && std::isnan(r2[2]) && std::isnan(r2[8]));
    const Values nan_profile = {1, std::nan(""), 2, 3};
    const auto nv = view(nan_profile.data(), 1, 4);
    N4M_TEST_REQUIRE(n4m_explain_compare_profiles_v1(ctx.p, &nv, 2, r2.data(), 9, nullptr, 0) ==
                     N4M_ERR_INVALID_ARGUMENT);
}

}  // namespace

int main() {
    n4m_testing::Runner r("importance");
    r.run("affine coefficients of PLS and Ridge (fit and N4ME import)",
          affine_coefficients_pls_and_ridge);
    r.run("latent structure, VIP, selectivity ratio, standardized coefficients",
          latent_structure_and_importances);
    r.run("pipeline raw-input affine map equals predictions; SNV refused",
          pipeline_input_affine_exact);
    r.run("declared affine transforms are exactly affine", affine_transform_traits_are_exact);
    r.run("GRADIENT equals linear SHAP on affine models",
          gradient_equals_linear_shap_on_affine_models);
    r.run("perturbed coalition rows", perturbed_rows);
    r.run("profile std error, aggregation and comparison", profile_helpers);
    return r.finalize();
}
