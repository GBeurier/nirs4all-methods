// SPDX-License-Identifier: CECILL-2.1
// Magnitude-sweep property tests for the Preflight numeric kernels owned by
// the SNV, noise-proxy and tolerance-pair diagnostics (ABI 2.20). Every
// output must equal its analytic or invariant reference (within the stated
// error) or be an explicit error status; a silently wrong number fails.
//
// Sweep: inputs scaled by 2^k, k in {-1074, -1022, -500, -100, 0, 100, 500,
// 1000}; translations by exactly representable offsets near 1e16 (2^53) and
// 1e300 (2^996, data aligned on 2^944); constant columns of huge amplitude.
#include "n4m/metrics.h"
#include "n4m/transform/scatter.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " @" + std::to_string(__LINE__)); } while (false)

namespace {
const int sweep[] = {-1074, -1022, -500, -100, 0, 100, 500, 1000};
const double dmax = std::numeric_limits<double>::max();
const double tiny = std::numeric_limits<double>::denorm_min();

n4m_matrix_view_t view(double* x, int64_t n, int64_t p) {
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x, n, p, N4M_DTYPE_F64) == N4M_OK);
    return v;
}
bool rel(double x, double y, double r) {
    return std::isfinite(x) && (y == 0 ? x == 0 : std::abs(x - y) <= r * std::abs(y));
}
// Scaled value equals ldexp(base, k): exactly when normal, else within one
// subnormal step and still positive when the base is positive.
bool scaled(double x, double base, int k) {
    const double want = std::ldexp(base, k);
    if (std::abs(want) >= std::numeric_limits<double>::min()) return x == want;
    return std::abs(x - want) <= tiny && (base <= 0 || x > 0);
}

struct Snv { std::vector<double> z, bound; n4m_status_t status; };
Snv snv(n4m_context_t* ctx, std::vector<double> x, int64_t n, int64_t p) {
    Snv out{std::vector<double>(x.size()), std::vector<double>(static_cast<size_t>(n)), N4M_OK};
    auto X = view(x.data(), n, p), O = view(out.z.data(), n, p);
    out.status = n4m_preflight_snv_compute(ctx, &X, nullptr, n, &O, out.bound.data());
    return out;
}

void snv_sweep(n4m_context_t* ctx) {
    // Integer rows (exact at every 2^k, including multiples of 2^-1074).
    const std::vector<double> base = {1, 2, 4, 8, 3, 16, 0, 5, 7, 7, 1, 9};
    const auto ref = snv(ctx, base, 2, 6);
    REQUIRE(ref.status == N4M_OK);
    for (int k : sweep) {
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        const auto s = snv(ctx, x, 2, 6);
        REQUIRE(s.status == N4M_OK);
        // Power-of-two scaling is exact, so the SNV is bitwise invariant.
        REQUIRE(s.z == ref.z && s.bound == ref.bound);
    }
    // Translation: even data + 2^53 (ulp 2) and data * 2^944 + 2^996.
    for (const auto& [unit, offset] : {std::pair{2.0, std::ldexp(1.0, 53)},
                                       std::pair{std::ldexp(1.0, 944), std::ldexp(1.0, 996)}}) {
        std::vector<double> x(base);
        for (double& v : x) v = v * unit + offset;
        const auto s = snv(ctx, x, 2, 6);
        REQUIRE(s.status == N4M_OK);
        for (size_t c = 0; c < x.size(); ++c)
            REQUIRE(std::abs(s.z[c] - ref.z[c]) <= s.bound[c / 6] + ref.bound[c / 6]);
    }
    // One huge entry among zeros: z = -1/sqrt(n-1) and sqrt(n-1) for any H.
    for (double h : {tiny, 1e-300, 1.0, 1e300, dmax, -dmax}) {
        const auto s = snv(ctx, {0, 0, 0, 0, h}, 1, 5);
        REQUIRE(s.status == N4M_OK);
        const double sign = h > 0 ? 1.0 : -1.0;
        for (int j = 0; j < 4; ++j) REQUIRE(rel(s.z[static_cast<size_t>(j)], -sign / 2.0, 1e-15));
        REQUIRE(rel(s.z[4], sign * 2.0, 1e-15) && s.bound[0] < 1e-14);
    }
    // Centre-only output that is not representable is an explicit error.
    n4m_pp_snv_handle_t* h = nullptr;
    REQUIRE(n4m_transform_snv_create(&h, 1, 0, 0) == N4M_OK);
    double over[3] = {dmax, dmax, -dmax}, out[3];
    REQUIRE(n4m_transform_snv_transform(h, view(over, 1, 3), view(out, 1, 3)) ==
            N4M_ERR_NUMERICAL_FAILURE);
    n4m_transform_snv_destroy(h);
}

struct Pairs { int64_t total; double distance; };
Pairs pairs(n4m_context_t* ctx, std::vector<double> q, std::vector<double> r, double tol) {
    const auto p = static_cast<int64_t>(q.size());
    auto Q = view(q.data(), 1, p), R = view(r.data(), 1, p);
    int64_t idx[2], counts[2];
    double dist[1] = {-1.0};
    uint8_t qf[1], rf[1];
    REQUIRE(n4m_preflight_pairs_within_tolerance(ctx, &Q, &R, nullptr, nullptr, tol, 1, idx,
                                                 dist, qf, rf, counts) == N4M_OK);
    return {counts[0], dist[0]};
}

void pairs_sweep(n4m_context_t* ctx) {
    // q - r = [3, 4, 0, 12]: RMS sqrt(169/4) = 6.5 exactly.
    const std::vector<double> q = {5, 4, 9, 12}, r = {2, 0, 9, 0};
    for (int k : sweep) {
        std::vector<double> qs(q), rs(r);
        for (double& v : qs) v = std::ldexp(v, k);
        for (double& v : rs) v = std::ldexp(v, k);
        // 6.5 2^-1074 is not representable (it rounds to 6 2^-1074): there the
        // smallest admissible tolerance is 7 2^-1074.
        const bool exact = std::ldexp(std::ldexp(6.5, k), -k) == 6.5;
        const double tol = std::ldexp(exact ? 6.5 : 7.0, k);
        const auto in = pairs(ctx, qs, rs, tol);
        REQUIRE(in.total == 1 && scaled(in.distance, 6.5, k));
        if (exact) REQUIRE(pairs(ctx, qs, rs, std::nextafter(tol, 0.0)).total == 0);
        REQUIRE(pairs(ctx, qs, rs, std::ldexp(6.0, k)).total == 0);
    }
    // Translation by exactly representable offsets leaves differences intact.
    for (const auto& [unit, offset] : {std::pair{1.0, std::ldexp(1.0, 52)},
                                       std::pair{std::ldexp(1.0, 944), std::ldexp(1.0, 996)}}) {
        std::vector<double> qs(q), rs(r);
        for (double& v : qs) v = v * unit + offset;
        for (double& v : rs) v = v * unit + offset;
        const auto in = pairs(ctx, qs, rs, 6.5 * unit);
        REQUIRE(in.total == 1 && in.distance == 6.5 * unit);
        REQUIRE(pairs(ctx, qs, rs, 6.4 * unit).total == 0);
    }
    // A constant column of huge amplitude adds a zero difference:
    // RMS over 5 columns = 13 / sqrt(5).
    for (double h : {1e300, dmax, -dmax}) {
        std::vector<double> qs(q), rs(r);
        qs.push_back(h);
        rs.push_back(h);
        const auto in = pairs(ctx, qs, rs, 13.0 / std::sqrt(5.0) * (1 + 1e-15));
        REQUIRE(in.total == 1 && rel(in.distance, 13.0 / std::sqrt(5.0), 4e-16));
        REQUIRE(pairs(ctx, qs, rs, 13.0 / std::sqrt(5.0) * (1 - 1e-15)).total == 0);
    }
    // A difference beyond DBL_MAX in one of 8 columns: RMS = 2 dmax / sqrt(8) is
    // representable; over 1 column it is not and cannot be within tolerance.
    std::vector<double> big(8, 0.0), neg(8, 0.0);
    big[0] = dmax;
    neg[0] = -dmax;
    const auto wide = pairs(ctx, big, neg, dmax);
    REQUIRE(wide.total == 1 && rel(wide.distance, dmax / std::sqrt(2.0), 4e-16));
    REQUIRE(pairs(ctx, {dmax}, {-dmax}, dmax).total == 0);
}

struct Noise { std::vector<double> band; double pooled; n4m_status_t status; };
Noise noise(n4m_context_t* ctx, std::vector<double> x, int64_t n, int64_t p, int32_t w,
            int32_t order) {
    Noise out{std::vector<double>(static_cast<size_t>(p)), 0.0, N4M_OK};
    std::vector<int64_t> counts(static_cast<size_t>(p));
    double two[2];
    auto X = view(x.data(), n, p);
    out.status = n4m_preflight_noise_proxy_compute(ctx, &X, nullptr, w, order, out.band.data(),
                                                   counts.data(), two);
    out.pooled = two[0];
    return out;
}

void noise_sweep(n4m_context_t* ctx) {
    // Quadratic row (zero residual) and linear row with an impulse of 35 at
    // band 5: window 5 / order 2 residuals are hand-known (test_preflight).
    std::vector<double> base;
    for (int j = 0; j < 11; ++j) base.push_back(j * j);
    for (int j = 0; j < 11; ++j) base.push_back(1 + 2 * j + (j == 5 ? 35 : 0));
    const auto ref = noise(ctx, base, 2, 11, 5, 2);
    REQUIRE(ref.status == N4M_OK && rel(ref.pooled, std::sqrt(87.5), 1e-14));
    for (int k : sweep) {
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        const auto s = noise(ctx, x, 2, 11, 5, 2);
        REQUIRE(s.status == N4M_OK);
        REQUIRE(scaled(s.pooled, ref.pooled, k));
        for (int j = 2; j < 9; ++j)
            REQUIRE(scaled(s.band[static_cast<size_t>(j)], ref.band[static_cast<size_t>(j)], k));
    }
    // Translation by exact offsets: identical residuals, identical output.
    for (const auto& [unit, offset] : {std::pair{1.0, std::ldexp(1.0, 52)},
                                       std::pair{std::ldexp(1.0, 944), std::ldexp(1.0, 996)}}) {
        std::vector<double> x(base);
        for (double& v : x) v = v * unit + offset;
        const auto s = noise(ctx, x, 2, 11, 5, 2);
        REQUIRE(s.status == N4M_OK && rel(s.pooled, ref.pooled * unit, 1e-15));
    }
    // Residual noise above DBL_MAX is an explicit error, not Inf.
    std::vector<double> alt;
    for (int j = 0; j < 9; ++j) alt.push_back(j % 2 ? -dmax : dmax);
    REQUIRE(noise(ctx, alt, 1, 9, 3, 1).status == N4M_ERR_NUMERICAL_FAILURE);
    // A positive residual far below 2^-1074 after unscaling is published as
    // 2^-1074 (documented), never as 0.
    std::vector<double> sub(11, 0.0);
    sub[5] = tiny;
    const auto s = noise(ctx, sub, 1, 11, 5, 2);
    REQUIRE(s.status == N4M_OK && s.band[5] > 0 && s.pooled > 0);
}
}  // namespace

int main() {
    try {
        n4m_context_t* ctx = nullptr;
        REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        snv_sweep(ctx);
        pairs_sweep(ctx);
        noise_sweep(ctx);
        n4m_context_destroy(ctx);
        std::puts("preflight robustness sweeps: ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
