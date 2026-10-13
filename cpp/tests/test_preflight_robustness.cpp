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
struct PairGolden { std::vector<double> q, r; double tol; bool within; double rms; };
struct SnvGolden { std::vector<double> x, z; };
// Test-only goldens: exact rational decisions (Python 3.10 fractions) and
// 400-digit decimal square roots, rounded to binary64 (repr round-trips).
const PairGolden pair_goldens[] = {
    {{1.0, 1.0}, {-5.551115123125783e-17, -5.551115123125783e-17}, 0.9999999999999999, false, 1.0},
    {{1.0, 1.0}, {-5.551115123125783e-17, -5.551115123125783e-17}, 1.0, false, 1.0},
    {{1.0, 1.0}, {-5.551115123125783e-17, -5.551115123125783e-17}, 1.0000000000000002, true, 1.0},
    {{1.0, 1.0, 1.4901161193847656e-08}, {0.0, 0.0, 0.0}, 0.8164965809277259, false, 0.816496580927726},
    {{1.0, 1.0, 1.4901161193847656e-08}, {0.0, 0.0, 0.0}, 0.816496580927726, false, 0.816496580927726},
    {{1.0, 1.0, 1.4901161193847656e-08}, {0.0, 0.0, 0.0}, 0.8164965809277261, true, 0.816496580927726},
    {{1.0, -1.0, 1.0, -1.0}, {0.0, 0.0, 0.0, 0.0}, 0.9999999999999999, false, 1.0},
    {{1.0, -1.0, 1.0, -1.0}, {0.0, 0.0, 0.0, 0.0}, 1.0, true, 1.0},
    {{1.0, -1.0, 1.0, -1.0}, {0.0, 0.0, 0.0, 0.0}, 1.0000000000000002, true, 1.0},
    {{4.149515568880993e+180, 2.409919865102884e-181, -2.7997908555096566e-301}, {4.149515568880993e+180, 0.0, 9.332636185032189e-302}, 1.3913678828425764e-181, false, 1.3913678828425767e-181},
    {{4.149515568880993e+180, 2.409919865102884e-181, -2.7997908555096566e-301}, {4.149515568880993e+180, 0.0, 9.332636185032189e-302}, 1.3913678828425767e-181, false, 1.3913678828425767e-181},
    {{4.149515568880993e+180, 2.409919865102884e-181, -2.7997908555096566e-301}, {4.149515568880993e+180, 0.0, 9.332636185032189e-302}, 1.391367882842577e-181, true, 1.3913678828425767e-181},
    {{1e+300, 1.0, -1e-300}, {-1e+300, 1.0, 1e-300}, 1.1547005383792515e+300, false, 1.1547005383792516e+300},
    {{1e+300, 1.0, -1e-300}, {-1e+300, 1.0, 1e-300}, 1.1547005383792516e+300, true, 1.1547005383792516e+300},
    {{1e+300, 1.0, -1e-300}, {-1e+300, 1.0, 1e-300}, 1.1547005383792518e+300, true, 1.1547005383792516e+300},
    {{3.0, -4.0, 8.673617379884035e-19}, {-3.0, 4.0, 0.0}, 5.773502691896256, false, 5.773502691896257},
    {{3.0, -4.0, 8.673617379884035e-19}, {-3.0, 4.0, 0.0}, 5.773502691896257, false, 5.773502691896257},
    {{3.0, -4.0, 8.673617379884035e-19}, {-3.0, 4.0, 0.0}, 5.773502691896258, true, 5.773502691896257},
    {{1e+308, 5e-324}, {1e+308, 0.0}, 5e-324, true, 5e-324},
    {{1e+308, 5e-324}, {1e+308, 0.0}, 1e-323, true, 5e-324},
    {{-1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1.271161006153646e+308, false, 1.2711610061536462e+308},
    {{-1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1.2711610061536462e+308, true, 1.2711610061536462e+308},
    {{-1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1.2711610061536464e+308, true, 1.2711610061536462e+308},
    {{1e-323, 0.0}, {0.0, 0.0}, 5e-324, false, 5e-324},
    {{1e-323, 0.0}, {0.0, 0.0}, 1e-323, true, 5e-324},
    {{1e+16, 1.0}, {9999999999999998.0, 0.0}, 1.5811388300841895, false, 1.5811388300841898},
    {{1e+16, 1.0}, {9999999999999998.0, 0.0}, 1.5811388300841898, true, 1.5811388300841898},
    {{1e+16, 1.0}, {9999999999999998.0, 0.0}, 1.58113883008419, true, 1.5811388300841898},
};
const SnvGolden snv_goldens[] = {
    {{1.0715086071862673e+301, 9.332636185032189e-302, -9.332636185032189e-302, 0.0}, {1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257}},
    {{1e+300, 1.0, 2.0, 3.0}, {1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257}},
    {{-1e-300, 1e-310, 5e-324, 0.0}, {-1.7320508075688772, 0.5773502693435858, 0.5773502691126458, 0.5773502691126458}},
    {{5e-324, 1e-323, 1.5e-323, 0.0}, {-0.4472135954999579, 0.4472135954999579, 1.3416407864998738, -1.3416407864998738}},
    {{1e+16, 1.0, -1.0, 1.0000000000000002e+16}, {0.9999999999999998, -0.9999999999999998, -1.0000000000000002, 1.0000000000000002}},
    {{1.7976931348623157e+308, -1.7976931348623157e+308, 1e-300, 0.0}, {1.4142135623730951, -1.4142135623730951, 0.0, -0.0}},
    {{1.0, 1.0000000000000002, 1.0, 1.0}, {-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896257}},
};

// Exact threshold decisions on mixed-exponent rows (b14fix2 #1): rational
// oracle; tolerances are the nearest double to the true RMS and its two
// neighbours. Accepted distances match the oracle RMS within 2^-50 relative
// (one subnormal step below 2^-1022) and never exceed the tolerance.
void oracle_pairs(n4m_context_t* ctx) {
    for (const auto& g : pair_goldens) {
        const auto got = pairs(ctx, g.q, g.r, g.tol);
        REQUIRE((got.total == 1) == g.within);
        if (!g.within) continue;
        REQUIRE(got.distance > 0.0 && got.distance <= g.tol);
        if (g.rms >= std::numeric_limits<double>::min())
            REQUIRE(std::abs(got.distance - g.rms) <= 0x1p-50 * g.rms);
        else
            REQUIRE(std::abs(got.distance - g.rms) <= tiny);
    }
}

// SNV of mixed-exponent rows against 400-digit references: error within the
// reported bound plus the rounding of the reference.
void oracle_snv(n4m_context_t* ctx) {
    for (const auto& g : snv_goldens) {
        const auto s = snv(ctx, g.x, 1, static_cast<int64_t>(g.x.size()));
        REQUIRE(s.status == N4M_OK && s.bound[0] < 1e-14);
        for (size_t j = 0; j < g.z.size(); ++j)
            REQUIRE(std::abs(s.z[j] - g.z[j]) <= s.bound[0] + 0x1p-53 * std::abs(g.z[j]));
    }
}

// Noise proxy with huge values next to small ones (b14fix2 #2). Row
// [H,0,0,0,0,1,0,0,0,0,0], index 1 masked, window 3 / order 1 (weights 1/3,
// gain sqrt(2/3)): band 5 = sqrt(2/3), pooled over the 7 evaluable bands =
// 1/sqrt(7), whatever H, because H is outside every evaluable window. With
// index 1 valid, H enters band 1 only: residual -H/3, band 5 unchanged.
void noise_mixed(n4m_context_t* ctx) {
    for (double big : {0.0, 1.0, 1e16, 1e300, dmax, -1e300, tiny, -dmax}) {
        std::vector<double> x = {big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        std::vector<uint8_t> mask(11, 1);
        mask[1] = 0;
        auto X = view(x.data(), 1, 11);
        std::vector<double> band(11);
        std::vector<int64_t> counts(11);
        double two[2];
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx, &X, mask.data(), 3, 1, band.data(),
                                                  counts.data(), two) == N4M_OK);
        REQUIRE(rel(band[5], std::sqrt(2.0 / 3.0), 1e-15) && rel(two[0], 1 / std::sqrt(7.0), 1e-15));
        REQUIRE(band[3] == 0.0 && counts[2] == 0 && counts[3] == 1);
        // Unmasked: band 1 residual -H/3, i.e. noise |H|/3 / sqrt(2/3).
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx, &X, nullptr, 3, 1, band.data(),
                                                  counts.data(), two) == N4M_OK);
        REQUIRE(rel(band[5], std::sqrt(2.0 / 3.0), 1e-15));
        if (std::abs(big) >= 1.0)
            REQUIRE(rel(band[1], std::abs(big) / 3 / std::sqrt(2.0 / 3.0), 1e-15));
        else if (big != 0.0)  // 0.41 2^-1074 is not representable: published as 2^-1074
            REQUIRE(band[1] == tiny);
    }
    // Signed mixture inside one window: [-2^500, 2^-500, 2^500] has the
    // residual 2^-500 (1 - 1/3) at the centre; |2^-500| is not lost.
    std::vector<double> x = {0, 0, -std::ldexp(1.0, 500), std::ldexp(1.0, -500),
                             std::ldexp(1.0, 500), 0, 0};
    auto X = view(x.data(), 1, 7);
    std::vector<double> band(7);
    std::vector<int64_t> counts(7);
    double two[2];
    REQUIRE(n4m_preflight_noise_proxy_compute(ctx, &X, nullptr, 3, 1, band.data(),
                                              counts.data(), two) == N4M_OK);
    REQUIRE(rel(band[3], std::ldexp(2.0 / 3.0, -500) / std::sqrt(2.0 / 3.0), 1e-15));
    // Window [-dmax, 2^-1000, dmax]: the second difference 2^-999 survives the
    // 1/8 prescaling of a window that reaches 2^1021.
    x = {0, 0, -dmax, std::ldexp(1.0, -1000), dmax, 0, 0};
    X = view(x.data(), 1, 7);
    REQUIRE(n4m_preflight_noise_proxy_compute(ctx, &X, nullptr, 3, 1, band.data(),
                                              counts.data(), two) == N4M_OK);
    REQUIRE(rel(band[3], std::ldexp(2.0 / 3.0, -1000) / std::sqrt(2.0 / 3.0), 1e-15));
}
}  // namespace

int main() {
    try {
        n4m_context_t* ctx = nullptr;
        REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        snv_sweep(ctx);
        pairs_sweep(ctx);
        noise_sweep(ctx);
        oracle_pairs(ctx);
        oracle_snv(ctx);
        noise_mixed(ctx);
        n4m_context_destroy(ctx);
        std::puts("preflight robustness sweeps: ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
