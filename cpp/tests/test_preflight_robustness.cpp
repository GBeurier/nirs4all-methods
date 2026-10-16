// SPDX-License-Identifier: CECILL-2.1
// Property and oracle tests for the Preflight SNV, noise-proxy and
// tolerance-pair kernels (ABI 2.20) on the supported numerical domain D
// (lead rule 13): every nonzero |x| in [2^-256, 2^256] and, per row,
// max/min nonzero |x| <= 2^64. Inside D every output must match its
// analytic, rational or invariant reference; just outside D the kernels must
// refuse with N4M_ERR_UNSUPPORTED. A silently wrong number fails.
//
// Sweeps: scales 2^k for k in {-256, -200, -100, 0, 100, 200, 248} and the
// out-of-domain neighbours 2^-257 and 2^253; exact translations near 1e16
// (2^53) and 2^200; huge constant columns up to the spread limit; mixed
// exponents inside one row against independent oracles.
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
const int sweep[] = {-256, -200, -100, 0, 100, 200, 248};
const double dmax = std::numeric_limits<double>::max();
const double tiny = std::numeric_limits<double>::denorm_min();
double p2(int e) { return std::ldexp(1.0, e); }

n4m_matrix_view_t view(double* x, int64_t n, int64_t p) {
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x, n, p, N4M_DTYPE_F64) == N4M_OK);
    return v;
}
bool rel(double x, double y, double r) {
    return std::isfinite(x) && (y == 0 ? x == 0 : std::abs(x - y) <= r * std::abs(y));
}

struct Snv { std::vector<double> z, bound; n4m_status_t status; };
Snv snv(n4m_context_t* ctx, std::vector<double> x, int64_t n, int64_t p) {
    Snv out{std::vector<double>(x.size()), std::vector<double>(static_cast<size_t>(n)), N4M_OK};
    auto X = view(x.data(), n, p), O = view(out.z.data(), n, p);
    out.status = n4m_preflight_snv_compute(ctx, &X, nullptr, n, &O, out.bound.data());
    return out;
}

void snv_sweep(n4m_context_t* ctx) {
    const std::vector<double> base = {1, 2, 4, 8, 3, 16, 0, 5, 7, 7, 1, 9};
    const auto ref = snv(ctx, base, 2, 6);
    REQUIRE(ref.status == N4M_OK);
    for (int k : sweep) {
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        const auto s = snv(ctx, x, 2, 6);
        // Power-of-two scaling is exact: the SNV is bitwise invariant.
        REQUIRE(s.status == N4M_OK && s.z == ref.z && s.bound == ref.bound);
    }
    for (int k : {-257, 253}) {  // 1 2^-257 or 16 2^253 = 2^257 leave D
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        REQUIRE(snv(ctx, x, 2, 6).status == N4M_ERR_UNSUPPORTED);
    }
    // Translation: even data + 2^53 (ulp 2) and data 2^148 + 2^200 (ulp 2^148).
    for (const auto& [unit, offset] : {std::pair{2.0, p2(53)}, std::pair{p2(148), p2(200)}}) {
        std::vector<double> x(base);
        for (double& v : x) v = v * unit + offset;
        const auto s = snv(ctx, x, 2, 6);
        REQUIRE(s.status == N4M_OK);
        for (size_t c = 0; c < x.size(); ++c)
            REQUIRE(std::abs(s.z[c] - ref.z[c]) <= s.bound[c / 6] + ref.bound[c / 6]);
    }
    // One nonzero entry: z = -1/sqrt(n-1) and sqrt(n-1) for any H in D.
    for (double h : {p2(-256), 1.0, p2(256), -p2(256)}) {
        const auto s = snv(ctx, {0, 0, 0, 0, h}, 1, 5);
        REQUIRE(s.status == N4M_OK);
        const double sign = h > 0 ? 1.0 : -1.0;
        for (int j = 0; j < 4; ++j) REQUIRE(rel(s.z[static_cast<size_t>(j)], -sign / 2.0, 1e-15));
        REQUIRE(rel(s.z[4], sign * 2.0, 1e-15) && s.bound[0] < 1e-14);
    }
    // Outside D, refused before computing.
    for (const auto& row : {std::vector<double>{1, 0, 0, dmax}, std::vector<double>{1, 0, 0, tiny},
                            std::vector<double>{1, 0, 0, std::nextafter(p2(64), INFINITY)}})
        REQUIRE(snv(ctx, row, 1, 4).status == N4M_ERR_UNSUPPORTED);
    REQUIRE(snv(ctx, {1, 0, 0, p2(64)}, 1, 4).status == N4M_OK);
}

struct Pairs { int64_t total; double distance; n4m_status_t status; };
Pairs pairs(n4m_context_t* ctx, std::vector<double> q, std::vector<double> r, double tol) {
    const auto p = static_cast<int64_t>(q.size());
    auto Q = view(q.data(), 1, p), R = view(r.data(), 1, p);
    int64_t idx[2], counts[2] = {-1, -1};
    double dist[1] = {-1.0};
    uint8_t qf[1], rf[1];
    const auto st = n4m_preflight_pairs_within_tolerance(ctx, &Q, &R, nullptr, nullptr, tol, 1,
                                                         idx, dist, qf, rf, counts);
    return {counts[0], dist[0], st};
}

void pairs_sweep(n4m_context_t* ctx) {
    // q - r = [3, 4, 0, 12]: RMS sqrt(169/4) = 6.5 exactly.
    const std::vector<double> q = {5, 4, 9, 12}, r = {2, 0, 9, 0};
    for (int k : sweep) {
        std::vector<double> qs(q), rs(r);
        for (double& v : qs) v = std::ldexp(v, k);
        for (double& v : rs) v = std::ldexp(v, k);
        const double tol = std::ldexp(6.5, k);
        const auto in = pairs(ctx, qs, rs, tol);
        REQUIRE(in.status == N4M_OK && in.total == 1 && in.distance == tol);
        REQUIRE(pairs(ctx, qs, rs, std::nextafter(tol, 0.0)).total == 0);
        REQUIRE(pairs(ctx, qs, rs, std::ldexp(6.0, k)).total == 0);
    }
    for (int k : {-258, 253}) {  // smallest value 2 2^-258 = 2^-257, largest 12 2^253 > 2^256
        std::vector<double> qs(q), rs(r);
        for (double& v : qs) v = std::ldexp(v, k);
        for (double& v : rs) v = std::ldexp(v, k);
        REQUIRE(pairs(ctx, qs, rs, 1.0).status == N4M_ERR_UNSUPPORTED);
    }
    for (const auto& [unit, offset] : {std::pair{1.0, p2(52)}, std::pair{p2(148), p2(200)}}) {
        std::vector<double> qs(q), rs(r);
        for (double& v : qs) v = v * unit + offset;
        for (double& v : rs) v = v * unit + offset;
        const auto in = pairs(ctx, qs, rs, 6.5 * unit);
        REQUIRE(in.status == N4M_OK && in.total == 1 && in.distance == 6.5 * unit);
        REQUIRE(pairs(ctx, qs, rs, std::nextafter(6.5 * unit, 0.0)).total == 0);
    }
    // A constant column at the spread limit (2^64 / smallest value 2) adds a
    // zero difference: RMS over 5 columns = 13 / sqrt(5). One step further is refused.
    for (double h : {p2(65), -p2(65)}) {
        std::vector<double> qs(q), rs(r);
        qs.push_back(h);
        rs.push_back(h);
        const auto in = pairs(ctx, qs, rs, 13.0 / std::sqrt(5.0) * (1 + 1e-15));
        REQUIRE(in.status == N4M_OK && in.total == 1 && rel(in.distance, 13.0 / std::sqrt(5.0), 4e-16));
        REQUIRE(pairs(ctx, qs, rs, 13.0 / std::sqrt(5.0) * (1 - 1e-15)).total == 0);
        qs.back() = rs.back() = 2 * h;
        REQUIRE(pairs(ctx, qs, rs, 1.0).status == N4M_ERR_UNSUPPORTED);
    }
    REQUIRE(pairs(ctx, {dmax}, {-dmax}, dmax).status == N4M_ERR_UNSUPPORTED);
}

struct Noise { std::vector<double> band; double pooled; n4m_status_t status; };
Noise noise(n4m_context_t* ctx, std::vector<double> x, int64_t n, int64_t p, int32_t w,
            int32_t order, const std::vector<uint8_t>& mask = {}) {
    Noise out{std::vector<double>(static_cast<size_t>(p)), 0.0, N4M_OK};
    std::vector<int64_t> counts(static_cast<size_t>(p));
    double two[2];
    auto X = view(x.data(), n, p);
    out.status = n4m_preflight_noise_proxy_compute(ctx, &X, mask.empty() ? nullptr : mask.data(),
                                                   w, order, out.band.data(), counts.data(), two);
    out.pooled = two[0];
    return out;
}

void noise_sweep(n4m_context_t* ctx) {
    // Quadratic row (zero residual) and linear row with an impulse of 35 at
    // band 5: window 5 / order 2 gives the hand-known pooled sqrt(87.5).
    std::vector<double> base;
    for (int j = 0; j < 11; ++j) base.push_back(j * j);
    for (int j = 0; j < 11; ++j) base.push_back(1 + 2 * j + (j == 5 ? 35 : 0));
    const auto ref = noise(ctx, base, 2, 11, 5, 2);
    REQUIRE(ref.status == N4M_OK && rel(ref.pooled, std::sqrt(87.5), 1e-14));
    for (int k : sweep) {
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        const auto s = noise(ctx, x, 2, 11, 5, 2);
        REQUIRE(s.status == N4M_OK && s.pooled == std::ldexp(ref.pooled, k));
        for (int j = 2; j < 9; ++j)
            REQUIRE(s.band[static_cast<size_t>(j)] == std::ldexp(ref.band[static_cast<size_t>(j)], k));
    }
    for (int k : {-257, 250}) {  // 1 2^-257 and 100 2^250 > 2^256 leave D
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        REQUIRE(noise(ctx, x, 2, 11, 5, 2).status == N4M_ERR_UNSUPPORTED);
    }
    for (const auto& [unit, offset] : {std::pair{1.0, p2(52)}, std::pair{p2(148), p2(200)}}) {
        std::vector<double> x(base);
        for (double& v : x) v = v * unit + offset;
        const auto s = noise(ctx, x, 2, 11, 5, 2);
        REQUIRE(s.status == N4M_OK && rel(s.pooled, ref.pooled * unit, 1e-15));
    }
    // Impulse a = 2^-256 (window 5 / order 2): residual (18/35) a, noise
    // sqrt(18/35) a, strictly positive.
    std::vector<double> small(11, 0.0);
    small[5] = p2(-256);
    const auto s = noise(ctx, small, 1, 11, 5, 2);
    REQUIRE(s.status == N4M_OK && rel(s.band[5], std::sqrt(18.0 / 35.0) * p2(-256), 1e-14));
}

struct PairGolden { std::vector<double> q, r; double tol; bool within; double rms; };
struct SnvGolden { std::vector<double> x, z; };
// Test-only goldens inside the supported domain: exact rational decisions
// (Python 3.10 fractions) and 400-digit decimal square roots, rounded to binary64.
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
    {{1.152921504606847e+18, 0.125, -0.1875}, {1.152921504606847e+18, 0.0, 0.0625}, 0.16137430609197567, false, 0.1613743060919757},
    {{1.152921504606847e+18, 0.125, -0.1875}, {1.152921504606847e+18, 0.0, 0.0625}, 0.1613743060919757, false, 0.1613743060919757},
    {{1.152921504606847e+18, 0.125, -0.1875}, {1.152921504606847e+18, 0.0, 0.0625}, 0.16137430609197573, true, 0.1613743060919757},
    {{1.8092513943330656e+75, 1.6069380442589903e+60, 2.4104070663884854e+60}, {-1.8092513943330656e+75, 1.6069380442589903e+60, 0.0}, 2.0891435590998018e+75, false, 2.089143559099802e+75},
    {{1.8092513943330656e+75, 1.6069380442589903e+60, 2.4104070663884854e+60}, {-1.8092513943330656e+75, 1.6069380442589903e+60, 0.0}, 2.089143559099802e+75, false, 2.089143559099802e+75},
    {{1.8092513943330656e+75, 1.6069380442589903e+60, 2.4104070663884854e+60}, {-1.8092513943330656e+75, 1.6069380442589903e+60, 0.0}, 2.0891435590998026e+75, true, 2.089143559099802e+75},
    {{3.0, -4.0, 8.673617379884035e-19}, {-3.0, 4.0, 0.0}, 5.773502691896256, false, 5.773502691896257},
    {{3.0, -4.0, 8.673617379884035e-19}, {-3.0, 4.0, 0.0}, 5.773502691896257, false, 5.773502691896257},
    {{3.0, -4.0, 8.673617379884035e-19}, {-3.0, 4.0, 0.0}, 5.773502691896258, true, 5.773502691896257},
    {{6.223015277861142e-61, 5.527147875260445e-76}, {0.0, -5.527147875260445e-76}, 4.4003363024031e-61, false, 4.400336302403101e-61},
    {{6.223015277861142e-61, 5.527147875260445e-76}, {0.0, -5.527147875260445e-76}, 4.400336302403101e-61, true, 4.400336302403101e-61},
    {{6.223015277861142e-61, 5.527147875260445e-76}, {0.0, -5.527147875260445e-76}, 4.4003363024031016e-61, true, 4.400336302403101e-61},
    {{1e+16, 1.0}, {9999999999999998.0, 0.0}, 1.5811388300841895, false, 1.5811388300841898},
    {{1e+16, 1.0}, {9999999999999998.0, 0.0}, 1.5811388300841898, true, 1.5811388300841898},
    {{1e+16, 1.0}, {9999999999999998.0, 0.0}, 1.58113883008419, true, 1.5811388300841898},
    {{1.157920892373162e+77, 6.277101735386681e+57}, {-1.157920892373162e+77, 6.277101735386681e+57}, 1.6375474301492824e+77, false, 1.6375474301492827e+77},
    {{1.157920892373162e+77, 6.277101735386681e+57}, {-1.157920892373162e+77, 6.277101735386681e+57}, 1.6375474301492827e+77, true, 1.6375474301492827e+77},
    {{1.157920892373162e+77, 6.277101735386681e+57}, {-1.157920892373162e+77, 6.277101735386681e+57}, 1.637547430149283e+77, true, 1.6375474301492827e+77},
    {{8.636168555094445e-78, 1.5930919111324523e-58}, {0.0, -1.5930919111324523e-58}, 2.2529721868303873e-58, false, 2.2529721868303876e-58},
    {{8.636168555094445e-78, 1.5930919111324523e-58}, {0.0, -1.5930919111324523e-58}, 2.2529721868303876e-58, true, 2.2529721868303876e-58},
    {{8.636168555094445e-78, 1.5930919111324523e-58}, {0.0, -1.5930919111324523e-58}, 2.252972186830388e-58, true, 2.2529721868303876e-58},
};
const SnvGolden snv_goldens[] = {
    {{1.152921504606847e+18, 0.125, -0.125, 0.0}, {1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257}},
    {{1.157920892373162e+77, 1.6069380442589903e+60, 6.277101735386681e+57, 0.0}, {1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257}},
    {{8.636168555094445e-78, 6.223015277861142e-61, 2.5908505665283334e-77, 0.0}, {-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896258}},
    {{1e+16, 1.0, -1.0, 1.0000000000000002e+16}, {0.9999999999999998, -0.9999999999999998, -1.0000000000000002, 1.0000000000000002}},
    {{9.223372036854776e+18, -9.223372036854776e+18, 0.5, 0.0}, {1.4142135623730951, -1.4142135623730951, 5.749850312562652e-20, -1.9166167708542177e-20}},
    {{1.0, 1.0000000000000002, 1.0, 1.0}, {-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896257}},
    {{-1.5930919111324523e-58, 8.636168555094445e-78, 1.727233711018889e-77, 0.0}, {-1.7320508075688772, 0.5773502691896257, 0.5773502691896257, 0.5773502691896257}},
};

// Exact threshold decisions on mixed-exponent rows (b14fix2 #1): rational
// oracle; tolerances are the nearest double to the true RMS and its two
// neighbours. Accepted distances match the oracle RMS within 2^-50 relative
// and never exceed the tolerance.
void oracle_pairs(n4m_context_t* ctx) {
    for (const auto& g : pair_goldens) {
        const auto got = pairs(ctx, g.q, g.r, g.tol);
        REQUIRE(got.status == N4M_OK && (got.total == 1) == g.within);
        if (!g.within) continue;
        REQUIRE(got.distance > 0.0 && got.distance <= g.tol);
        REQUIRE(std::abs(got.distance - g.rms) <= 0x1p-50 * g.rms);
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

// Noise proxy with mixed exponents in one row (b14fix2 #2, b14fix3 #3/#4).
// Window 3 / order 1 has weights 1/3 and gain sqrt(2/3); window 5 / order 1
// has weights 1/5 and gain sqrt(4/5).
void noise_mixed(n4m_context_t* ctx) {
    // [H,0,0,0,0,1,0,0,0,0,0], index 1 masked: H is outside every evaluable
    // window, so band 5 = sqrt(2/3) and the pooled value over the 7 evaluable
    // bands = 1/sqrt(7) for every H in D; masked huge values are never read.
    std::vector<uint8_t> mask(11, 1);
    mask[1] = 0;
    for (double big : {0.0, 1.0, 1e16, p2(64), -p2(60)}) {
        const auto s = noise(ctx, {big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, 1, 11, 3, 1, mask);
        REQUIRE(s.status == N4M_OK && rel(s.band[5], std::sqrt(2.0 / 3.0), 1e-15) &&
                rel(s.pooled, 1 / std::sqrt(7.0), 1e-15));
        // Unmasked: H enters band 1 only (residual -H/3), band 5 unchanged.
        const auto u = noise(ctx, {big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, 1, 11, 3, 1);
        REQUIRE(u.status == N4M_OK && rel(u.band[5], std::sqrt(2.0 / 3.0), 1e-15));
        REQUIRE(rel(u.band[1], std::abs(big) / 3 / std::sqrt(2.0 / 3.0), 1e-15));
    }
    REQUIRE(noise(ctx, {p2(65), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, 1, 11, 3, 1, mask).status ==
            N4M_ERR_UNSUPPORTED);
    std::vector<uint8_t> masked_big(11, 1);
    masked_big[0] = masked_big[1] = 0;
    for (double big : {1e300, dmax, tiny}) {
        const auto s = noise(ctx, {big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, 1, 11, 3, 1, masked_big);
        REQUIRE(s.status == N4M_OK && rel(s.pooled, 1 / std::sqrt(7.0), 1e-15));
    }
    // Spread 2^60 inside one window (b14fix3 #3, interior of D):
    // [-2^60, 2^60, 1, 0, 0], window 5 / order 1: residual 1 - 1/5 = 4/5,
    // noise (4/5) / sqrt(4/5) = sqrt(4/5); band and pooled agree.
    const auto w = noise(ctx, {-p2(60), p2(60), 1, 0, 0}, 1, 5, 5, 1);
    REQUIRE(w.status == N4M_OK && rel(w.band[2], std::sqrt(0.8), 1e-14) &&
            rel(w.pooled, std::sqrt(0.8), 1e-14));
    // Signed spread 2^64 in a window: [-2^60, 2^-4, 2^60] keeps its centre:
    // residual (2/3) 2^-4, noise sqrt(2/3) 2^-4.
    const auto c = noise(ctx, {0, -p2(60), p2(-4), p2(60), 0}, 1, 5, 3, 1);
    REQUIRE(c.status == N4M_OK && rel(c.band[2], std::sqrt(2.0 / 3.0) * p2(-4), 1e-14));
    // Outside D, refused (b14fix3 #3/#4).
    REQUIRE(noise(ctx, {-p2(100), p2(100), 1, 0, 0}, 1, 5, 5, 1).status == N4M_ERR_UNSUPPORTED);
    REQUIRE(noise(ctx, {-dmax, tiny, dmax}, 1, 3, 3, 1).status == N4M_ERR_UNSUPPORTED);
}
struct SummaryGolden { std::vector<double> x, want; };  // mean, sd, q1, median, q3
struct KnnGolden { std::vector<double> q, refs; std::vector<int64_t> order; std::vector<double> dist; };
struct ContrastGolden { std::vector<double> y; std::vector<int64_t> neighbor; double contrast; };
// Test-only goldens inside the supported domain (Python 3.10 fractions,
// 400-digit decimal square roots, rounded to binary64).
const SummaryGolden summary_goldens[] = {
    {{1e+16, 1.0000000000000002e+16, 1.0000000000000004e+16, 1.0000000000000008e+16, 9999999999999998.0}, {1.0000000000000002e+16, 3.847076812334269, 1e+16, 1.0000000000000002e+16, 1.0000000000000004e+16}},
    {{1.152921504606847e+18, 0.125, -0.0625, 0.0, 1.0}, {2.305843009213694e+17, 5.1560217140444934e+17, 0.0, 0.125, 1.0}},
    {{0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6}, {0.35000000000000003, 0.18708286933869706, 0.22500000000000003, 0.35000000000000003, 0.475}},
    {{8.636168555094445e-78, 2.5908505665283334e-77, 6.223015277861142e-61, 0.0}, {1.5557538194652854e-61, 3.111507638930571e-61, 6.477126416320833e-78, 1.727233711018889e-77, 1.5557538194652858e-61}},
    {{-1.6069380442589903e+60, 1.6069380442589903e+60, 1.42724769270596e+45, -1.393796574908164e+42, 1.0727468786061222e+55}, {2.1454937574974152e+54, 1.1362767880523081e+60, -1.393796574908164e+42, 1.42724769270596e+45, 1.0727468786061222e+55}},
    {{1.0000000000000002, 1.0, 0.9999999999999999, 1.0000000000000004}, {1.0000000000000002, 2.461759443637518e-16, 1.0, 1.0, 1.0000000000000002}},
};
const KnnGolden knn_goldens[] = {
    {{1.0000000000000024, 1.0000000000000089, 1.0000000000000029}, {1.5000000000000062, 1.5000000000000135, 0.9999999999999969, 0.499999999999996, 0.5000000000000095, 1.500000000000002, 0.49999999999999756, 1.0000000000000004, 1.5000000000000095, 0.4999999999999971, 1.0000000000000153, 1.5000000000000095, 0.49999999999999556, 0.5000000000000073, 1.0000000000000049}, {4, 0, 2, 3, 1}, {0.7071067811865535, 0.8660254037844415, 0.7071067811865557, 0.707106781186556, 0.7071067811865535}},
    {{1.000000000000007, 1.0000000000000042, 1.000000000000007}, {1.0000000000000064, 0.49999999999999667, 1.4999999999999984, 1.0000000000000158, 0.5000000000000075, 1.5000000000000093, 1.5000000000000098, 1.0000000000000095, 1.000000000000002, 1.5000000000000013, 1.5000000000000062, 1.50000000000001, 0.5000000000000098, 1.0000000000000022, 1.5}, {2, 4, 1, 0, 3}, {0.7071067811865467, 0.7071067811865467, 0.5000000000000027, 0.8660254037844382, 0.7071067811865406}},
    {{1.000000000000008, 1.000000000000008, 1.0000000000000042}, {1.000000000000004, 0.5000000000000091, 0.9999999999999998, 1.0000000000000027, 0.5000000000000084, 1.5000000000000118, 1.0000000000000129, 1.5000000000000098, 0.500000000000004, 0.5000000000000111, 0.5000000000000093, 1.500000000000002, 0.5000000000000064, 1.0000000000000118, 0.5000000000000038}, {0, 4, 2, 1, 3}, {0.4999999999999989, 0.7071067811865526, 0.7071067811865489, 0.8660254037844348, 0.7071067811865489}},
    {{1.0000000000000113, 1.0000000000000069, 1.0000000000000022}, {0.5000000000000107, 0.5000000000000004, 0.5000000000000044, 1.5000000000000135, 0.9999999999999976, 1.5000000000000095, 1.5000000000000193, 1.000000000000002, 0.9999999999999971, 1.0000000000000173, 1.4999999999999982, 0.9999999999999991, 1.500000000000016, 0.500000000000002, 1.0000000000000089}, {3, 2, 4, 1, 0}, {0.8660254037844415, 0.7071067811865542, 0.500000000000008, 0.49999999999999134, 0.7071067811865542}},
    {{1.0000000000000016, 1.000000000000005, 1.0000000000000033}, {0.4999999999999971, 1.4999999999999976, 1.5000000000000075, 1.5000000000000018, 1.5000000000000007, 1.5000000000000087, 1.0000000000000093, 1.500000000000012, 1.000000000000006, 1.5000000000000029, 1.0000000000000007, 1.0000000000000127, 0.5000000000000093, 0.9999999999999969, 1.4999999999999964}, {3, 2, 4, 1, 0}, {0.8660254037844393, 0.8660254037844393, 0.5000000000000069, 0.5000000000000013, 0.7071067811865371}},
    {{1.0000000000000107, 1.000000000000009, 1.0000000000000056}, {1.0000000000000024, 0.5000000000000049, 1.0000000000000133, 1.0000000000000044, 1.5000000000000133, 1.0000000000000082, 0.5000000000000107, 1.0000000000000155, 0.5000000000000051, 0.5000000000000109, 1.0000000000000115, 1.500000000000011, 1.5000000000000142, 0.5000000000000031, 1.5000000000000064}, {1, 0, 2, 3, 4}, {0.5000000000000042, 0.5000000000000042, 0.7071067811865478, 0.7071067811865513, 0.8660254037844447}},
    {{0.1, 0.7}, {0.4, 1.1, -0.30000000000000004, 1.0, 0.1, 0.19999999999999996}, {2, 1, 0}, {0.5000000000000001, 0.5000000000000001, 0.5}},
};
const ContrastGolden contrast_goldens[] = {
    {{1e+16, 1.0000000000000002e+16, 1.0000000000000004e+16, 1.0000000000000006e+16, 1.0000000000000008e+16}, {1, 0, 3, 2, 3}, 0.8},
    {{0.1, 0.2, 0.30000000000000004, 0.7, 0.9}, {1, 2, 1, 4, 3}, 0.9067796610169491},
    {{1.152921504606847e+18, 0.125, -0.0625, 1.0}, {2, 3, 1, 0}, 1.9877039828900914e-19},
};

bool ulp_close(double x, double y, int ulps) {
    return std::isfinite(x) && std::abs(x - y) <= ulps * 0x1p-52 * std::max(std::abs(y), 0x1p-1022);
}

// Summary (ABI 2.18) inside its magnitude bound: exact rational oracle (mean,
// quartiles within 1-2 ulp, SD within 1e-15) for any spread, power-of-two
// equivariance, exact offsets and both magnitude boundaries.
void summary_checks(n4m_context_t* ctx) {
    const auto run = [&](std::vector<double> v, double* out) {
        return n4m_preflight_summary_compute(ctx, v.data(), nullptr,
                                             static_cast<int64_t>(v.size()), out);
    };
    double out[8];
    for (const auto& g : summary_goldens) {
        REQUIRE(run(g.x, out) == N4M_OK);
        REQUIRE(ulp_close(out[3], g.want[0], 1) && rel(out[4], g.want[1], 1e-15));
        for (int c = 0; c < 3; ++c) REQUIRE(ulp_close(out[5 + c], g.want[static_cast<size_t>(2 + c)], 2));
    }
    const std::vector<double> base = {3, 1, 4, 1, 5, 9, 2, 6};
    double ref[8];
    REQUIRE(run(base, ref) == N4M_OK);
    for (int k : sweep) {
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        REQUIRE(run(x, out) == N4M_OK);
        for (int c = 1; c < 8; ++c) REQUIRE(out[c] == std::ldexp(ref[c], k));
    }
    for (int k : {-257, 250}) {  // 1 2^-257 and 9 2^250 > 2^253: 9 2^250 > 2^256? 9 2^250 = 1.125 2^253 inside
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, k);
        REQUIRE(run(x, out) == (k == -257 ? N4M_ERR_UNSUPPORTED : N4M_OK));
    }
    {
        std::vector<double> x(base);
        for (double& v : x) v = std::ldexp(v, 254);  // 9 2^254 > 2^256
        REQUIRE(run(x, out) == N4M_ERR_UNSUPPORTED);
    }
    std::vector<double> shifted(base);
    for (double& v : shifted) v = v * p2(148) + p2(200);
    REQUIRE(run(shifted, out) == N4M_OK && rel(out[4], ref[4] * p2(148), 1e-15));
    // No spread bound for the summary (lead arbitration): 1e-20 next to 1 and
    // 2^-256 next to 2^256 stay correct against the rational oracle;
    // magnitudes outside [2^-256, 2^256] (1e-200, 1e200) are refused.
    const SummaryGolden spread[] = {
        {{1e-20, 1.0, 0.5, 1e-20, 2.0}, {0.7, 0.8366600265340756, 1e-20, 0.5, 1.0}},
        {{8.636168555094445e-78, 1.157920892373162e+77, 0.0, -8.636168555094445e-78, 4.820814132776971e+60},
         {2.315841784746324e+76, 5.178379655827216e+76, 0.0, 8.636168555094445e-78, 4.820814132776971e+60}},
        {{1e-20, -1e-20, 1.0, 0.0}, {0.25, 0.5, -2.5e-21, 5e-21, 0.25}},
        {{1.0, 1.0000000000000002, 1e-25, 7.0}, {2.25, 3.2015621187164243, 0.75, 1.0, 2.5}},
    };
    for (const auto& g : spread) {
        REQUIRE(run(g.x, out) == N4M_OK);
        REQUIRE(ulp_close(out[3], g.want[0], 1) && rel(out[4], g.want[1], 1e-15));
        for (int c = 0; c < 3; ++c) REQUIRE(ulp_close(out[5 + c], g.want[static_cast<size_t>(2 + c)], 2));
    }
    REQUIRE(run({1.0, std::nextafter(p2(64), INFINITY)}, out) == N4M_OK);
    REQUIRE(run({1e-200, 1.0}, out) == N4M_ERR_UNSUPPORTED);
    REQUIRE(run({1e200, 1.0}, out) == N4M_ERR_UNSUPPORTED);
}

// Nearest neighbours (ABI 2.18): candidate orders where a naive double
// distance disagrees with the exact one, and exact ties computed with
// different roundings; distances within 1e-15 of the 400-digit references.
void neighbor_checks(n4m_context_t* ctx) {
    for (const auto& g : knn_goldens) {
        std::vector<double> q(g.q), refs(g.refs);
        const auto p = static_cast<int64_t>(q.size());
        const auto m = static_cast<int64_t>(refs.size()) / p;
        auto Q = view(q.data(), 1, p), R = view(refs.data(), m, p);
        std::vector<int64_t> idx(static_cast<size_t>(m));
        std::vector<double> dist(static_cast<size_t>(m));
        REQUIRE(n4m_preflight_nearest_neighbors(ctx, &Q, &R, nullptr, nullptr,
                                                static_cast<int32_t>(m), idx.data(), dist.data()) == N4M_OK);
        REQUIRE(idx == g.order);
        for (size_t a = 0; a < idx.size(); ++a)
            REQUIRE(rel(dist[a], g.dist[static_cast<size_t>(idx[a])], 1e-15));
    }
    // Scale equivariance and domain boundaries.
    for (int k : sweep) {
        std::vector<double> q = {std::ldexp(1.0, k), 0}, refs = {0, 0, std::ldexp(4.0, k), std::ldexp(3.0, k)};
        auto Q = view(q.data(), 1, 2), R = view(refs.data(), 2, 2);
        int64_t idx[2]; double dist[2];
        REQUIRE(n4m_preflight_nearest_neighbors(ctx, &Q, &R, nullptr, nullptr, 2, idx, dist) == N4M_OK);
        REQUIRE(idx[0] == 0 && dist[0] == std::ldexp(1.0, k) && dist[1] == std::ldexp(std::sqrt(18.0), k));
    }
    for (double bad : {p2(-257), std::nextafter(p2(256), INFINITY)}) {
        double q[2] = {bad, 0}, refs[2] = {0, 0};
        auto Q = view(q, 1, 2), R = view(refs, 1, 2);
        int64_t idx[1]; double dist[1];
        REQUIRE(n4m_preflight_nearest_neighbors(ctx, &Q, &R, nullptr, nullptr, 1, idx, dist) ==
                N4M_ERR_UNSUPPORTED);
    }
}

// Local contrast (ABI 2.18), continuous targets: exact rational oracle,
// power-of-two invariance (bitwise) and domain boundaries; category codes are
// labels and are not bounded by the domain.
void contrast_checks(n4m_context_t* ctx) {
    const auto run = [&](std::vector<double> y, std::vector<int64_t> nb, int categorical, double* out) {
        return n4m_preflight_local_contrast(ctx, y.data(), nb.data(), nullptr,
                                            static_cast<int64_t>(y.size()), 1, categorical, out);
    };
    double out[4], ref[4];
    for (const auto& g : contrast_goldens) {
        REQUIRE(run(g.y, g.neighbor, 0, out) == N4M_OK);
        REQUIRE(std::abs(out[0] - g.contrast) <= 1e-15 * std::max(1.0, std::abs(g.contrast)));
    }
    const std::vector<double> base = {3, 1, 4, 1, 5};
    const std::vector<int64_t> nb = {1, 0, 3, 2, 2};
    REQUIRE(run(base, nb, 0, ref) == N4M_OK);
    for (int k : sweep) {
        std::vector<double> y(base);
        for (double& v : y) v = std::ldexp(v, k);
        REQUIRE(run(y, nb, 0, out) == N4M_OK && out[0] == ref[0]);
    }
    REQUIRE(run({1, 0, p2(-257), 1, 2}, nb, 0, out) == N4M_ERR_UNSUPPORTED);
    REQUIRE(run({1, 0, std::nextafter(p2(64), INFINITY), 1, 2}, nb, 0, out) == N4M_ERR_UNSUPPORTED);
    REQUIRE(run({1, 0, p2(64), 1, 2}, nb, 0, out) == N4M_OK);
    REQUIRE(run({1, 0, 9007199254740991.0, 1, 2}, nb, 1, out) == N4M_OK);
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
        summary_checks(ctx);
        neighbor_checks(ctx);
        contrast_checks(ctx);
        n4m_context_destroy(ctx);
        std::puts("preflight robustness sweeps: ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
