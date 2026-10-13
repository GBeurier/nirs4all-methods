// SPDX-License-Identifier: CECILL-2.1
// One-way random-effects REML and confidence intervals against independent
// references recorded as test-only constants:
//  * "dense": REML maximised on the full matrices V = s2b ZZ' + s2w I with
//    the Harville (1977) score equations, NumPy 2.5.4 / SciPy 1.18.1;
//  * "statsmodels": MixedLM("y ~ 1", groups).fit(reml=True), statsmodels
//    0.15.0 (its default optimiser stops near 1e-6 relative);
//  * balanced closed forms (REML = ANOVA when MSB > MSW; REML s2b = 0 and
//    s2w = sample variance otherwise);
//  * SciPy chi2.ppf / f.ppf for the exact intervals; the bootstrap reference
//    redraws numpy.random.default_rng(seed).integers(0, U, (B, U)), refits
//    each replicate with the dense REML and takes numpy.quantile(linear).
// Generator: parity/generators/variance_components_v1.py (Python 3.14.3).
#include "n4m/metrics.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
static bool rel(double x, double y, double tol) {
    return std::abs(x - y) <= tol * std::max(1.0, std::abs(y));
}
static bool relative(double x, double y, double tol) { return std::abs(x - y) <= tol * std::abs(y); }

struct Reml { double out[8]; int32_t status; };

static n4m_matrix_view_t column(std::vector<double>& y) {
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, y.data(), static_cast<int64_t>(y.size()), 1,
                                          N4M_DTYPE_F64) == N4M_OK);
    return v;
}
static Reml reml(n4m_context_t* ctx, std::vector<double> y, const std::vector<int64_t>& g) {
    Reml r{};
    auto v = column(y);
    REQUIRE(n4m_stats_variance_components_reml(ctx, &v, g.data(), nullptr, r.out, &r.status) == N4M_OK);
    return r;
}
static std::vector<double> intervals(n4m_context_t* ctx, std::vector<double> y,
                                     const std::vector<int64_t>& g, double confidence,
                                     const int64_t* clusters = nullptr, int32_t b = 0,
                                     uint64_t seed = 0, int64_t* valid = nullptr) {
    std::vector<double> out(8);
    int64_t n_valid = -1;
    auto v = column(y);
    REQUIRE(n4m_stats_variance_components_intervals(ctx, &v, g.data(), clusters, nullptr,
                                                    confidence, b, seed, out.data(),
                                                    &n_valid) == N4M_OK);
    if (valid) *valid = n_valid;
    return out;
}

int main() {
    try {
        n4m_context_t* ctx = nullptr;
        REQUIRE(n4m_context_create(&ctx) == N4M_OK);

        // A: unbalanced sizes 2,3,4,1,2 with one singleton group.
        const std::vector<int64_t> ga{0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4};
        const std::vector<double> ya{1.0, 1.4, 2.2, 2.0, 2.9, 0.3, 0.9, 0.4, 0.8, 1.7, 3.1, 2.5};
        auto a = reml(ctx, ya, ga);
        REQUIRE(a.status == 0 && a.out[4] == 5 && a.out[5] == 12 && a.out[6] == 1);
        REQUIRE(relative(a.out[0], 0.7446105326226952, 1e-10));
        REQUIRE(relative(a.out[1], 0.13630617544724544, 1e-10));
        REQUIRE(relative(a.out[2], 0.8452678054592837, 1e-10));
        REQUIRE(rel(a.out[3], -10.963218720046495, 1e-12));
        REQUIRE(relative(a.out[0], 0.744610306998525, 1e-6));    // statsmodels
        REQUIRE(relative(a.out[1], 0.13630619603723484, 1e-6));  // statsmodels
        REQUIRE(rel(a.out[3], -10.963218720046608, 1e-12));      // statsmodels llf
        auto ia = intervals(ctx, ya, ga, 0.95);
        REQUIRE(relative(ia[0], 0.24570003389152809, 1e-10));
        REQUIRE(relative(ia[1], 0.7563308737449301, 1e-10));
        REQUIRE(std::isnan(ia[2]) && std::isnan(ia[3]) && std::isnan(ia[4]));
        ia = intervals(ctx, ya, ga, 0.90);
        REQUIRE(relative(ia[0], 0.26214130662257856, 1e-10));
        REQUIRE(relative(ia[1], 0.6678421962737774, 1e-10));

        // Cluster bootstrap of A: groups {0,1} share cluster 7. Reference
        // draws begin [[2,0,3,1],[0,3,2,2],[3,1,3,1]] over units (3,5,7,9).
        const std::vector<int64_t> ca{7, 7, 7, 7, 7, 3, 3, 3, 3, 9, 5, 5};
        int64_t valid = 0;
        ia = intervals(ctx, ya, ga, 0.95, ca.data(), 40, 12345, &valid);
        REQUIRE(valid == 40);
        REQUIRE(relative(ia[4], 0.3094525017261184, 1e-10));
        REQUIRE(relative(ia[5], 0.42426406871192857, 1e-10));
        REQUIRE(ia[6] == 0.0);  // boundary replicates stay at ICC 0
        REQUIRE(relative(ia[7], 0.9233717840878614, 1e-8));
        std::vector<int64_t> bad = ca;
        bad[2] = 9;  // group 1 would span clusters 7 and 9
        std::vector<double> out(8);
        std::vector<double> ya_copy = ya;
        auto v = column(ya_copy);
        REQUIRE(n4m_stats_variance_components_intervals(ctx, &v, ga.data(), bad.data(), nullptr,
                                                        0.95, 10, 1, out.data(), &valid) ==
                N4M_ERR_INVALID_ARGUMENT);

        // B: balanced 4 x 3, MSB = 547/300, MSW = 17/600: REML = ANOVA.
        const std::vector<int64_t> gb{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
        const std::vector<double> yb{1.0, 1.2, 0.9, 2.0, 2.3, 2.1, 0.2, 0.1, 0.5, 1.4, 1.6, 1.3};
        const double msw = 17.0 / 600.0, msb = 547.0 / 300.0;
        auto b = reml(ctx, yb, gb);
        REQUIRE(b.status == 0);
        REQUIRE(relative(b.out[0], (msb - msw) / 3.0, 1e-11));
        REQUIRE(relative(b.out[1], msw, 1e-11));
        REQUIRE(relative(b.out[2], (msb - msw) / (msb + 2.0 * msw), 1e-11));
        REQUIRE(rel(b.out[3], -3.496911437154305, 1e-11));
        auto ib = intervals(ctx, yb, gb, 0.95);
        REQUIRE(relative(ib[0], 0.11369636090977235, 1e-10));
        REQUIRE(relative(ib[1], 0.32247229155084495, 1e-10));
        REQUIRE(relative(ib[2], 0.7838941809128407, 1e-10));
        REQUIRE(relative(ib[3], 0.9968006295172774, 1e-10));

        // C: balanced, negative ANOVA ICC -> REML boundary s2b = 0 with s2w
        // the sample variance; the exact F interval is not truncated.
        const std::vector<int64_t> gc{0, 0, 1, 1, 2, 2};
        const std::vector<double> yc{0.0, 2.0, 0.1, 1.9, -0.1, 2.2};
        auto c = reml(ctx, yc, gc);
        REQUIRE(c.status == 2 && c.out[0] == 0.0 && c.out[2] == 0.0 && c.out[7] == 0);
        REQUIRE(relative(c.out[1], 1.2536666666666667, 1e-12));
        REQUIRE(rel(c.out[3], -8.555753877687614, 1e-12));
        auto ic = intervals(ctx, yc, gc, 0.95);
        REQUIRE(rel(ic[2], -0.9999005186230587, 1e-10));
        REQUIRE(rel(ic[3], -0.9393800643512333, 1e-10));

        // D: unbalanced NIRS-like band, between SD 1 and repeat SD 1e-3.
        const std::vector<int64_t> gd{0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4,
                                      5, 5, 5, 5, 5, 6, 6, 7, 8, 8, 8, 9, 9};
        const std::vector<double> yd{
            1.064243488713, 1.064250411816, 1.065793345449, 1.260062263727, 1.261329984465,
            0.170161222993, 0.17054620527, 0.168837388802, 0.167661328384, 1.321442205573,
            0.344533091463, 0.345563606375, 0.345262063448, 0.647099301452, 0.648513834033,
            0.648078877011, 0.648273206332, 0.647010175944, 0.783413220805, 0.783957142208,
            -0.619075287474, 1.628492983907, 1.627344618573, 1.628858545073, 1.322475126154,
            1.323941555553};
        auto d = reml(ctx, yd, gd);
        REQUIRE(d.status == 0);
        REQUIRE(relative(d.out[0], 0.46169941459038055, 1e-8));
        REQUIRE(relative(d.out[1], 7.844000990133166e-07, 1e-8));
        REQUIRE(relative(1.0 - d.out[2], 1.0 - 0.999998301061813, 1e-6));
        REQUIRE(rel(d.out[3], 75.13429340814226, 1e-10));
        REQUIRE(relative(d.out[0], 0.4599569441375687, 5e-3));  // statsmodels

        // E: unbalanced, REML at the boundary s2b = 0 (score -12.3 < 0 there).
        const std::vector<int64_t> ge{0, 0, 0, 1, 1, 2, 2, 2, 2, 3};
        const std::vector<double> ye{1.0, 2.0, 3.0, 2.0, 2.1, 1.5, 2.5, 1.9, 2.2, 2.05};
        auto e = reml(ctx, ye, ge);
        REQUIRE(e.status == 2 && e.out[0] == 0.0 && e.out[2] == 0.0);
        REQUIRE(relative(e.out[1], 0.28402777778428895, 1e-10));
        REQUIRE(rel(e.out[3], -8.257664780966449, 1e-12));

        // Large balanced design, df = 2000 and F(999, 2000): y_ij = i/100 + (j-1)/2.
        std::vector<int64_t> gl;
        std::vector<double> yl;
        for (int64_t i = 0; i < 1000; ++i)
            for (int j = 0; j < 3; ++j) {
                gl.push_back(i);
                yl.push_back(0.01 * static_cast<double>(i) + 0.5 * (j - 1));
            }
        auto l = reml(ctx, yl, gl);
        REQUIRE(l.status == 0 && relative(l.out[2], 0.970617042115573, 1e-10));
        auto il = intervals(ctx, yl, gl, 0.95);
        REQUIRE(relative(il[0], 0.4849751429481101, 1e-10));
        REQUIRE(relative(il[1], 0.5159925697716798, 1e-10));
        REQUIRE(relative(il[2], 0.9673880863930334, 1e-10));
        REQUIRE(relative(il[3], 0.9735827791418388, 1e-10));

        // One within df: SSW = 2, chi2_{0.025,1} = 9.8207e-4.
        auto one = intervals(ctx, {1.0, 3.0, 10.0}, {0, 0, 1}, 0.95);
        REQUIRE(relative(one[0], 0.6309502281565684, 1e-10));
        REQUIRE(relative(one[1], 45.12778012975313, 1e-9));

        // Degenerate designs.
        REQUIRE(reml(ctx, {1.0, 2.0, 3.0}, {0, 1, 2}).status == 1);  // all singletons
        REQUIRE(reml(ctx, {1.0, 2.0}, {0, 0}).status == 1);          // one group
        REQUIRE(reml(ctx, {4.0, 4.0, 4.0, 4.0}, {0, 0, 1, 1}).status == 3);
        auto w = reml(ctx, {1.0, 1.0, 3.0, 3.0, 8.0}, {0, 0, 1, 1, 2});
        REQUIRE(w.status == 4 && w.out[1] == 0.0 && w.out[2] == 1.0 && std::isnan(w.out[3]));
        REQUIRE(relative(w.out[0], 13.0, 1e-12));  // sample variance of means 1, 3, 8

        // Two features: column-major storage, and a mask that unbalances
        // feature 1 by hiding one value of group 1 (it then equals case A
        // without its fifth value).
        const int64_t n = 12;
        std::vector<double> xc(2 * n);
        std::vector<uint8_t> mask(2 * n, 1);
        for (int64_t i = 0; i < n; ++i) {
            xc[static_cast<size_t>(i)] = yb[static_cast<size_t>(i)];
            xc[static_cast<size_t>(n + i)] = ya[static_cast<size_t>(i)];
        }
        mask[4 * 2 + 1] = 0;
        n4m_matrix_view_t cm{};
        REQUIRE(n4m_matrix_view_init_colmajor(&cm, xc.data(), n, 2, N4M_DTYPE_F64) == N4M_OK);
        double two[16];
        int32_t st[2];
        REQUIRE(n4m_stats_variance_components_reml(ctx, &cm, ga.data(), mask.data(), two, st) == N4M_OK);
        std::vector<double> ya_masked = ya;
        ya_masked.erase(ya_masked.begin() + 4);
        std::vector<int64_t> ga_masked = ga;
        ga_masked.erase(ga_masked.begin() + 4);
        auto am = reml(ctx, ya_masked, ga_masked);
        auto bm = reml(ctx, yb, ga);
        for (int k = 0; k < 8; ++k) {
            REQUIRE(two[k] == bm.out[k] || (std::isnan(two[k]) && std::isnan(bm.out[k])));
            REQUIRE(two[8 + k] == am.out[k] || (std::isnan(two[8 + k]) && std::isnan(am.out[k])));
        }
        REQUIRE(st[0] == bm.status && st[1] == am.status && two[13] == 11);

        // Invalid arguments.
        std::vector<double> bad_x{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0, 3.0};
        auto bv = column(bad_x);
        const std::vector<int64_t> g4{0, 0, 1, 1};
        REQUIRE(n4m_stats_variance_components_reml(ctx, &bv, g4.data(), nullptr, out.data(), st) ==
                N4M_ERR_INVALID_ARGUMENT);
        const uint8_t hide[4] = {1, 0, 1, 1};
        REQUIRE(n4m_stats_variance_components_reml(ctx, &bv, g4.data(), hide, out.data(), st) == N4M_OK);
        REQUIRE(n4m_stats_variance_components_reml(ctx, &bv, nullptr, nullptr, out.data(), st) ==
                N4M_ERR_NULL_POINTER);
        REQUIRE(n4m_stats_variance_components_intervals(ctx, &bv, g4.data(), nullptr, hide, 1.0, 0,
                                                        0, out.data(), &valid) ==
                N4M_ERR_INVALID_ARGUMENT);
        std::vector<double> wide(2 * 100, 1.0);
        n4m_matrix_view_t wv{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&wv, wide.data(), 2, 100, N4M_DTYPE_F64) == N4M_OK);
        std::vector<double> wide_out(800);
        std::vector<int64_t> wide_valid(100);
        const int64_t g2[2] = {0, 0};
        REQUIRE(n4m_stats_variance_components_intervals(ctx, &wv, g2, nullptr, nullptr, 0.95,
                                                        100000, 0, wide_out.data(),
                                                        wide_valid.data()) == N4M_ERR_UNSUPPORTED);
        n4m_context_destroy(ctx);
        std::puts("variance components: ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
}
