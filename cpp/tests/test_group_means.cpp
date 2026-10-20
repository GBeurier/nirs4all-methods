// SPDX-License-Identifier: CECILL-2.1
// n4m_stats_group_means and n4m_stats_regression_scores: exact dyadic cases,
// magnitude sweeps and offsets inside the declared domain D, both sides of
// D's boundaries, column-major strides and argument refusals.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " @" + std::to_string(__LINE__)); } while (false)

namespace {

n4m_status_t means(n4m_context_t* ctx, std::vector<double>& x, int64_t n, int64_t k,
                   const std::vector<int64_t>& codes, int64_t g, std::vector<double>& out,
                   std::vector<int64_t>& counts) {
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x.data(), n, k, N4M_DTYPE_F64) == N4M_OK);
    out.assign(static_cast<size_t>(g * k), 0.0);
    counts.assign(static_cast<size_t>(g), 0);
    return n4m_stats_group_means(ctx, &v, codes.data(), g, out.data(), counts.data());
}

n4m_status_t scores(n4m_context_t* ctx, const std::vector<double>& y,
                    const std::vector<double>& p, double* out) {
    return n4m_stats_regression_scores(ctx, y.data(), p.data(), static_cast<int64_t>(y.size()), out);
}

void group_means_exact(n4m_context_t* ctx) {
    // Rows: group 1 {1,2},{3,6}; group 0 {10,-4}; group 2 {0,0},{1,1},{2,8}.
    std::vector<double> x = {1, 2, 3, 6, 10, -4, 0, 0, 1, 1, 2, 8};
    const std::vector<int64_t> codes = {1, 1, 0, 2, 2, 2};
    std::vector<double> m;
    std::vector<int64_t> c;
    REQUIRE(means(ctx, x, 6, 2, codes, 3, m, c) == N4M_OK);
    const double want[6] = {10, -4, 2, 4, 1, 3};
    for (int i = 0; i < 6; ++i) REQUIRE(m[static_cast<size_t>(i)] == want[i]);
    REQUIRE(c[0] == 1 && c[1] == 2 && c[2] == 3);
    // Same matrix stored column-major through strides.
    std::vector<double> col(12);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 2; ++j) col[static_cast<size_t>(j * 6 + i)] = x[static_cast<size_t>(i * 2 + j)];
    n4m_matrix_view_t cv{col.data(), 6, 2, 1, 6, N4M_DTYPE_F64, 0};
    std::vector<double> cm(6);
    REQUIRE(n4m_stats_group_means(ctx, &cv, codes.data(), 3, cm.data(), c.data()) == N4M_OK);
    REQUIRE(cm == m);
}

void group_means_sweep(n4m_context_t* ctx) {
    // Scale equivariance on dyadic data (exact) for 2^k inside D, and an
    // offset where naive left-to-right sums lose the small parts:
    // mean(2^52 + 1, 2^52 + 2, 2^52 + 4) = 2^52 + 7/3 (rounded once).
    for (int k : {-200, -100, 0, 100, 200}) {
        std::vector<double> x = {std::ldexp(3.0, k), std::ldexp(5.0, k), std::ldexp(1.5, k)};
        std::vector<double> m;
        std::vector<int64_t> c;
        REQUIRE(means(ctx, x, 3, 1, {0, 0, 0}, 1, m, c) == N4M_OK);
        REQUIRE(m[0] == std::ldexp(9.5 / 3.0, k));
    }
    const double big = std::ldexp(1.0, 52);
    std::vector<double> x = {big + 1, big + 2, big + 4, 1.0, -1.0};
    std::vector<double> m;
    std::vector<int64_t> c;
    REQUIRE(means(ctx, x, 5, 1, {0, 0, 0, 1, 1}, 2, m, c) == N4M_OK);
    REQUIRE(m[0] == big + 2.0);  // 2^52 + 7/3 rounds to the nearest integer
    REQUIRE(m[1] == 0.0);        // exact zero, allowed
}

void group_means_domain(n4m_context_t* ctx) {
    std::vector<double> m;
    std::vector<int64_t> c;
    const auto run = [&](std::vector<double> x) { return means(ctx, x, static_cast<int64_t>(x.size()), 1, std::vector<int64_t>(x.size(), 0), 1, m, c); };
    // Inside D at its magnitude boundaries; a mean is a summary, so a wide
    // spread is accepted and the dyadic mean stays exact.
    REQUIRE(run({std::ldexp(1.0, 256), std::ldexp(1.0, 192)}) == N4M_OK);
    REQUIRE(run({std::ldexp(1.0, -256), std::ldexp(1.0, -192)}) == N4M_OK);
    REQUIRE(run({1.0, std::ldexp(1.0, 100), 0.0}) == N4M_OK);
    REQUIRE(m[0] == (1.0 + std::ldexp(1.0, 100)) / 3.0);
    // Outside D: magnitude.
    REQUIRE(run({std::ldexp(1.0, 257)}) == N4M_ERR_UNSUPPORTED);
    REQUIRE(run({std::ldexp(1.0, -257)}) == N4M_ERR_UNSUPPORTED);
    REQUIRE(run({std::numeric_limits<double>::max()}) == N4M_ERR_UNSUPPORTED);
    const char* message = n4m_context_last_error(ctx);
    REQUIRE(message && std::string(message).find("supported numerical domain") != std::string::npos);
    REQUIRE(run({1.0, std::numeric_limits<double>::quiet_NaN()}) == N4M_ERR_INVALID_ARGUMENT);
}

void group_means_refusals(n4m_context_t* ctx) {
    std::vector<double> x = {1, 2, 3};
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x.data(), 3, 1, N4M_DTYPE_F64) == N4M_OK);
    double out[4];
    int64_t counts[4];
    const int64_t gap[] = {0, 0, 2};
    const int64_t outside[] = {0, 1, 3};
    REQUIRE(n4m_stats_group_means(ctx, &v, gap, 3, out, counts) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_group_means(ctx, &v, outside, 3, out, counts) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_group_means(ctx, &v, gap, 4, out, counts) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_group_means(ctx, &v, nullptr, 2, out, counts) == N4M_ERR_NULL_POINTER);
}

void regression_exact(n4m_context_t* ctx) {
    // y = (1, 2, 3, 4), prediction (1, 3, 3, 2): SSE = 5, SAE = 3, SST = 5.
    double out[4];
    REQUIRE(scores(ctx, {1, 2, 3, 4}, {1, 3, 3, 2}, out) == N4M_OK);
    REQUIRE(out[0] == std::sqrt(5.0 / 4.0));
    REQUIRE(out[1] == 0.75);
    REQUIRE(out[2] == 0.0 && out[3] == 0.0);
    // Perfect fit and constant target (R^2 undefined, documented NaN).
    REQUIRE(scores(ctx, {1, 2, 3}, {1, 2, 3}, out) == N4M_OK);
    REQUIRE(out[0] == 0.0 && out[1] == 0.0 && out[2] == 1.0);
    REQUIRE(scores(ctx, {2, 2}, {1, 3}, out) == N4M_OK);
    REQUIRE(out[0] == 1.0 && std::isnan(out[2]));
}

void regression_sweep(n4m_context_t* ctx) {
    // Scale equivariance (RMSE, MAE scale by 2^k, R^2 invariant), exact on
    // dyadic data, and translation invariance under a 2^52 offset where a
    // naive mean of y loses the deviations.
    const std::vector<double> y = {1, 2, 3, 4}, p = {1, 3, 3, 2};
    for (int k : {-200, -100, 0, 100, 200}) {
        std::vector<double> ys, ps;
        for (size_t i = 0; i < y.size(); ++i) {
            ys.push_back(std::ldexp(y[i], k));
            ps.push_back(std::ldexp(p[i], k));
        }
        double out[4];
        REQUIRE(scores(ctx, ys, ps, out) == N4M_OK);
        REQUIRE(out[0] == std::ldexp(std::sqrt(1.25), k));
        REQUIRE(out[1] == std::ldexp(0.75, k));
        REQUIRE(out[2] == 0.0);
    }
    const double big = std::ldexp(1.0, 52);
    std::vector<double> ys, ps;
    for (size_t i = 0; i < y.size(); ++i) {
        ys.push_back(big + y[i]);
        ps.push_back(big + p[i]);
    }
    double out[4];
    REQUIRE(scores(ctx, ys, ps, out) == N4M_OK);
    REQUIRE(out[0] == std::sqrt(1.25) && out[1] == 0.75 && out[2] == 0.0);
}

void regression_domain(n4m_context_t* ctx) {
    double out[4];
    REQUIRE(scores(ctx, {std::ldexp(1.0, 256), std::ldexp(1.0, 255)}, {0.0, std::ldexp(1.0, 255)}, out) == N4M_OK);
    REQUIRE(scores(ctx, {std::ldexp(1.0, 257), 1.0}, {1.0, 1.0}, out) == N4M_ERR_UNSUPPORTED);
    REQUIRE(scores(ctx, {1.0, 2.0}, {std::ldexp(1.0, -300), 1.0}, out) == N4M_ERR_UNSUPPORTED);
    // Spread limit 2^32 (rule 15), both sides.
    REQUIRE(scores(ctx, {1.0, std::ldexp(1.0, 32)}, {1.0, 1.0}, out) == N4M_OK);
    REQUIRE(scores(ctx, {1.0, std::ldexp(1.0, 33)}, {1.0, 1.0}, out) == N4M_ERR_UNSUPPORTED);
    REQUIRE(scores(ctx, {1.0, 2.0}, {1.0, std::ldexp(1.0, 33)}, out) == N4M_ERR_UNSUPPORTED);
    // Inside D, a deviation of one ulp next to 2^65 keeps R^2 exact:
    // y = (2^65, 2^65 + 2^13), prediction (2^65 + 2^13, 2^65): SSE = 2 * 2^26,
    // SST = 2^25, R^2 = -3.
    REQUIRE(scores(ctx, {std::ldexp(1.0, 65), std::ldexp(1.0, 65) + std::ldexp(1.0, 13)},
                   {std::ldexp(1.0, 65) + std::ldexp(1.0, 13), std::ldexp(1.0, 65)}, out) == N4M_OK);
    REQUIRE(out[2] == -3.0 && out[3] == 0.0);
    // R^2 beyond the double range inside D (Astra counter-example):
    // y = (2^-255, 2^-254), prediction (2^256, 2^256). Rational oracle:
    // SST = 2^-511 exactly; SSE = (2^256 - 2^-255)^2 + (2^256 - 2^-254)^2
    // = 2^513 (1 - 3 * 2^-512 + ...), so SSE/SST = 2^1024 (1 - ~2^-510) and
    // R^2 = 1 - SSE/SST rounds to -2^1024 = -0.5 * 2^1025.
    REQUIRE(scores(ctx, {std::ldexp(1.0, -255), std::ldexp(1.0, -254)},
                   {std::ldexp(1.0, 256), std::ldexp(1.0, 256)}, out) == N4M_OK);
    REQUIRE(std::isfinite(out[0]) && out[0] == std::ldexp(1.0, 256));
    REQUIRE(out[2] == -0.5 && out[3] == 1025.0);
    REQUIRE(scores(ctx, {1.0, std::numeric_limits<double>::infinity()}, {1.0, 1.0}, out) ==
            N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_regression_scores(ctx, nullptr, nullptr, 1, out) == N4M_ERR_NULL_POINTER);
}

}  // namespace

int main() {
    n4m_context_t* ctx = nullptr;
    if (n4m_context_create(&ctx) != N4M_OK) return 1;
    int failures = 0;
    for (auto [name, fn] : {std::pair{"group_means_exact", &group_means_exact},
                            std::pair{"group_means_sweep", &group_means_sweep},
                            std::pair{"group_means_domain", &group_means_domain},
                            std::pair{"group_means_refusals", &group_means_refusals},
                            std::pair{"regression_exact", &regression_exact},
                            std::pair{"regression_sweep", &regression_sweep},
                            std::pair{"regression_domain", &regression_domain}}) {
        try {
            fn(ctx);
            std::printf("ok %s\n", name);
        } catch (const std::exception& e) {
            std::printf("FAILED %s: %s\n", name, e.what());
            ++failures;
        }
    }
    n4m_context_destroy(ctx);
    return failures == 0 ? 0 : 1;
}
