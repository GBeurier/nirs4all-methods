// SPDX-License-Identifier: CECILL-2.1
// Magnitude, translation and constant-column property tests for the
// clustering kernels (HDBSCAN, linear sum assignment, multivariate normal and
// adjusted Rand index). Every scaled run must either reproduce the unscaled
// result under the exact invariance of the statistic, or return an explicit
// N4M_ERR_UNSUPPORTED without touching the caller's buffers.
#include <algorithm>
#include <array>
#include <cfloat>
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

constexpr int64_t sentinel_label = -777;
const int sweep[] = {-1074 + 60, -1022, -500, -100, 0, 100, 500, 1000};

struct Fit {
    n4m_status_t status = N4M_OK;
    std::vector<int64_t> labels, parent;
    std::vector<double> membership, outlier, persistence, birth, death, stability;
};

Fit hdbscan(n4m_context_t* ctx, std::vector<double> x, int64_t rows, int64_t cols) {
    n4m_matrix_view_t X{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&X, x.data(), rows, cols, N4M_DTYPE_F64) == N4M_OK);
    const auto n = static_cast<size_t>(rows);
    Fit f;
    f.labels.assign(n, sentinel_label);
    f.parent.assign(n, 0);
    std::vector<int64_t> size(n), label(n);
    for (auto* v : {&f.membership, &f.outlier, &f.persistence, &f.birth, &f.death, &f.stability})
        v->assign(n, -1.0);
    int64_t clusters = 0, nodes = 0;
    f.status = n4m_cluster_hdbscan(ctx, &X, 4, 3, 0, 0, f.labels.data(), f.membership.data(),
                                   f.outlier.data(), &clusters, f.persistence.data(), &nodes,
                                   f.parent.data(), size.data(), label.data(), f.birth.data(),
                                   f.death.data(), f.stability.data());
    if (f.status == N4M_OK) {
        f.persistence.resize(static_cast<size_t>(clusters));
        for (auto* v : {&f.birth, &f.death, &f.stability}) v->resize(static_cast<size_t>(nodes));
        f.parent.resize(static_cast<size_t>(nodes));
    }
    return f;
}

bool rel_equal(double got, double want) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want)) return got == want;
    return std::fabs(got - want) <= 1e-12 * std::fabs(want);
}

// Two irregular blobs and an outlier, all coordinates in [1, 16].
std::vector<double> blobs() {
    std::vector<double> x;
    for (int i = 0; i < 20; ++i) {
        const double centre = i < 10 ? 2.0 : 10.0;
        x.push_back(centre + 0.9 * std::fabs(std::sin(1.7 * i + 0.3)));
        x.push_back(centre + 0.8 * std::fabs(std::cos(2.3 * i + 0.1)));
    }
    x.push_back(16.0);
    x.push_back(1.0);
    return x;
}

void hdbscan_sweep(n4m_context_t* ctx) {
    const auto base_x = blobs();
    const Fit base = hdbscan(ctx, base_x, 21, 2);
    REQUIRE(base.status == N4M_OK);
    double low = DBL_MAX, high = 0.0;
    for (const auto* v : {&base.birth, &base.death, &base.stability})
        for (const double value : *v)
            if (std::isfinite(value) && value > 0.0) {
                low = std::min(low, value);
                high = std::max(high, value);
            }
    for (const int k : sweep) {
        std::vector<double> x(base_x);
        for (double& v : x) v = std::ldexp(v, k);
        const Fit f = hdbscan(ctx, x, 21, 2);
        // Densities and stabilities scale like 2^-k: representable or refused.
        const double scaled_high = std::ldexp(high, -k), scaled_low = std::ldexp(low, -k);
        const bool representable = std::isfinite(scaled_high) && scaled_low >= DBL_MIN;
        if (!representable) {
            REQUIRE(f.status == N4M_ERR_UNSUPPORTED);
            for (const int64_t l : f.labels) REQUIRE(l == sentinel_label);
            std::printf("hdbscan sweep k=%5d refused (densities not representable)\n", k);
            continue;
        }
        REQUIRE(f.status == N4M_OK);
        REQUIRE(f.labels == base.labels && f.parent == base.parent);
        for (size_t i = 0; i < f.membership.size(); ++i)
            REQUIRE(rel_equal(f.membership[i], base.membership[i]) &&
                    rel_equal(f.outlier[i], base.outlier[i]));
        for (size_t c = 0; c < f.persistence.size(); ++c)
            REQUIRE(rel_equal(f.persistence[c], base.persistence[c]) && f.persistence[c] > 0.0);
        for (size_t j = 0; j < f.birth.size(); ++j) {
            REQUIRE(rel_equal(f.birth[j], std::ldexp(base.birth[j], -k)));
            REQUIRE(rel_equal(f.death[j], std::ldexp(base.death[j], -k)));
            REQUIRE(rel_equal(f.stability[j], std::ldexp(base.stability[j], -k)));
        }
        std::printf("hdbscan sweep k=%5d invariant\n", k);
    }

    // Review 2026-10-13: 1e-308 x [0,1,3,10,11,14] made the stability sums
    // overflow (persistence NaN), and 1e-310 an infinite density for a
    // positive distance (membership 1), both with N4M_OK.
    const std::vector<double> line = {0, 1, 3, 10, 11, 14};
    n4m_matrix_view_t L{};
    for (const double factor : {1e-308, 1e-310}) {
        std::vector<double> x(line);
        for (double& v : x) v *= factor;
        REQUIRE(n4m_matrix_view_init_rowmajor(&L, x.data(), 6, 1, N4M_DTYPE_F64) == N4M_OK);
        std::vector<int64_t> lab(6, sentinel_label), par(6), siz(6), tl(6);
        std::vector<double> mem(6), out(6), per(6), bir(6), dea(6), sta(6);
        int64_t nc = 0, nn = 0;
        const auto st = n4m_cluster_hdbscan(ctx, &L, 2, 2, 0, 0, lab.data(), mem.data(), out.data(),
                                            &nc, per.data(), &nn, par.data(), siz.data(), tl.data(),
                                            bir.data(), dea.data(), sta.data());
        if (st == N4M_OK) {
            for (int64_t c = 0; c < nc; ++c)
                REQUIRE(std::isfinite(per[static_cast<size_t>(c)]) && per[static_cast<size_t>(c)] > 0);
            for (int64_t j = 0; j < nn; ++j) {
                REQUIRE(std::isfinite(dea[static_cast<size_t>(j)]));
                REQUIRE(std::isfinite(sta[static_cast<size_t>(j)]));
            }
        } else {
            REQUIRE(st == N4M_ERR_UNSUPPORTED);
            for (const int64_t l : lab) REQUIRE(l == sentinel_label);
        }
        std::printf("hdbscan line x %g: %s\n", factor, st == N4M_OK ? "ok" : "refused");
    }

    // Translation on an exactly representable grid and a constant column of
    // huge amplitude leave every output bit-identical.
    std::vector<double> grid;
    for (int i = 0; i < 21; ++i) {
        grid.push_back(4.0 * ((i * 7) % 23 + (i < 10 ? 0 : 60)));
        grid.push_back(4.0 * ((i * 11) % 19));
    }
    const Fit g = hdbscan(ctx, grid, 21, 2);
    REQUIRE(g.status == N4M_OK);
    std::vector<double> shifted(grid), wide, huge_grid(grid), huge_shift(grid);
    for (double& v : shifted) v += std::ldexp(1.0, 54);
    for (double& v : huge_grid) v = std::ldexp(v, 944);
    for (double& v : huge_shift) v = std::ldexp(v, 944) + std::ldexp(1.0, 996);
    for (size_t i = 0; i < 21; ++i) {
        wide.push_back(grid[2 * i]);
        wide.push_back(1e300);
        wide.push_back(grid[2 * i + 1]);
    }
    const Fit s = hdbscan(ctx, shifted, 21, 2), w = hdbscan(ctx, wide, 21, 3);
    const Fit h = hdbscan(ctx, huge_grid, 21, 2), hs = hdbscan(ctx, huge_shift, 21, 2);
    for (const Fit* f : {&s, &w}) {
        REQUIRE(f->status == N4M_OK && f->labels == g.labels && f->membership == g.membership &&
                f->stability == g.stability && f->persistence == g.persistence);
    }
    REQUIRE(h.status == N4M_OK && hs.status == N4M_OK);
    REQUIRE(hs.labels == h.labels && hs.membership == h.membership &&
            hs.stability == h.stability);
    std::puts("hdbscan translation and constant column: bit-identical");
}

std::vector<int64_t> assign(n4m_context_t* ctx, std::vector<double> cost, int64_t rows,
                            int64_t cols, int32_t maximize, n4m_status_t expect = N4M_OK) {
    n4m_matrix_view_t C{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&C, cost.data(), rows, cols, N4M_DTYPE_F64) == N4M_OK);
    std::vector<int64_t> r(static_cast<size_t>(std::min(rows, cols))), c(r.size());
    int64_t assigned = -1;
    REQUIRE(n4m_linear_sum_assignment(ctx, &C, maximize, r.data(), c.data(), &assigned) == expect);
    return c;
}

// Independent exact oracle: a finite double is m * 2^q with an integer
// mantissa m, so every sum is an integer multiple of 2^-1074. Positive and
// negative parts accumulate in 2240-bit unsigned integers (70 x 32-bit limbs).
struct Exact {
    std::array<uint32_t, 70> pos{}, neg{};
};

void add_bits(std::array<uint32_t, 70>& acc, uint64_t mantissa, int shift) {
    for (int part = 0; part < 2; ++part) {
        uint64_t chunk = (mantissa >> (32 * part)) & 0xffffffffu;
        if (!chunk) continue;
        const int bit = shift + 32 * part;
        auto limb = static_cast<size_t>(bit / 32);
        const int offset = bit % 32;
        uint64_t carry = chunk << offset;
        while (carry) {
            const uint64_t total = static_cast<uint64_t>(acc[limb]) + (carry & 0xffffffffu);
            acc[limb] = static_cast<uint32_t>(total);
            carry = (carry >> 32) + (total >> 32);
            ++limb;
        }
    }
}

void accumulate(Exact& e, double v) {
    if (v == 0.0) return;
    int exponent = 0;
    const double fraction = std::frexp(std::fabs(v), &exponent);  // |v| = f 2^exponent
    auto mantissa = static_cast<uint64_t>(std::ldexp(fraction, 53));
    int shift = exponent - 53 + 1074;
    for (; shift < 0; ++shift) mantissa >>= 1;  // subnormal: the dropped bits are zero
    add_bits(v > 0 ? e.pos : e.neg, mantissa, shift);
}

int compare_bits(const std::array<uint32_t, 70>& a, const std::array<uint32_t, 70>& b) {
    for (size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

// Sign of (sum x) - (sum y) for exact sums x and y.
int compare_sums(const Exact& x, const Exact& y) {
    Exact left = x, right = y;  // compare x.pos + y.neg with y.pos + x.neg
    for (size_t i = 0; i < 70; ++i) {
        add_bits(left.pos, y.neg[i], static_cast<int>(32 * i));
        add_bits(right.pos, x.neg[i], static_cast<int>(32 * i));
    }
    return compare_bits(left.pos, right.pos);
}

// Exact optimum of a small rows <= cols minimisation by enumeration, and the
// exact cost of the returned columns.
void check_optimal(n4m_context_t* ctx, const std::vector<double>& cost, int64_t rows, int64_t cols) {
    const auto got = assign(ctx, cost, rows, cols, 0);
    std::vector<int64_t> perm(static_cast<size_t>(cols));
    for (int64_t j = 0; j < cols; ++j) perm[static_cast<size_t>(j)] = j;
    auto cost_of = [&](const std::vector<int64_t>& columns) {
        Exact e;
        for (int64_t i = 0; i < rows; ++i)
            accumulate(e, cost[static_cast<size_t>(i * cols + columns[static_cast<size_t>(i)])]);
        return e;
    };
    const Exact returned = cost_of(got);
    bool first = true;
    Exact best;
    do {
        const Exact candidate = cost_of(perm);
        if (first || compare_sums(candidate, best) < 0) best = candidate;
        first = false;
    } while (std::next_permutation(perm.begin(), perm.end()));
    REQUIRE(compare_sums(returned, best) == 0);
}

void assignment_exactness(n4m_context_t* ctx) {
    // The oracle itself: -1e20 + 1 + 1 exceeds -1e20 + 0 + 0, which double
    // arithmetic cannot see.
    Exact two, zero;
    for (const double v : {-1e20, 1.0, 1.0}) accumulate(two, v);
    for (const double v : {-1e20, 0.0, 0.0}) accumulate(zero, v);
    REQUIRE(compare_sums(two, zero) == 1 && compare_sums(zero, two) == -1 &&
            compare_sums(two, two) == 0 && -1e20 + 1.0 + 1.0 == -1e20);
    // Review 2026-10-13 P0: global centring on the minimum turned the lower
    // block into a constant 1e20 and returned [0, 1, 2] (cost -1e20 + 2)
    // instead of [0, 2, 1] (cost -1e20). SciPy's double arithmetic also
    // loses such differences in other matrices; the exact solver does not.
    const std::vector<double> centring = {-1e20, 1e20, 1e20, 1e20, 1, 0, 1e20, 0, 1};
    REQUIRE((assign(ctx, centring, 3, 3, 0) == std::vector<int64_t>{0, 2, 1}));
    check_optimal(ctx, centring, 3, 3);
    // Centring made 1e308 - (-1e308) overflow and declared this infeasible.
    const std::vector<double> extreme = {-1e308, 1e308, -1e308, 1e308};
    const auto columns = assign(ctx, extreme, 2, 2, 0);
    REQUIRE((columns == std::vector<int64_t>{0, 1} || columns == std::vector<int64_t>{1, 0}));
    check_optimal(ctx, extreme, 2, 2);
    // Signed mixtures with exponents spread over [-300, 300] and a few
    // near-DBL_MAX entries, against the exact enumeration.
    uint64_t state = 0x9e3779b97f4a7c15u;
    auto next = [&state]() {
        state = state * 6364136223846793005u + 1442695040888963407u;
        return state >> 11;
    };
    for (int trial = 0; trial < 600; ++trial) {
        const auto rows = static_cast<int64_t>(2 + next() % 3);
        const auto cols = rows + static_cast<int64_t>(next() % 3);
        const int spread = trial % 3 == 0 ? 60 : (trial % 3 == 1 ? 300 : 1000);
        std::vector<double> cost(static_cast<size_t>(rows * cols));
        for (double& v : cost) {
            const double mantissa = 1.0 + static_cast<double>(next() % 4096) / 4096.0;
            int exponent = static_cast<int>(next() % static_cast<uint64_t>(2 * spread)) - spread;
            exponent = std::min(exponent, 1022);
            v = (next() % 2 ? -1.0 : 1.0) * std::ldexp(mantissa, exponent);
        }
        check_optimal(ctx, cost, rows, cols);
    }
    std::puts("assignment: exact optimum on mixed signs and exponent spreads");
}

void assignment_sweep(n4m_context_t* ctx) {
    std::vector<double> base(35);
    for (size_t i = 0; i < base.size(); ++i)
        base[i] = 1.0 + std::fmod(static_cast<double>(i) * 7.31, 9.0);
    for (const int32_t maximize : {0, 1}) {
        const auto want = assign(ctx, base, 5, 7, maximize);
        for (const int k : sweep) {
            std::vector<double> x(base);
            for (double& v : x) v = std::ldexp(v, k);
            REQUIRE(assign(ctx, x, 5, 7, maximize) == want);
        }
        // Costs just below DBL_MAX: the dual sums no longer overflow.
        std::vector<double> top(base);
        for (double& v : top) v = std::ldexp(v / 16.0, 1023);
        REQUIRE(assign(ctx, top, 5, 7, maximize) == want);
        // Exact translation of an integer grid.
        std::vector<double> grid(base), shifted(base);
        for (size_t i = 0; i < base.size(); ++i) {
            grid[i] = 256.0 * std::floor(base[i] * 1000.0);
            shifted[i] = grid[i] + std::ldexp(1.0, 60);
        }
        REQUIRE(assign(ctx, shifted, 5, 7, maximize) == assign(ctx, grid, 5, 7, maximize));
    }
    // A cost range wider than the exponent range is refused, not merged with 0.
    // Costs near DBL_MAX next to subnormal costs cannot be scaled exactly.
    std::vector<double> span(base);
    for (double& v : span) v *= 1e-310;
    span[0] = 1.7e308;
    assign(ctx, span, 5, 7, 0, N4M_ERR_UNSUPPORTED);
    std::puts("assignment sweep: invariant");
}

void normal_sweep(n4m_context_t* ctx) {
    const double cov[9] = {2.0, 0.3, 0.1, 0.3, 1.0, -0.2, 0.1, -0.2, 0.5};
    const double zero[3] = {0, 0, 0};
    auto draw = [&](int k) {
        std::vector<double> c(cov, cov + 9), out(18, -1.0);
        for (double& v : c) v = std::ldexp(v, 2 * k);
        n4m_matrix_view_t C{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&C, c.data(), 3, 3, N4M_DTYPE_F64) == N4M_OK);
        REQUIRE(n4m_random_multivariate_normal(ctx, 42, zero, &C, 6, out.data()) == N4M_OK);
        return out;
    };
    const auto base = draw(0);
    // Covariances scaled by 4^k from k = -509 (smallest entry 0.1 * 2^-1018,
    // still normal) to k = 511 (largest entry 2^1023): the factor is computed
    // in a scaled space, so the draws are the exact 2^k multiples. Without it,
    // L*L products underflow at k = -509.
    for (const int k : {-509, -500, -100, 100, 500, 511}) {
        const auto got = draw(k);
        for (size_t i = 0; i < got.size(); ++i) REQUIRE(got[i] == std::ldexp(base[i], k));
    }
    std::puts("multivariate normal sweep: exact 2^k scaling");
}

double ari(n4m_context_t* ctx, const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    double out = 0.0;
    REQUIRE(n4m_preflight_adjusted_rand_index(ctx, a.data(), b.data(),
                                              static_cast<int64_t>(a.size()), &out) == N4M_OK);
    return out;
}

void ari_invariance(n4m_context_t* ctx) {
    std::vector<int64_t> a, b;
    for (int64_t i = 0; i < 200; ++i) {
        a.push_back(i % 7);
        b.push_back((i / 3) % 5);
    }
    const double want = ari(ctx, a, b);
    std::vector<int64_t> big_a(a), big_b(b);
    for (auto& v : big_a) v = v * 1000003 + (int64_t{1} << 62);
    for (auto& v : big_b) v = std::numeric_limits<int64_t>::min() + v;
    REQUIRE(ari(ctx, big_a, big_b) == want);
    double out = 0.0;
    REQUIRE(n4m_preflight_adjusted_rand_index(ctx, a.data(), b.data(), (int64_t{1} << 31) + 1, &out) ==
            N4M_ERR_UNSUPPORTED);
    std::puts("ari: invariant to label values");
}

}  // namespace

int main() {
    n4m_context_t* ctx = nullptr;
    try {
        REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        hdbscan_sweep(ctx);
        assignment_sweep(ctx);
        assignment_exactness(ctx);
        normal_sweep(ctx);
        ari_invariance(ctx);
        n4m_context_destroy(ctx);
        std::puts("n4m_cluster_robustness_tests: ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "n4m_cluster_robustness_tests failed: %s\n", e.what());
        if (ctx) n4m_context_destroy(ctx);
        return 1;
    }
}
