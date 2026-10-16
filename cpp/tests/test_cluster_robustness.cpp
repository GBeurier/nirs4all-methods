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
#include <utility>
#include <vector>

#include "n4m/n4m.h"

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " @" + std::to_string(__LINE__)); } while (false)

namespace {

constexpr int64_t sentinel_label = -777;
// Domain predicate of rules 13/15, written independently of the library:
// finite, nonzero |x| in [2^-256, 2^256] and max/min nonzero |x| <= 2^32.
bool in_domain(const std::vector<double>& x) {
    double low = 0.0, high = 0.0;
    for (const double v : x) {
        if (!std::isfinite(v)) return false;
        const double a = std::fabs(v);
        if (a == 0.0) continue;
        if (low == 0.0 || a < low) low = a;
        if (a > high) high = a;
    }
    return high == 0.0 || (low >= 0x1p-256 && high <= 0x1p256 && high / low <= 0x1p32);
}

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
    // Coordinates lie in [1, 16]: k in [-256, 252] stays inside the domain.
    for (const int k : {-1014, -257, -256, -200, -100, 100, 200, 252, 253, 1000}) {
        std::vector<double> x(base_x);
        for (double& v : x) v = std::ldexp(v, k);
        const Fit f = hdbscan(ctx, x, 21, 2);
        if (!in_domain(x)) {
            REQUIRE(f.status == N4M_ERR_UNSUPPORTED);
            for (const int64_t l : f.labels) REQUIRE(l == sentinel_label);
            std::printf("hdbscan sweep k=%5d refused (outside the domain)\n", k);
            continue;
        }
        // Inside the domain no distance, density or sum leaves the normal
        // range, so power-of-two scaling is exact on every output.
        REQUIRE(f.status == N4M_OK);
        REQUIRE(f.labels == base.labels && f.parent == base.parent);
        REQUIRE(f.membership == base.membership && f.outlier == base.outlier &&
                f.persistence == base.persistence);
        for (size_t j = 0; j < f.birth.size(); ++j) {
            REQUIRE(f.birth[j] == std::ldexp(base.birth[j], -k));
            REQUIRE(f.death[j] == std::ldexp(base.death[j], -k));
            REQUIRE(f.stability[j] == std::ldexp(base.stability[j], -k));
        }
        std::printf("hdbscan sweep k=%5d exact\n", k);
    }

    // Review 2026-10-13: 1e-308 x [0,1,3,10,11,14] made the stability sums
    // overflow (persistence NaN) and 1e-310 an infinite density for a
    // positive distance, both with N4M_OK. Both lie outside the domain now.
    const std::vector<double> line = {0, 1, 3, 10, 11, 14};
    for (const double factor : {1e-308, 1e-310}) {
        std::vector<double> x(line);
        for (double& v : x) v *= factor;
        const Fit f = hdbscan(ctx, x, 6, 1);
        REQUIRE(f.status == N4M_ERR_UNSUPPORTED);
        for (const int64_t l : f.labels) REQUIRE(l == sentinel_label);
    }

    // Exact translations and a constant column inside the domain leave every
    // output bit-identical.
    std::vector<double> grid;
    for (int i = 0; i < 21; ++i) {
        grid.push_back(4.0 * ((i * 7) % 23 + (i < 10 ? 0 : 60)));
        grid.push_back(4.0 * ((i * 11) % 19));
    }
    const Fit g = hdbscan(ctx, grid, 21, 2);
    REQUIRE(g.status == N4M_OK);
    std::vector<double> shifted(grid), wide, high_grid(grid), high_shift(grid);
    for (double& v : shifted) v += std::ldexp(1.0, 54);
    for (double& v : high_grid) v = std::ldexp(v, 150);
    for (double& v : high_shift) v = std::ldexp(v, 150) + std::ldexp(1.0, 200);
    for (size_t i = 0; i < 21; ++i) {
        wide.push_back(grid[2 * i]);
        wide.push_back(std::ldexp(1.0, 30));
        wide.push_back(grid[2 * i + 1]);
    }
    for (const auto* x : {&shifted, &wide, &high_grid, &high_shift}) REQUIRE(in_domain(*x));
    const Fit s = hdbscan(ctx, shifted, 21, 2), w = hdbscan(ctx, wide, 21, 3);
    const Fit h = hdbscan(ctx, high_grid, 21, 2), hs = hdbscan(ctx, high_shift, 21, 2);
    for (const Fit* f : {&s, &w}) {
        REQUIRE(f->status == N4M_OK && f->labels == g.labels && f->membership == g.membership &&
                f->stability == g.stability && f->persistence == g.persistence);
    }
    REQUIRE(h.status == N4M_OK && hs.status == N4M_OK);
    REQUIRE(hs.labels == h.labels && hs.membership == h.membership && hs.stability == h.stability &&
            h.labels == g.labels && h.membership == g.membership);
    // The same constant column at 1e300 leaves the domain and is refused.
    for (size_t i = 0; i < 21; ++i) wide[3 * i + 1] = 1e300;
    REQUIRE(hdbscan(ctx, wide, 21, 3).status == N4M_ERR_UNSUPPORTED);
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

std::vector<std::pair<int64_t, int64_t>> assign_rows(n4m_context_t* ctx, std::vector<double> cost,
                                                     int64_t rows, int64_t cols) {
    n4m_matrix_view_t C{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&C, cost.data(), rows, cols, N4M_DTYPE_F64) == N4M_OK);
    std::vector<int64_t> r(static_cast<size_t>(std::min(rows, cols))), c(r.size());
    int64_t assigned = -1;
    REQUIRE(n4m_linear_sum_assignment(ctx, &C, 0, r.data(), c.data(), &assigned) == N4M_OK);
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE(i == 0 || r[i] > r[i - 1]);
        pairs.emplace_back(r[i], c[i]);
    }
    return pairs;
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
    // block of [[-1e20,1e20,1e20],[1e20,1,0],[1e20,0,1]] into a constant and
    // returned [0, 1, 2]. That matrix (spread 1e20 > 2^32) is now outside the
    // domain; its analogue at the spread limit 2^32 keeps the optimum.
    const std::vector<double> review = {-1e20, 1e20, 1e20, 1e20, 1, 0, 1e20, 0, 1};
    assign(ctx, review, 3, 3, 0, N4M_ERR_UNSUPPORTED);
    const double big = std::ldexp(1.0, N4M_DOMAIN_MAX_SPREAD_LOG2);
    const std::vector<double> centring = {-big, big, big, big, 1, 0, big, 0, 1};
    REQUIRE((assign(ctx, centring, 3, 3, 0) == std::vector<int64_t>{0, 2, 1}));
    check_optimal(ctx, centring, 3, 3);
    // Centring made 1e308 - (-1e308) overflow and declared that matrix
    // infeasible; it is outside the domain now. At the domain edge 2^256 the
    // matrix is feasible with exact cost 0.
    assign(ctx, {-1e308, 1e308, -1e308, 1e308}, 2, 2, 0, N4M_ERR_UNSUPPORTED);
    const std::vector<double> edge = {-0x1p256, 0x1p256, -0x1p256, 0x1p256};
    const auto columns = assign(ctx, edge, 2, 2, 0);
    REQUIRE((columns == std::vector<int64_t>{0, 1} || columns == std::vector<int64_t>{1, 0}));
    check_optimal(ctx, edge, 2, 2);
    // Signed mixtures inside the domain: every matrix draws full 53-bit
    // mantissas with exponents in a window [b, b + 31], b in [-256, 224] (spread
    // below 2^32; sums of such costs are not exact in double), plus exact zeros; checked
    // against the exact enumeration, for minimisation, maximisation and tall
    // (transposed) shapes.
    uint64_t state = 0x9e3779b97f4a7c15u;
    auto next = [&state]() {
        state = state * 6364136223846793005u + 1442695040888963407u;
        return state >> 11;
    };
    for (int trial = 0; trial < 600; ++trial) {
        const auto rows = static_cast<int64_t>(2 + next() % 3);
        const auto cols = rows + static_cast<int64_t>(next() % 3);
        const int base = static_cast<int>(next() % 481) - 256;
        std::vector<double> cost(static_cast<size_t>(rows * cols));
        for (double& v : cost) {
            if (next() % 10 == 0) {
                v = 0.0;
                continue;
            }
            const double mantissa =
                1.0 + std::ldexp(static_cast<double>(next() & ((uint64_t{1} << 52) - 1)), -52);
            const int exponent = base + static_cast<int>(next() % 32);
            v = (next() % 2 ? -1.0 : 1.0) * std::ldexp(mantissa, exponent);
        }
        REQUIRE(in_domain(cost));
        check_optimal(ctx, cost, rows, cols);
        // Maximisation is minimisation of the negated costs.
        std::vector<double> negated(cost);
        for (double& v : negated) v = -v;
        REQUIRE(assign(ctx, cost, rows, cols, 1) == assign(ctx, negated, rows, cols, 0));
        // A tall matrix is solved on its transpose; same optimum value.
        std::vector<double> tall(cost.size());
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < cols; ++c)
                tall[static_cast<size_t>(c * rows + r)] = cost[static_cast<size_t>(r * cols + c)];
        const auto tall_rows = assign_rows(ctx, tall, cols, rows);
        const auto wide_cols = assign(ctx, cost, rows, cols, 0);
        Exact tall_cost, wide_cost;
        for (int64_t r = 0; r < rows; ++r) {
            accumulate(tall_cost, tall[static_cast<size_t>(tall_rows[static_cast<size_t>(r)].first * rows +
                                                           tall_rows[static_cast<size_t>(r)].second)]);
            accumulate(wide_cost, cost[static_cast<size_t>(r * cols + wide_cols[static_cast<size_t>(r)])]);
        }
        REQUIRE(compare_sums(tall_cost, wide_cost) == 0);
    }
    // Domain boundaries: smallest and largest magnitudes, spread 2^32.
    assign(ctx, {0x1p-256, 0x1p-225, 0x1p-230, 0x1p-240}, 2, 2, 0);
    assign(ctx, {0x1p-257, 0x1p-226, 0x1p-230, 0x1p-240}, 2, 2, 0, N4M_ERR_UNSUPPORTED);
    assign(ctx, {0x1p256, 0x1p230, 0x1p255, 0x1p250}, 2, 2, 0);
    assign(ctx, {0x1p256 * (1 + 0x1p-52), 0x1p230, 0x1p255, 0x1p250}, 2, 2, 0, N4M_ERR_UNSUPPORTED);
    const double spread = std::ldexp(1.0, N4M_DOMAIN_MAX_SPREAD_LOG2);
    assign(ctx, {1.0, spread, 2.0, 3.0}, 2, 2, 0);
    assign(ctx, {1.0, spread * (1 + 0x1p-52), 2.0, 3.0}, 2, 2, 0, N4M_ERR_UNSUPPORTED);
    assign(ctx, {1.0, 0x1p40, 2.0, 3.0}, 2, 2, 0, N4M_ERR_UNSUPPORTED);
    // A forbidden pair (+inf) is not a numerical value.
    assign(ctx, {std::numeric_limits<double>::infinity(), 1.0, 2.0, 3.0}, 2, 2, 0);
    std::puts("assignment: exact optimum inside the domain, refusals at its edges");
}

void assignment_sweep(n4m_context_t* ctx) {
    std::vector<double> base(35);
    for (size_t i = 0; i < base.size(); ++i)
        base[i] = 1.0 + std::fmod(static_cast<double>(i) * 7.31, 9.0);
    for (const int32_t maximize : {0, 1}) {
        const auto want = assign(ctx, base, 5, 7, maximize);
        // Costs in [1, 10): k in [-256, 252] stays inside the domain.
        for (const int k : {-1014, -257, -256, -100, 100, 252, 253, 1000}) {
            std::vector<double> x(base);
            for (double& v : x) v = std::ldexp(v, k);
            if (in_domain(x)) {
                REQUIRE(assign(ctx, x, 5, 7, maximize) == want);
            } else {
                assign(ctx, x, 5, 7, maximize, N4M_ERR_UNSUPPORTED);
            }
        }
        // Exact translation of an integer grid.
        std::vector<double> grid(base), shifted(base);
        for (size_t i = 0; i < base.size(); ++i) {
            grid[i] = 256.0 * std::floor(base[i] * 1000.0);
            shifted[i] = grid[i] + std::ldexp(1.0, 60);
        }
        REQUIRE(assign(ctx, shifted, 5, 7, maximize) == assign(ctx, grid, 5, 7, maximize));
    }
    std::puts("assignment sweep: invariant inside the domain");
}

// Review 2026-10-13 P0 (memory): a 1 x 32,000,000 zero matrix passed the
// cell and operation limits but needed more than 2 GiB in WASM32. The 256 MiB
// working-memory budget refuses it before any allocation; an overlapping
// 4 x 2^22 view checks the same refusal with a small backing buffer.
void assignment_memory(n4m_context_t* ctx) {
    std::vector<double> backing((size_t{1} << 22) + 3, 0.0);
    n4m_matrix_view_t V{};
    REQUIRE(n4m_matrix_view_init_strided(&V, backing.data(), 4, int64_t{1} << 22, 1, 1,
                                         N4M_DTYPE_F64) == N4M_OK);
    std::vector<int64_t> r(4, -5), c(4, -5);
    int64_t assigned = -1;
    REQUIRE(n4m_linear_sum_assignment(ctx, &V, 0, r.data(), c.data(), &assigned) ==
            N4M_ERR_UNSUPPORTED);
    REQUIRE(r[0] == -5 && c[0] == -5);
    std::vector<double> row(32000000, 0.0);
    REQUIRE(n4m_matrix_view_init_rowmajor(&V, row.data(), 1, 32000000, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_linear_sum_assignment(ctx, &V, 0, r.data(), c.data(), &assigned) ==
            N4M_ERR_UNSUPPORTED);
    std::puts("assignment: memory budget refuses before allocating");
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
    // Covariance entries in [0.1, 2]: k in [-126, 127] keeps them inside the
    // domain, where the draws are the exact 2^k multiples.
    for (const int k : {-126, -100, 100, 127}) {
        const auto got = draw(k);
        for (size_t i = 0; i < got.size(); ++i) REQUIRE(got[i] == std::ldexp(base[i], k));
    }
    // Just outside: smallest entry 0.1 * 4^-127 < 2^-256, largest 2 * 4^128 > 2^256.
    for (const int k : {-127, 128}) {
        std::vector<double> c(cov, cov + 9), out(18, -1.0);
        for (double& v : c) v = std::ldexp(v, 2 * k);
        n4m_matrix_view_t C{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&C, c.data(), 3, 3, N4M_DTYPE_F64) == N4M_OK);
        REQUIRE(n4m_random_multivariate_normal(ctx, 42, zero, &C, 6, out.data()) ==
                N4M_ERR_UNSUPPORTED);
        REQUIRE(out[0] == -1.0);
    }
    // Spread: a mean with 1 and 2^32 is accepted, one ulp more is refused.
    {
        double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        n4m_matrix_view_t C{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&C, identity, 3, 3, N4M_DTYPE_F64) == N4M_OK);
        std::vector<double> out(9);
        const double ok_mean[3] = {1.0, 0x1p32, 0.0}, bad_mean[3] = {1.0, 0x1p32 * (1 + 0x1p-52), 0.0};
        REQUIRE(n4m_random_multivariate_normal(ctx, 1, ok_mean, &C, 3, out.data()) == N4M_OK);
        REQUIRE(n4m_random_multivariate_normal(ctx, 1, bad_mean, &C, 3, out.data()) ==
                N4M_ERR_UNSUPPORTED);
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
    // 32 bytes per element: more than 2^23 elements exceed the 256 MiB budget,
    // refused before the labels are read.
    REQUIRE(n4m_preflight_adjusted_rand_index(ctx, a.data(), b.data(), (int64_t{1} << 23) + 1, &out) ==
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
        assignment_memory(ctx);
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
