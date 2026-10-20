// SPDX-License-Identifier: CECILL-2.1
// Reusable, caller-buffer diagnostics for Preflight and other C ABI clients.
#include "n4m/metrics.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "core/common/centring.hpp"
#include "core/common/context.hpp"
#include "core/common/double_double.hpp"
#include "core/common/exact_sum.hpp"
#include "core/common/matrix_view.hpp"
#include "core/common/numerical_domain.hpp"
#include "core/clustering/_vendored/hdbscan.hpp"
#include "core/diagnostics/preflight_pca_budget.hpp"
#include "core/optimization/_vendored/rectangular_lsap.hpp"
#include "core/common/pca_helper.h"
#include "core/common/rng_pcg64.h"
#include "core/common/svd.h"
#include "core/preprocessing/derivatives/savitzky_golay.h"
#include "core/preprocessing/scatter/snv.h"

namespace {
using ::n4m::core::bounding_exponent;
using ::n4m::core::center_in_place;
using ::n4m::core::Context;
using ::n4m::core::DD;
using ::n4m::core::dd_add;
using ::n4m::core::dd_div;
using ::n4m::core::dd_mul;
using ::n4m::core::dd_neg;
using ::n4m::core::dd_two_sum;
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr long double max_moment_products = 250000000.0L;
constexpr long double max_hdbscan_products = 2500000000.0L;
// Contingency-table budget of the categorical association kernels (cells).
constexpr size_t max_table_cells = 10000000;
constexpr int32_t max_noise_window = 1001;
constexpr int32_t max_noise_polyorder = 10;
// Absolute tolerance on normalized Savitzky-Golay reproduction moments.
constexpr long double moment_tolerance = 1e-9L;
// Cumulative working memory of one clustering-kernel call, checked before any
// allocation: the WASM32 build has at most 2 GiB and aborts on a failed
// allocation (C++ exceptions are not caught there), so each call stays below
// 256 MiB.
constexpr long double max_working_bytes = 268435456.0L;

// Supported numerical domain D (core/common/numerical_domain.hpp).
using ::n4m::core::domain_message;
using ::n4m::core::DomainRange;
using ::n4m::core::in_numerical_domain;
using ::n4m::core::magnitude_domain_message;

// Holm and BH only multiply p-values by small integers or ratios and take
// running min/max: no subtraction, so no spread limit. Their domain is
// p in [0, 1] (checked by the callers) with nonzero p >= 2^-1022: subnormal
// p-values are refused rather than carried with reduced precision.
constexpr const char* p_value_domain_message =
    "input outside the supported numerical domain (nonzero p-values >= 2^-1022)";
bool p_values_in_domain(const double* p, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != 0.0 && p[i] < DBL_MIN) return false;
    return true;
}

n4m_status_t fail(n4m_context_t* ctx, n4m_status_t code, const char* message) noexcept {
    if (ctx) static_cast<Context*>(ctx)->set_error(message);
    return code;
}

n4m_status_t check_view(n4m_context_t* ctx, const n4m_matrix_view_t* v) noexcept {
    if (!v) return fail(ctx, N4M_ERR_NULL_POINTER, "null matrix view");
    const auto st = ::n4m::core::validate_nonnull_view(*v);
    if (st != N4M_OK) return fail(ctx, st, "invalid matrix view");
    if (v->dtype != N4M_DTYPE_F64)
        return fail(ctx, N4M_ERR_DTYPE_MISMATCH, "matrix must be f64");
    const auto max_index = static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double));
    const auto row = v->rows > 0 ? static_cast<uint64_t>(v->rows - 1) : 0;
    const auto col = v->cols > 0 ? static_cast<uint64_t>(v->cols - 1) : 0;
    const auto rs = static_cast<uint64_t>(v->row_stride);
    const auto cs = static_cast<uint64_t>(v->col_stride);
    if ((rs && row > max_index / rs) || (cs && col > max_index / cs) ||
        row * rs > max_index - col * cs)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "matrix view offset overflow");
    return N4M_OK;
}

// Supported numerical domain (rules 13 and 15) of the PCA kernel: every
// nonzero |x| in [2^-pca_domain_exp, 2^pca_domain_exp] and max/min nonzero
// |x| <= 2^N4M_DOMAIN_MAX_SPREAD_LOG2. The PCA declares a magnitude bound
// tighter than 2^256 because trace(C^2) is quartic in X: with |x| <= 2^128
// every published value (resolved eigenvalues, resolution, both moments,
// scores) is a normal double.
constexpr int pca_domain_exp = 128;
// Absolute accuracy of the PCA variances, in units of max(n,p)*eps*lambda_max.
constexpr double pca_resolution_factor = 16.0;

double at(const n4m_matrix_view_t& v, int64_t i, int64_t j) noexcept {
    return static_cast<const double*>(v.data)[i * v.row_stride + j * v.col_stride];
}
void put(const n4m_matrix_view_t& v, int64_t i, int64_t j, double x) noexcept {
    static_cast<double*>(v.data)[i * v.row_stride + j * v.col_stride] = x;
}

// Unsigned 128-bit helpers for exact pair-count products (portable to MSVC).
struct U128 {
    uint64_t hi = 0, lo = 0;
};
U128 mul_u64(uint64_t a, uint64_t b) noexcept {
    const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t middle = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    return {p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32), (middle << 32) | (p00 & 0xffffffffu)};
}
U128 add_u128(U128 a, U128 b) noexcept {
    const uint64_t lo = a.lo + b.lo;
    return {a.hi + b.hi + (lo < a.lo ? 1u : 0u), lo};
}
U128 sub_u128(U128 a, U128 b) noexcept {  // requires a >= b
    return {a.hi - b.hi - (a.lo < b.lo ? 1u : 0u), a.lo - b.lo};
}
bool less_u128(U128 a, U128 b) noexcept { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
long double to_long_double(U128 v) noexcept {
    return static_cast<long double>(v.hi) * 18446744073709551616.0L + static_cast<long double>(v.lo);
}

// Type-7 quantile a + w (b - a) of sorted values. (n - 1) q and w are exact
// for q in {1/4, 1/2, 3/4}; b - a is kept exactly (TwoSum) and the result is
// rounded essentially once, so it stays within [a, b].
double quantile(const std::vector<double>& sorted, double q) noexcept {
    const double h = static_cast<double>(sorted.size() - 1) * q;
    const auto lo = static_cast<size_t>(h);
    const auto hi = std::min(lo + 1, sorted.size() - 1);
    const double w = h - static_cast<double>(lo);
    const double a = sorted[lo], b = sorted[hi];
    if (w == 0.0 || a == b) return a;
    const DD diff = dd_two_sum(b, -a);
    return std::fma(w, diff.hi, a) + w * diff.lo;
}

// Mean of finite values as a double-double m_hi + m_lo: the double-double
// sum (hi, lo) carries the exact sum to an absolute error of order
// n u^2 sum|x|, and the division remainder hi - m_hi n is exact (fma). No
// refinement pass: one would add u mean|x - m|, which dominates under
// cancellation. Returns the rounded mean and the double-double sum of squared
// deviations from m_hi + m_lo. Mean error of order n u^2 mean|x| + u |m|
// (final rounding).
struct MeanSquares { double mean = 0.0, ss = 0.0; };
MeanSquares mean_and_squares(const std::vector<double>& v) {
    const auto n = static_cast<double>(v.size());
    DD sum;
    for (double x : v) sum = dd_add(sum, x);
    const double m_hi = sum.hi / n;
    const double m_lo = (std::fma(-m_hi, n, sum.hi) + sum.lo) / n;
    DD ss;
    for (double x : v) {
        const double d = (x - m_hi) - m_lo;
        ss = dd_add(ss, d * d);
    }
    return {m_hi + m_lo, ss.hi + ss.lo};
}

struct GroupCategoryHash {
    size_t operator()(const std::pair<int64_t, int64_t>& value) const noexcept {
        const auto first = std::hash<int64_t>{}(value.first);
        const auto second = std::hash<int64_t>{}(value.second);
        return first ^ (second + size_t{0x9e3779b9} + (first << 6) + (first >> 2));
    }
};

// Dense 0..k-1 indices for sorted distinct values; counts per index.
template <typename T>
std::vector<int64_t> dense_codes(const std::vector<T>& values, std::vector<int64_t>& counts) {
    std::vector<T> levels(values);
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    counts.assign(levels.size(), 0);
    std::vector<int64_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = std::lower_bound(levels.begin(), levels.end(), values[i]) - levels.begin();
        ++counts[static_cast<size_t>(out[i])];
    }
    return out;
}

double log10_factorial_ratio(int64_t n, const std::vector<int64_t>& counts) noexcept {
    double value = std::lgamma(static_cast<double>(n) + 1.0);
    for (int64_t c : counts) value -= std::lgamma(static_cast<double>(c) + 1.0);
    return value / std::log(10.0);
}

// Centered copy, stable under translation and scale: values are divided by
// a power of two 2^scale bounding max|v| (exact), centered with
// center_in_place, then divided by a power of two 2^variation bounding the
// centered variation (exact), so max|c| lies in [1/2, 1) whatever the input
// magnitude. Returns the sum of squares of that copy (>= 1/4, or 0 when
// every value is equal); the true centered sum of squares is that value
// times 2^(2 * (scale + variation)).
double stable_center(const std::vector<double>& values, std::vector<double>& centered,
                     int* scale = nullptr, int* variation = nullptr) {
    const int e1 = bounding_exponent(values);
    centered.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) centered[i] = std::ldexp(values[i], -e1);
    center_in_place(centered);
    const int e2 = bounding_exponent(centered);
    if (scale) *scale = e1;
    if (variation) *variation = e2;
    double ss = 0.0;
    for (double& v : centered) {
        v = std::ldexp(v, -e2);
        ss += v * v;
    }
    return ss;
}

// Sum of squares with a power-of-two scale chosen from max|d|: returns
// q and the exponent e such that sum(d^2) = q * 2^(2e), q in [1/4, n).
double scaled_sum_squares(const std::vector<double>& d, int& exponent) noexcept {
    exponent = bounding_exponent(d);
    double q = 0.0;
    for (double v : d) {
        const double scaled = std::ldexp(v, -exponent);
        q += scaled * scaled;
    }
    return q;
}

// Distinct arrangements n!/prod(counts!) when at most cap, else cap + 1.
uint64_t arrangement_count(const std::vector<int64_t>& counts, uint64_t cap) noexcept {
    uint64_t result = 1, seen = 0;
    for (int64_t k : counts)
        for (int64_t j = 1; j <= k; ++j) {
            ++seen;
            if (seen > UINT64_MAX / result) return cap + 1;
            result = result * seen / static_cast<uint64_t>(j);
            if (result > cap) return cap + 1;
        }
    return result;
}

// One categorical factor against one target, one row per exchangeable unit.
// Targets are encoded by dense value codes; for categorical targets value
// codes are the class codes.
struct FactorProblem {
    bool categorical = false;
    std::vector<int64_t> level, level_n, value, value_n;
    std::vector<double> value_centered;  // centered target per value code
    std::vector<double> raw;             // original targets (-0.0 as +0.0)
    double sst = 0.0;
    int scale = 0, variation = 0;  // true SST = sst * 2^(2 (scale + variation))
    std::vector<double> sums;
    std::vector<int64_t> table;

    // eta^2 = sum_g S_g^2/n_g / SST, or Cramer's V^2, of L[i] x Y[i].
    double statistic(const int64_t* L, const int64_t* Y) {
        const size_t n = level.size(), g = level_n.size();
        if (!categorical) {
            std::fill(sums.begin(), sums.end(), 0.0);
            for (size_t i = 0; i < n; ++i)
                sums[static_cast<size_t>(L[i])] += value_centered[static_cast<size_t>(Y[i])];
            double ssb = 0.0;
            for (size_t a = 0; a < g; ++a)
                ssb += sums[a] * sums[a] / static_cast<double>(level_n[a]);
            return std::min(1.0, ssb / sst);
        }
        // Chi-square from exact integer residuals d = O*n - r*c (so that
        // O - E = d/n), never as sum(O^2/E) - n: every term d^2/(n r c) is
        // nonnegative, independence gives exactly 0 and the relative error
        // is O(cells * DBL_EPSILON). |d| <= n^2 < 2^63 (n is capped).
        const size_t c = value_n.size();
        std::fill(table.begin(), table.end(), 0);
        for (size_t i = 0; i < n; ++i)
            ++table[static_cast<size_t>(L[i]) * c + static_cast<size_t>(Y[i])];
        const auto total = static_cast<int64_t>(n);
        const double nd = static_cast<double>(n);
        double chi2 = 0.0;
        for (size_t a = 0; a < g; ++a) for (size_t b = 0; b < c; ++b) {
            const int64_t d = table[a * c + b] * total - level_n[a] * value_n[b];
            if (d != 0) {
                const double dd = static_cast<double>(d);
                chi2 += dd * dd /
                        (nd * static_cast<double>(level_n[a]) * static_cast<double>(value_n[b]));
            }
        }
        return std::min(1.0, chi2 / (nd * static_cast<double>(std::min(g, c) - 1)));
    }
    bool defined() const noexcept {
        return level_n.size() >= 2 && value_n.size() >= 2 && (categorical || sst > 0.0);
    }
    // Cost of one statistic: rows plus the contingency table it scans.
    long double cost() const noexcept {
        return static_cast<long double>(level.size()) +
               static_cast<long double>(level_n.size()) *
                   static_cast<long double>(categorical ? value_n.size() : 1);
    }
    // Work buffers, allocated only after the caller's budget checks.
    void allocate() {
        sums.assign(level_n.size(), 0.0);
        if (categorical) table.assign(level_n.size() * value_n.size(), 0);
    }
};

n4m_status_t prepare_factor(n4m_context_t* ctx, const int64_t* factor_codes,
                            const double* targets, int64_t n, int32_t categorical,
                            FactorProblem& p) {
    if (!factor_codes || !targets)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null association buffer");
    if (n < 2 || (categorical != 0 && categorical != 1) ||
        static_cast<uint64_t>(n) > static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid association dimensions");
    constexpr double max_exact_category = 9007199254740991.0;
    // +0.0 replaces -0.0 so both zero encodings share one value.
    std::vector<double> exact(targets, targets + n);
    for (double& y : exact) {
        if (!std::isfinite(y)) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite target");
        if (categorical && (std::abs(y) > max_exact_category || std::trunc(y) != y))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "category codes must be exact integers");
        y += 0.0;
    }
    // Category codes are labels; only continuous targets carry magnitudes.
    if (!categorical && !in_numerical_domain(exact.data(), exact.size()))
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    p.categorical = categorical != 0;
    p.level = dense_codes(std::vector<int64_t>(factor_codes, factor_codes + n), p.level_n);
    p.value = dense_codes(exact, p.value_n);
    std::vector<double> centered;
    p.sst = stable_center(exact, centered, &p.scale, &p.variation);
    p.raw = exact;
    p.value_centered.assign(p.value_n.size(), 0.0);
    for (size_t i = 0; i < exact.size(); ++i)
        p.value_centered[static_cast<size_t>(p.value[i])] = centered[i];
    // Exact integer chi-square residuals need n^2 < 2^63.
    if (p.categorical && n > 3000000000LL)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "categorical association limited to 3e9 units");
    // Checked levels x classes product: refused before any table allocation.
    if (p.categorical && p.level_n.size() > max_table_cells / p.value_n.size())
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "association contingency table exceeds 10 million cells");
    return N4M_OK;
}

// Average ranks 1..n; tied values share the mean of their ranks.
std::vector<double> average_ranks(const std::vector<double>& values) {
    const size_t n = values.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(),
              [&values](size_t a, size_t b) { return values[a] < values[b]; });
    std::vector<double> ranks(n);
    for (size_t start = 0; start < n;) {
        size_t end = start + 1;
        while (end < n && values[order[end]] == values[order[start]]) ++end;
        const double rank = 0.5 * static_cast<double>(start + 1 + end);
        for (size_t k = start; k < end; ++k) ranks[order[k]] = rank;
        start = end;
    }
    return ranks;
}

// Paired x, y (ranks for Spearman), one row per exchangeable unit.
struct CorrelationProblem {
    std::vector<int64_t> x_code, x_n, y_code, y_n;
    std::vector<double> x_tab, y_tab;  // centered transform per value code
    double denominator = 0.0;

    double statistic(const int64_t* X, const int64_t* Y) const {
        double sxy = 0.0;
        for (size_t i = 0; i < x_code.size(); ++i)
            sxy += x_tab[static_cast<size_t>(X[i])] * y_tab[static_cast<size_t>(Y[i])];
        return std::max(-1.0, std::min(1.0, sxy / denominator));
    }
    bool defined() const noexcept { return denominator > 0.0; }
};

n4m_status_t prepare_correlation(n4m_context_t* ctx, const double* x, const double* y,
                                 int64_t n, int32_t method, CorrelationProblem& p) {
    if (!x || !y) return fail(ctx, N4M_ERR_NULL_POINTER, "null correlation buffer");
    if (n < 2 || (method != 0 && method != 1) ||
        static_cast<uint64_t>(n) > static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid correlation dimensions or method");
    std::vector<double> xs(x, x + n), ys(y, y + n);
    for (size_t i = 0; i < xs.size(); ++i) {
        if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite correlation input");
        xs[i] += 0.0;
        ys[i] += 0.0;
    }
    if (!in_numerical_domain(xs.data(), xs.size()) || !in_numerical_domain(ys.data(), ys.size()))
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    p.x_code = dense_codes(xs, p.x_n);
    p.y_code = dense_codes(ys, p.y_n);
    if (method == 1) {
        xs = average_ranks(xs);
        ys = average_ranks(ys);
    }
    std::vector<double> xc, yc;
    const double sxx = stable_center(xs, xc), syy = stable_center(ys, yc);
    p.x_tab.assign(p.x_n.size(), 0.0);
    p.y_tab.assign(p.y_n.size(), 0.0);
    for (size_t i = 0; i < xc.size(); ++i) {
        p.x_tab[static_cast<size_t>(p.x_code[i])] = xc[i];
        p.y_tab[static_cast<size_t>(p.y_code[i])] = yc[i];
    }
    if (p.x_n.size() >= 2 && p.y_n.size() >= 2 && sxx > 0.0 && syy > 0.0)
        p.denominator = std::sqrt(sxx) * std::sqrt(syy);
    return N4M_OK;
}

// Exact permutation distribution over the distinct arrangements of `side`
// (a multiset of codes) with `other` fixed: every distinct arrangement has
// equal probability under uniform permutation. Writes N, #{T >= T_obs -
// tol}/N, #{T >= T_max - tol}/N and T_max (NaN when undefined).
// `side` arrives in its observed order; nothing is allocated or evaluated
// before the arrangement count and the operation budget are accepted.
template <typename Setup, typename Stat>
n4m_status_t exact_distribution(n4m_context_t* ctx, std::vector<int64_t> side,
                                const std::vector<int64_t>& side_n, int64_t max_arrangements,
                                bool defined, long double cost_per_statistic, Setup setup,
                                Stat stat, double* out_four) {
    if (max_arrangements < 1 || max_arrangements > 10000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "max_arrangements must lie in [1, 1e7]");
    const uint64_t cap = static_cast<uint64_t>(max_arrangements);
    const uint64_t count = arrangement_count(side_n, cap);
    if (count > cap)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "distinct arrangements exceed max_arrangements");
    if (2.0L * static_cast<long double>(count) * cost_per_statistic > max_moment_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "exact enumeration exceeds 250 million operation budget");
    for (int i = 0; i < 4; ++i) out_four[i] = nan_value;
    out_four[0] = static_cast<double>(count);
    if (!defined) return N4M_OK;
    setup();
    const double observed = stat(side.data());
    const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
    std::sort(side.begin(), side.end());
    std::vector<int64_t> start(side);
    double maximum = -std::numeric_limits<double>::infinity();
    uint64_t at_least = 0;
    do {
        const double t = stat(side.data());
        maximum = std::max(maximum, t);
        if (t >= observed - tolerance) ++at_least;
    } while (std::next_permutation(side.begin(), side.end()));
    uint64_t at_max = 0;
    side = start;
    do {
        if (stat(side.data()) >= maximum - tolerance) ++at_max;
    } while (std::next_permutation(side.begin(), side.end()));
    out_four[1] = static_cast<double>(at_least) / static_cast<double>(count);
    out_four[2] = static_cast<double>(at_max) / static_cast<double>(count);
    out_four[3] = maximum;
    return N4M_OK;
}
// Explicit double -> long double promotion (clang -Wdouble-promotion).
inline long double ld(double x) noexcept { return static_cast<long double>(x); }

// Error estimate of a residual sum of squares computed from n x k centred
// data by an explicit projection off a design of the given rank (or, for
// PERMDISP, an ANOVA of distances on k coordinates): every residual is off by
// about gamma times the range of its column (centring rounds relative to the
// shifted values, the projection relative to the centred data), gamma = 4
// (rank + k + 2) n DBL_EPSILON, so the residual SS is uncertain by about
// gamma^2 * range_ss, range_ss = n * sum_j (max_i x_ij - min_i x_ij)^2 in
// the scaled unit. This is a conservative first-order estimate for a
// well-conditioned design, not a proof. A residual SS at or below it is
// "numerically unresolved": the statistic is still computed and published,
// flagged, with the ratio estimate / residual SS (relative error of a
// resolved residual SS about 2 sqrt(ratio)).
long double residual_error_estimate(long double range_ss, size_t rank, size_t n, size_t k) {
    const long double gamma = 4.0L * static_cast<long double>(rank + k + 2) *
                              static_cast<long double>(n) *
                              static_cast<long double>(std::numeric_limits<double>::epsilon());
    return gamma * gamma * range_ss;
}

// Writes the resolution ratio estimate / residual SS (+inf for a zero
// residual SS) and the flag 1 (unresolved, ratio >= 1) or 0.
void write_resolution(long double estimate, long double residual, double* ratio, double* flag) {
    const long double r = residual > 0.0L ? estimate / residual
                                          : std::numeric_limits<long double>::infinity();
    *ratio = static_cast<double>(r);
    *flag = r >= 1.0L ? 1.0 : 0.0;
}

// Domain range over every entry of the given matrices together (one O(n k)
// pre-pass, no copy). range.finite is false when an entry is not finite.
DomainRange domain_of(std::initializer_list<const n4m_matrix_view_t*> views) noexcept {
    DomainRange range;
    for (const auto* v : views)
        for (int64_t i = 0; i < v->rows; ++i)
            for (int64_t j = 0; j < v->cols; ++j) range.add(at(*v, i, j));
    return range;
}

// Centred copy of X (row-major n x k) in a common power-of-two unit chosen
// from the CENTRED variation, never from raw magnitudes. Each column reuses
// the association helpers: exact prescale by 2^-e1 (bounding_exponent of the
// raw column), center_in_place (shift, mean, residual mean), then the
// exponent e2 bounding the centred column, C_j = e1 + e2. The common
// exponent is G = max_j C_j and y_ij = c_ij 2^(e1_j - G), so |y| < 1 and a
// constant column, whatever its amplitude, is exactly zero.
// Inside D (spread at most 2^32 over X) the rescaling is exact. Let m be the
// smallest nonzero |x|; every value is a multiple of g = ulp(m) >= 2^-53 m
// and at most 2^32 m, so every centred value is at most 2^33 m and a nonzero
// computed centred value is at least 2^-107 m / n (a rounded mean of
// multiples of g, or a difference with a mean rounded in the same binade):
// the rescaling spans at most 2^(140 + log2 n), the smallest nonzero y and
// its square stay normal doubles, no varying column is erased and no squared
// term underflows, even where long double is double. Rounding: the shift to
// the first row is rounded relative to the shifted value, so each centred
// value is off by about (n + 2) DBL_EPSILON times the range of its column,
// whatever the offset; the spread bound keeps that range below 2^33 m, so
// a difference of the inputs is kept to about n 2^-20 m. range_ss = n sum_j
// (max_i y_ij - min_i y_ij)^2 feeds residual_error_estimate. exponent = G
// (0 for all-constant X). False on nonfinite X.
bool centred_rows(const n4m_matrix_view_t& v, std::vector<double>& y, int& exponent,
                  long double& range_ss) {
    const auto n = static_cast<size_t>(v.rows), k = static_cast<size_t>(v.cols);
    std::vector<std::vector<double>> columns(k, std::vector<double>(n));
    std::vector<int> raw_exponent(k, 0), centred_exponent(k, 0);
    std::vector<bool> varies(k, false);
    for (size_t j = 0; j < k; ++j) {
        auto& col = columns[j];
        for (size_t i = 0; i < n; ++i) {
            col[i] = at(v, static_cast<int64_t>(i), static_cast<int64_t>(j));
            if (!std::isfinite(col[i])) return false;
        }
        raw_exponent[j] = bounding_exponent(col);
        for (double& x : col) x = std::ldexp(x, -raw_exponent[j]);
        center_in_place(col);
        for (double x : col)
            if (x != 0.0) varies[j] = true;
        if (varies[j]) centred_exponent[j] = raw_exponent[j] + bounding_exponent(col);
    }
    bool any = false;
    exponent = 0;
    for (size_t j = 0; j < k; ++j)
        if (varies[j]) {
            exponent = any ? std::max(exponent, centred_exponent[j]) : centred_exponent[j];
            any = true;
        }
    y.assign(n * k, 0.0);
    range_ss = 0.0L;
    for (size_t j = 0; j < k; ++j)
        if (varies[j]) {
            double low = 0.0, high = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double value = std::ldexp(columns[j][i], raw_exponent[j] - exponent);
                y[i * k + j] = value;
                low = std::min(low, value);
                high = std::max(high, value);
            }
            const long double range = ld(high) - ld(low);
            range_ss += static_cast<long double>(n) * range * range;
        }
    return true;
}

// Publishes non-negative values computed in the scaled unit (true value =
// scaled 2^shift). When every positive true value is a normal double, the
// true values are written and *exponent = 0. Otherwise (overflow, or a
// positive value below DBL_MIN, including one that would underflow to 0)
// the scaled values are written and *exponent = shift, so the true value is
// out * 2^exponent: a positive quantity is never published as 0. False only
// when a positive value is not representable even in the scaled unit (it is
// below 2^-1074 of the largest), which the caller reports as an error.
bool publish(const std::vector<long double>& scaled, int shift,
             const std::vector<double*>& out, int32_t& exponent) {
    bool normal = true;
    for (long double v : scaled) {
        const long double value = std::scalbn(v, shift);
        if (v > 0.0L && !(value >= static_cast<long double>(std::numeric_limits<double>::min()) &&
                          value <= static_cast<long double>(std::numeric_limits<double>::max())))
            normal = false;
    }
    exponent = normal ? 0 : shift;
    for (size_t i = 0; i < scaled.size(); ++i) {
        *out[i] = static_cast<double>(normal ? std::scalbn(scaled[i], shift) : scaled[i]);
        if (scaled[i] > 0.0L && !(*out[i] > 0.0)) return false;
    }
    return true;
}

// Term sum of squares and EXPLICIT residual sum of squares of the additive
// model strata + term, for rows of within-strata centred data xw. With D the
// term indicators and S the strata indicators, the reduced normal equations
// M = D'(I - P_S)D have M_ab = delta_ab n_a - sum_s n_as n_bs / n_s;
// dropping one reference level per connected component of the level-stratum
// graph leaves M_r positive definite. t_j = per-level sums of column j of xw
// (= D'(I - P_S) xw), beta_j = M_r^-1 t_j, SS_term = t' M_r^-1 t, and the
// residual of row i is xw_i - (beta_{a(i)} - sum_a n_{a s(i)} beta_a / n_s(i)),
// so the residual SS is a sum of squares, never a difference.
struct PermanovaKernel {
    const std::vector<int64_t>& level;
    const std::vector<int64_t>& stratum;
    const std::vector<double>& xw;
    size_t k;
    std::vector<int64_t> reduced;   // level -> row of M_r, -1 for references
    std::vector<double> chol;       // lower Cholesky factor of M_r, r x r
    size_t r = 0;
    // Per stratum: (level, count) of its occupied cells.
    std::vector<std::vector<std::pair<int64_t, int64_t>>> stratum_cells;
    std::vector<int64_t> stratum_n;
    std::vector<double> sums, rhs, effect, stratum_effect;

    // Returns {SS_term, SS_residual}; perm == nullptr is the observation.
    std::pair<long double, long double> operator()(const int64_t* perm) {
        const size_t n = level.size(), a = reduced.size(), s = stratum_n.size();
        std::fill(sums.begin(), sums.end(), 0.0);
        for (size_t i = 0; i < n; ++i) {
            const size_t src = static_cast<size_t>(perm ? perm[i] : static_cast<int64_t>(i));
            double* row = &sums[static_cast<size_t>(level[i]) * k];
            for (size_t j = 0; j < k; ++j) row[j] += xw[src * k + j];
        }
        long double ss_term = 0.0L;
        std::fill(effect.begin(), effect.end(), 0.0);
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < a; ++l)
                if (reduced[l] >= 0) rhs[static_cast<size_t>(reduced[l])] = sums[l * k + j];
            for (size_t u = 0; u < r; ++u) {  // L y = t
                long double v = ld(rhs[u]);
                for (size_t w = 0; w < u; ++w)
                    v -= ld(chol[u * r + w]) * ld(rhs[w]);
                rhs[u] = static_cast<double>(v / ld(chol[u * r + u]));
                ss_term += ld(rhs[u]) * ld(rhs[u]);
            }
            for (size_t u = r; u-- > 0;) {  // L' beta = y
                long double v = ld(rhs[u]);
                for (size_t w = u + 1; w < r; ++w)
                    v -= ld(chol[w * r + u]) * ld(rhs[w]);
                rhs[u] = static_cast<double>(v / ld(chol[u * r + u]));
            }
            for (size_t l = 0; l < a; ++l)
                if (reduced[l] >= 0) effect[l * k + j] = rhs[static_cast<size_t>(reduced[l])];
        }
        for (size_t g = 0; g < s; ++g)
            for (size_t j = 0; j < k; ++j) {
                long double m = 0.0L;
                for (const auto& [l, count] : stratum_cells[g])
                    m += static_cast<long double>(count) * ld(effect[static_cast<size_t>(l) * k + j]);
                stratum_effect[g * k + j] =
                    static_cast<double>(m / static_cast<long double>(stratum_n[g]));
            }
        long double ss_res = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            const size_t src = static_cast<size_t>(perm ? perm[i] : static_cast<int64_t>(i));
            const size_t l = static_cast<size_t>(level[i]), g = static_cast<size_t>(stratum[i]);
            for (size_t j = 0; j < k; ++j) {
                const long double e = static_cast<long double>(xw[src * k + j]) -
                                      ld(effect[l * k + j]) + ld(stratum_effect[g * k + j]);
                ss_res += e * e;
            }
        }
        return {ss_term, ss_res};
    }
};

// Appends to `basis` (orthonormal n-vectors) the direction of `candidate`
// if it is not already spanned (modified Gram-Schmidt with one
// reorthogonalization; dependent when the residual norm is <= 1e-9 of the
// candidate norm). Returns 1 when appended, 0 otherwise.
size_t extend_basis(std::vector<std::vector<double>>& basis, std::vector<double> v) {
    long double norm0 = 0.0L;
    for (double x : v) norm0 += ld(x) * ld(x);
    if (!(norm0 > 0.0L)) return 0;
    for (int pass = 0; pass < 2; ++pass)
        for (const auto& q : basis) {
            long double dot = 0.0L;
            for (size_t i = 0; i < v.size(); ++i) dot += ld(q[i]) * ld(v[i]);
            for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(ld(v[i]) - dot * ld(q[i]));
        }
    long double norm = 0.0L;
    for (double x : v) norm += ld(x) * ld(x);
    if (!(norm > 1e-18L * norm0)) return 0;
    const long double scale = 1.0L / std::sqrt(norm);
    for (double& x : v) x = static_cast<double>(ld(x) * scale);
    basis.push_back(std::move(v));
    return 1;
}

// Appends the indicator columns of dense codes (levels 0..count-1) one at a
// time; returns the rank increment.
size_t extend_with_indicators(std::vector<std::vector<double>>& basis,
                              const std::vector<int64_t>& dense, size_t levels) {
    size_t added = 0;
    for (size_t l = 0; l < levels; ++l) {
        std::vector<double> col(dense.size(), 0.0);
        for (size_t i = 0; i < dense.size(); ++i)
            if (static_cast<size_t>(dense[i]) == l) col[i] = 1.0;
        added += extend_basis(basis, std::move(col));
    }
    return added;
}

// Projects the columns of y (n x k) off the orthonormal basis (modified
// Gram-Schmidt order). Returns {squared norm of the coefficients of
// basis[first..), EXPLICIT residual sum of squares |y - sum_b q_b q_b' y|^2};
// work receives the residual.
std::pair<long double, long double> project(const std::vector<std::vector<double>>& basis,
                                            size_t first, const std::vector<double>& y,
                                            size_t k, std::vector<double>& work) {
    const size_t n = y.size() / k;
    work = y;
    long double tail = 0.0L;
    for (size_t b = 0; b < basis.size(); ++b)
        for (size_t j = 0; j < k; ++j) {
            long double dot = 0.0L;
            for (size_t i = 0; i < n; ++i)
                dot += ld(basis[b][i]) * ld(work[i * k + j]);
            if (b >= first) tail += dot * dot;
            for (size_t i = 0; i < n; ++i)
                work[i * k + j] = static_cast<double>(ld(work[i * k + j]) - dot * ld(basis[b][i]));
        }
    long double residual = 0.0L;
    for (double v : work) residual += ld(v) * ld(v);
    return {tail, residual};
}

constexpr long double max_energy_work = 2500000000.0L;
constexpr long double max_energy_pairs = 20000000.0L;
constexpr long double max_monitor_work = 10000000000.0L;
// Largest X (rows x cols) copied by the domain-shift kernels: 400 MB.
constexpr long double max_shift_cells = 50000000.0L;
// Largest n_groups^2 table of the energy kernel.
constexpr long double max_energy_table = 10000000.0L;
constexpr double unit_roundoff = std::numeric_limits<double>::epsilon() / 2.0;


// Euclidean norm of a - b: differences of domain values, squares summed in
// double-double (relative error of the norm <= 3u + O(u^2)).
double difference_norm(const double* a, const double* b, size_t k) noexcept {
    DD s;
    for (size_t j = 0; j < k; ++j) {
        const double d = a[j] - b[j];
        s = dd_add(s, d * d);
    }
    return std::sqrt(s.hi + s.lo);
}

// Relative error bound of difference_norm: difference, square, sum, sqrt.
constexpr double distance_gamma = 8.0 * unit_roundoff;

// Energy statistics of a labelling over a condensed distance matrix (pairs
// i < j, row-major) with double-double per-group-pair sums, so the
// combination E = 2M_ab - M_aa - M_bb keeps every bit of the sums.
struct EnergyKernel {
    const std::vector<double>& dist;
    const std::vector<int64_t>& group_n;
    int64_t n;
    std::vector<DD> sums;

    void accumulate(const int64_t* labels) {
        const auto g = group_n.size();
        std::fill(sums.begin(), sums.end(), DD{});
        size_t idx = 0;
        for (int64_t i = 0; i < n; ++i) {
            DD* row = sums.data() + static_cast<size_t>(labels[i]) * g;
            for (int64_t j = i + 1; j < n; ++j, ++idx) {
                DD& cell = row[static_cast<size_t>(labels[j])];
                cell = dd_add(cell, dist[idx]);
            }
        }
    }
    // Mean distance over all ordered pairs of groups a and b (V-statistic:
    // the zero self-pairs of a group are included).
    DD mean_distance(size_t a, size_t b) const {
        const auto g = group_n.size();
        const auto na = static_cast<double>(group_n[a]);
        const auto nb = static_cast<double>(group_n[b]);
        if (a == b) return dd_div(dd_mul(sums[a * g + a], 2.0), na * na);
        return dd_div(dd_add(sums[a * g + b], sums[b * g + a]), na * nb);
    }
    // E(a, b) in double-double, symmetric by construction (a < b order).
    DD energy(size_t a, size_t b) const {
        if (a > b) std::swap(a, b);
        return dd_add(dd_mul(mean_distance(a, b), 2.0),
                      dd_neg(dd_add(mean_distance(a, a), mean_distance(b, b))));
    }
    // Absolute error bound of E(a, b) from the distance rounding (relative
    // distance_gamma) and the double-double sums (2^-80 relative, n <= 2e7).
    double energy_bound(size_t a, size_t b) const {
        if (a > b) std::swap(a, b);
        const double scale = 2.0 * mean_distance(a, b).hi + mean_distance(a, a).hi +
                             mean_distance(b, b).hi;
        return (distance_gamma + 0x1p-80) * scale * (1.0 + 4.0 * unit_roundoff);
    }
    // Sum over group pairs of n_a n_b / (n_a + n_b) * E(a, b) and its bound.
    std::pair<DD, double> statistic() const {
        const auto g = group_n.size();
        DD total;
        double bound = 0.0;
        for (size_t a = 0; a < g; ++a) for (size_t b = a + 1; b < g; ++b) {
            const auto na = static_cast<double>(group_n[a]);
            const auto nb = static_cast<double>(group_n[b]);
            const double w = na * nb / (na + nb);
            total = dd_add(total, dd_mul(energy(a, b), w));
            bound += w * energy_bound(a, b);
        }
        return {total, bound * (1.0 + 4.0 * unit_roundoff)};
    }
};

// Column means in double-double of rows [0, n) of a row-major n x p buffer
// restricted to `rows` (all when empty).
std::vector<DD> dd_means(const std::vector<double>& x, size_t p, const std::vector<size_t>& rows) {
    std::vector<DD> mean(p);
    for (size_t i : rows)
        for (size_t j = 0; j < p; ++j) mean[j] = dd_add(mean[j], x[i * p + j]);
    for (DD& m : mean) m = dd_div(m, static_cast<double>(rows.size()));
    return mean;
}

// x - mean, with the double-double mean subtracted in two steps.
void centre(const double* x, const std::vector<DD>& mean, double* out) noexcept {
    for (size_t j = 0; j < mean.size(); ++j) out[j] = (x[j] - mean[j].hi) - mean[j].lo;
}

// Squared norm in double-double, rounded once.
double squared_norm(const double* v, size_t p) noexcept {
    DD s;
    for (size_t j = 0; j < p; ++j) s = dd_add(s, v[j] * v[j]);
    return s.hi + s.lo;
}

// Compensated dot product.
double dot(const double* a, const double* b, size_t p) noexcept {
    DD s;
    for (size_t j = 0; j < p; ++j) {
        const double prod = a[j] * b[j];
        s = dd_add(s, prod);
        s = dd_add(s, std::fma(a[j], b[j], -prod));
    }
    return s.hi + s.lo;
}

// T^2 and Q of one centred row in a reference PCA and the row's squared
// centred norm (the scale of its Q rounding).
struct RowStatistics {
    double t2 = 0.0, q = 0.0, norm2 = 0.0;
};

// PCA of centred reference rows (n4m_pca_fit) evaluated on its first k
// components.
struct ReferencePca {
    n4m_pca_fit_t fit{};
    int64_t rows = 0;
    std::vector<DD> mean;
    ~ReferencePca() { n4m_pca_fit_free(&fit); }
    ReferencePca() = default;
    ReferencePca(const ReferencePca&) = delete;
    ReferencePca& operator=(const ReferencePca&) = delete;

    // Rows `train` of the raw reference buffer x (row-major n x p).
    n4m_status_t compute(const std::vector<double>& x, size_t p, const std::vector<size_t>& train,
                         int64_t k) {
        rows = static_cast<int64_t>(train.size());
        mean = dd_means(x, p, train);
        std::vector<double> centred(train.size() * p);
        for (size_t r = 0; r < train.size(); ++r)
            centre(x.data() + train[r] * p, mean, centred.data() + r * p);
        const auto st = n4m_pca_fit(centred.data(), rows, static_cast<int64_t>(p), k, 1, &fit);
        if (st != N4M_OK) return st;
        if (fit.k != k) return N4M_ERR_NUMERICAL_FAILURE;
        for (int64_t a = 0; a < k; ++a)
            if (!std::isfinite(fit.explained_var[a])) return N4M_ERR_NUMERICAL_FAILURE;
        return N4M_OK;
    }
    // Eigenvalue resolution (rule 14): absolute error c m eps lambda_1 with
    // c = 16 and m = max(rows, cols); only components above it are resolved.
    double resolution() const {
        return 16.0 * static_cast<double>(std::max(rows, fit.cols)) *
               std::numeric_limits<double>::epsilon() * std::max(fit.explained_var[0], 0.0);
    }
    int64_t resolved_components() const {
        const double r = resolution();
        int64_t k = 0;
        while (k < fit.k && fit.explained_var[k] > r) ++k;
        return k;
    }
    RowStatistics evaluate(const double* xc, int64_t k, double* scores,
                           std::vector<double>& residual) const {
        const auto p = static_cast<size_t>(fit.cols);
        const auto kk = static_cast<size_t>(k);
        RowStatistics s;
        s.norm2 = squared_norm(xc, p);
        for (size_t a = 0; a < kk; ++a) {
            const double t = dot(xc, fit.components + a * p, p);
            scores[a] = t;
            s.t2 += t * t / fit.explained_var[a];
        }
        residual.assign(xc, xc + p);
        for (size_t a = 0; a < kk; ++a)
            for (size_t j = 0; j < p; ++j)
                residual[j] = std::fma(-scores[a], fit.components[a * p + j], residual[j]);
        s.q = squared_norm(residual.data(), p);
        if (k == 0) s.t2 = nan_value;
        return s;
    }
};

// Three-way comparison of value > limit with a tolerance: +1 above the limit
// by more than the tolerance, 0 at or below limit - tolerance (with a zero
// tolerance, equality is "not above"), -1 strictly within the tolerance
// band (numerically undetermined; never reported as a decision).
int compare_with_tolerance(double value, double limit, double tolerance) noexcept {
    if (value - limit > tolerance) return 1;
    if (limit - value >= tolerance) return 0;
    return -1;
}
}  // namespace

namespace {
// Shared body of n4m_preflight_pca_compute (2.19) and
// n4m_preflight_pca_compute_resolved (2.20). unresolved has k entries.
n4m_status_t pca_compute(n4m_context_t* ctx, const n4m_matrix_view_t* X, int32_t k,
                         const n4m_matrix_view_t* scores, const n4m_matrix_view_t* loadings,
                         double* eigenvalues, double* full_moments, double* resolution,
                         uint8_t* unresolved) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!eigenvalues || !full_moments || !resolution || !unresolved)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null PCA output");
    for (const auto* v : {X, scores, loadings}) {
        const auto st = check_view(ctx, v);
        if (st != N4M_OK) return st;
    }
    const int64_t n = X->rows, p = X->cols;
    if (n < 2 || p < 1 || k < 1 || k > std::min(n, p))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PCA dimensions or component count");
    if (scores->rows != n || scores->cols != k || loadings->rows != k || loadings->cols != p)
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH, "PCA output shape mismatch");
    // Guaranteed work and peak-memory bounds, checked before any allocation
    // (core/diagnostics/preflight_pca_budget.hpp lists every counted pass and
    // buffer; the QL iteration enters at the solver's ceiling, not at its
    // typical cost).
    {
        const auto budget = ::n4m::core::preflight_pca_budget(n, p, k);
        if (budget.work > ::n4m::core::preflight_pca_max_work)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "PCA exceeds the 2.4 billion operation budget (guaranteed bound)");
        if (budget.doubles > ::n4m::core::preflight_pca_max_doubles)
            return fail(ctx, N4M_ERR_UNSUPPORTED, "PCA working buffers exceed 1 GiB budget");
    }
    try {
        if (static_cast<long double>(n) * p > static_cast<long double>(SIZE_MAX / sizeof(double)))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "PCA matrix size overflow");
        const auto un = static_cast<size_t>(n), up = static_cast<size_t>(p);
        std::vector<double> data(un * up);
        for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j) {
            const double x = at(*X, i, j);
            if (!std::isfinite(x))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PCA input");
            data[static_cast<size_t>(i) * up + static_cast<size_t>(j)] = x;
        }
        {
            DomainRange range;
            for (double x : data) range.add(x);
            if (!range.inside() ||
                (range.high > 0.0 && (range.high > std::ldexp(1.0, pca_domain_exp) ||
                                      range.low < std::ldexp(1.0, -pca_domain_exp))))
                return fail(ctx, N4M_ERR_UNSUPPORTED,
                            "input outside the supported numerical domain (nonzero |x| in "
                            "[2^-128, 2^128], max/min nonzero |x| <= 2^32)");
        }
        // Centre every column in its own exactly rescaled space (max|x| in
        // [0.5, 1)), then bring all centred columns to one power-of-two scale
        // 2^g set by the largest centred variation. Constant columns centre
        // to exact zeros and do not influence g.
        int g = 0;
        bool varying = false;
        std::vector<int> col_exp(up, 0);
        for (size_t j = 0; j < up; ++j) {
            double maxabs = 0.0;
            for (size_t i = 0; i < un; ++i) maxabs = std::max(maxabs, std::fabs(data[i * up + j]));
            int e = 0;
            if (maxabs > 0.0) (void)std::frexp(maxabs, &e);
            // Two-term mean mean_hi + mean_lo from compensated (Neumaier)
            // sums: a plain sum loses small values next to large ones of
            // opposite sign, and a single rounded mean would shift every
            // centred value by up to ulp(offset).
            for (size_t i = 0; i < un; ++i) data[i * up + j] = std::ldexp(data[i * up + j], -e);
            const auto compensated_mean = [&](double shift) {
                double sum = 0.0, comp = 0.0;
                for (size_t i = 0; i < un; ++i) {
                    const double x = data[i * up + j] - shift;
                    const double t = sum + x;
                    comp += (std::fabs(sum) >= std::fabs(x)) ? (sum - t) + x : (x - t) + sum;
                    sum = t;
                }
                return (sum + comp) / static_cast<double>(n);
            };
            const double mean_hi = compensated_mean(0.0);
            const double mean_lo = compensated_mean(mean_hi);
            double cmax = 0.0;
            for (size_t i = 0; i < un; ++i) {
                data[i * up + j] = (data[i * up + j] - mean_hi) - mean_lo;
                cmax = std::max(cmax, std::fabs(data[i * up + j]));
            }
            col_exp[j] = e;
            if (cmax > 0.0) {
                int ce = 0;
                (void)std::frexp(cmax, &ce);
                g = varying ? std::max(g, e + ce) : e + ce;
                varying = true;
            }
        }
        for (size_t j = 0; j < up; ++j)
            for (size_t i = 0; i < un; ++i)
                data[i * up + j] = std::ldexp(data[i * up + j], col_exp[j] - g);

        // One Gram matrix on the smaller side, accumulated in long double,
        // gives both the exact full-spectrum moments of the rescaled
        // covariance (trace(C) = ||Xc||_F^2 / (n-1), trace(C^2) =
        // ||Gram||_F^2 / (n-1)^2) and the decomposition. A converged
        // symmetric QL eigensolver (n4m_symmetric_eigh, backward stable) on
        // that Gram is exact for every n, p and k: no randomized projection,
        // so every component meets the accuracy contract below.
        const bool wide = n <= p;  // Gram = Xc Xc^T (n x n), else Xc^T Xc (p x p)
        const size_t m = wide ? un : up;
        std::vector<double> gram(m * m);
        long double sum1 = 0.0L, sum2 = 0.0L;
        for (double x : data) sum1 += static_cast<long double>(x) * static_cast<long double>(x);
        for (size_t a = 0; a < m; ++a) for (size_t b = 0; b <= a; ++b) {
            long double dot = 0.0L;
            if (wide) {
                for (size_t j = 0; j < up; ++j)
                    dot += static_cast<long double>(data[a * up + j]) *
                           static_cast<long double>(data[b * up + j]);
            } else {
                for (size_t i = 0; i < un; ++i)
                    dot += static_cast<long double>(data[i * up + a]) *
                           static_cast<long double>(data[i * up + b]);
            }
            sum2 += (a == b ? 1.0L : 2.0L) * dot * dot;
            gram[a * m + b] = gram[b * m + a] = static_cast<double>(dot);
        }
        const long double d = static_cast<long double>(n - 1);
        // Inside the domain every rescaled output below is a normal double.
        const double m1 = std::ldexp(static_cast<double>(sum1 / d), 2 * g);
        const double m2 = std::ldexp(static_cast<double>(sum2 / (d * d)), 4 * g);

        std::vector<double> lam(m), vec(m * m);  // eigenvectors are the columns of vec
        const auto st = n4m_symmetric_eigh(gram.data(), static_cast<int64_t>(m), lam.data(),
                                           vec.data());
        if (st != N4M_OK) return fail(ctx, st, "PCA eigendecomposition failed");
        std::vector<size_t> order(m);
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t x, size_t y) { return lam[x] > lam[y]; });
        const double nm1 = static_cast<double>(n - 1);
        const auto variance = [&](int64_t a) {
            return std::max(lam[order[static_cast<size_t>(a)]], 0.0) / nm1;
        };
        // Accuracy contract (rule 14): variances carry an absolute error of
        // order max(n,p) * eps * lambda_max, plus the worst-case Gram
        // accumulation term documented in metrics.h. The operational
        // threshold pca_resolution_factor * max(n,p) * eps * lambda_max
        // separates resolved components; a component not above it is unresolved:
        // flagged, and published with a zero variance and zero scores (its
        // loading row stays a unit vector of the orthonormal basis). Decided
        // in the rescaled space, where every resolved value maps to a normal
        // double.
        const double res_scaled = pca_resolution_factor * static_cast<double>(std::max(n, p)) *
                                  std::numeric_limits<double>::epsilon() * variance(0);
        const auto uk = static_cast<size_t>(k);
        std::vector<double> load(uk * up, 0.0), sc(un * uk, 0.0);
        for (size_t a = 0; a < uk; ++a) {
            const auto ia = static_cast<int64_t>(a);
            const bool lost = !(variance(ia) > res_scaled);
            unresolved[a] = lost ? 1 : 0;
            eigenvalues[a] = lost ? 0.0 : std::ldexp(variance(ia), 2 * g);
            const size_t c = order[a];
            double* row = load.data() + a * up;
            if (!wide) {
                // Loadings are Gram eigenvectors; scores = Xc v.
                for (size_t j = 0; j < up; ++j) row[j] = vec[j * m + c];
                if (!lost)
                    for (size_t i = 0; i < un; ++i) {
                        long double s = 0.0L;
                        for (size_t j = 0; j < up; ++j)
                            s += static_cast<long double>(data[i * up + j]) *
                                 static_cast<long double>(row[j]);
                        sc[i * uk + a] = static_cast<double>(s);
                    }
                continue;
            }
            // Wide: scores = u sigma, loadings = Xc^T u / sigma (for an
            // unresolved component rounding noise, left 0); the rows are
            // orthonormalised and completed below.
            if (!lost) {
                const double sigma = std::sqrt(lam[c]);
                for (size_t i = 0; i < un; ++i) sc[i * uk + a] = vec[i * m + c] * sigma;
                for (size_t j = 0; j < up; ++j) {
                    long double s = 0.0L;
                    for (size_t i = 0; i < un; ++i)
                        s += static_cast<long double>(data[i * up + j]) *
                             static_cast<long double>(vec[i * m + c]);
                    row[j] = static_cast<double>(s / sigma);
                }
            }
        }
        // Wide input: the rows Xc^T u / sigma lose orthonormality as sigma
        // nears the resolution (error of order eps lambda_max / lambda), and
        // unresolved rows (a suffix, the variances being sorted) have none.
        // Resolved rows are re-orthonormalised by modified Gram-Schmidt
        // applied twice (stable for these nearly orthonormal rows; it only
        // combines resolved rows, so exact zeros such as constant columns
        // stay exact). Unresolved rows are Q e_r..Q e_{k-1}, with A =
        // [resolved rows]^T = Q R the Householder QR (p x r): unit and
        // orthogonal to the others to working precision, with no threshold and
        // no retries. Cost 2 r^2 p + p r^2 + 2 p r (k - r) <= 3 k^2 p.
        if (wide) {
            size_t r = 0;
            while (r < uk && !unresolved[r]) ++r;
            for (size_t a = 0; a < r; ++a) {
                double* row = load.data() + a * up;
                for (int pass = 0; pass < 2; ++pass)
                    for (size_t b = 0; b < a; ++b) {
                        const double* prev = load.data() + b * up;
                        double dot = 0.0;
                        for (size_t j = 0; j < up; ++j) dot += prev[j] * row[j];
                        for (size_t j = 0; j < up; ++j) row[j] -= dot * prev[j];
                    }
                double norm2 = 0.0;
                for (size_t j = 0; j < up; ++j) norm2 += row[j] * row[j];
                const double norm = std::sqrt(norm2);
                for (size_t j = 0; j < up; ++j) row[j] /= norm;
            }
            std::vector<double> house(load.data(), load.data() + r * up);
            std::vector<double> beta(r, 0.0);
            if (r < uk)
                for (size_t j = 0; j < r; ++j) {
                    // Reflector j, stored in house[j*up + j ..], maps column j
                    // of A (below its first j entries) onto a multiple of e_j.
                    double* v = house.data() + j * up;
                    double norm2 = 0.0;
                    for (size_t i = j; i < up; ++i) norm2 += v[i] * v[i];
                    if (norm2 == 0.0) continue;
                    const double alpha = v[j] >= 0.0 ? -std::sqrt(norm2) : std::sqrt(norm2);
                    v[j] -= alpha;
                    double vv = 0.0;
                    for (size_t i = j; i < up; ++i) vv += v[i] * v[i];
                    beta[j] = 2.0 / vv;
                    for (size_t t = j + 1; t < r; ++t) {
                        double* w = house.data() + t * up;
                        double dot = 0.0;
                        for (size_t i = j; i < up; ++i) dot += v[i] * w[i];
                        dot *= beta[j];
                        for (size_t i = j; i < up; ++i) w[i] -= dot * v[i];
                    }
                }
            for (size_t a = r; a < uk; ++a) {
                double* row = load.data() + a * up;
                std::fill(row, row + up, 0.0);
                row[a] = 1.0;
                for (size_t j = r; j-- > 0;) {
                    if (beta[j] == 0.0) continue;
                    const double* v = house.data() + j * up;
                    double dot = 0.0;
                    for (size_t i = j; i < up; ++i) dot += v[i] * row[i];
                    dot *= beta[j];
                    for (size_t i = j; i < up; ++i) row[i] -= dot * v[i];
                }
            }
        }
        // Sign convention (as n4m_pca_fit / sklearn): the largest-magnitude
        // loading of each component is non-negative; scores follow.
        for (size_t a = 0; a < uk; ++a) {
            double* row = load.data() + a * up;
            size_t arg = 0;
            for (size_t j = 1; j < up; ++j)
                if (std::fabs(row[j]) > std::fabs(row[arg])) arg = j;
            const double sign = row[arg] < 0.0 ? -1.0 : 1.0;
            for (size_t j = 0; j < up; ++j)
                put(*loadings, static_cast<int64_t>(a), static_cast<int64_t>(j), sign * row[j]);
            for (size_t i = 0; i < un; ++i)
                put(*scores, static_cast<int64_t>(i), static_cast<int64_t>(a),
                    std::ldexp(sign * sc[i * uk + a], g) + 0.0);  // no -0
        }
        resolution[0] = std::ldexp(res_scaled, 2 * g);
        full_moments[0] = varying ? m1 : 0.0;
        full_moments[1] = varying ? m2 : 0.0;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PCA allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PCA internal error"); }
}
}  // namespace

extern "C" {
N4M_API n4m_status_t n4m_preflight_pca_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, int32_t k,
    const n4m_matrix_view_t* scores, const n4m_matrix_view_t* loadings,
    double* eigenvalues, double* full_moments) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    // Validate k before sizing the flag buffer (pca_compute repeats it).
    if (k < 1 || k > std::min(X->rows, X->cols))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PCA dimensions or component count");
    try {
        double resolution = 0.0;
        std::vector<uint8_t> unresolved(static_cast<size_t>(k));
        return pca_compute(ctx, X, k, scores, loadings, eigenvalues, full_moments, &resolution,
                           unresolved.data());
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PCA allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PCA internal error"); }
}

N4M_API n4m_status_t n4m_preflight_pca_compute_resolved(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, int32_t k,
    const n4m_matrix_view_t* scores, const n4m_matrix_view_t* loadings,
    double* eigenvalues, double* full_moments, double* resolution, uint8_t* unresolved) {
    return pca_compute(ctx, X, k, scores, loadings, eigenvalues, full_moments, resolution,
                       unresolved);
}

N4M_API n4m_status_t n4m_preflight_repeatability_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    const uint8_t* valid_mask, double* pooled_sd, double* icc,
    int64_t* n_complete_groups, int32_t* band_status) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_ids || !pooled_sd || !icc || !n_complete_groups || !band_status)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null repeatability input or output");
    if (X->rows < 1 || X->cols < 1)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "empty repeatability matrix");
    if (static_cast<uint64_t>(X->rows) > SIZE_MAX / static_cast<uint64_t>(X->cols))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "repeatability mask size overflow");
    try {
        for (int64_t j = 0; j < X->cols; ++j) {
            double scale = 0.0;
            for (int64_t i = 0; i < X->rows; ++i) {
                if (valid_mask &&
                    !valid_mask[static_cast<size_t>(i) * static_cast<size_t>(X->cols) +
                                static_cast<size_t>(j)]) continue;
                const double x = at(*X, i, j);
                if (!std::isfinite(x))
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                                "nonfinite unmasked repeatability input");
                scale = std::max(scale, std::abs(x));
            }
            if (scale == 0.0) scale = 1.0;
            struct Group { int64_t count = 0; long double mean = 0, m2 = 0; };
            std::unordered_map<int64_t, Group> groups;
            for (int64_t i = 0; i < X->rows; ++i) {
                if (valid_mask &&
                    !valid_mask[static_cast<size_t>(i) * static_cast<size_t>(X->cols) +
                                static_cast<size_t>(j)]) continue;
                const long double x = static_cast<long double>(at(*X, i, j)) /
                                      static_cast<long double>(scale);
                auto& g = groups[group_ids[i]];
                ++g.count;
                const long double delta = x - g.mean;
                g.mean += delta / g.count;
                g.m2 += delta * (x - g.mean);
            }
            n_complete_groups[j] = static_cast<int64_t>(groups.size());
            pooled_sd[j] = nan_value; icc[j] = nan_value; band_status[j] = 1;
            long double within = 0, grand = 0, between = 0;
            int64_t df_within = 0, total = 0, r = -1;
            bool balanced = true;
            for (const auto& entry : groups) {
                const auto& g = entry.second;
                if (r < 0) r = g.count;
                else if (g.count != r) balanced = false;
                if (g.count >= 2) {
                    within += std::max(0.0L, g.m2);
                    df_within += g.count - 1;
                }
                grand += g.mean * g.count; total += g.count;
            }
            if (df_within > 0) {
                pooled_sd[j] = static_cast<double>(
                    static_cast<long double>(scale) * std::sqrt(within / df_within));
                if (!std::isfinite(pooled_sd[j]))
                    return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                "repeatability SD is not representable as f64");
            }
            if (groups.size() < 2 || df_within == 0) continue;
            if (!balanced || r < 2) { band_status[j] = 2; continue; }
            const long double mean = grand / total;
            for (const auto& entry : groups) {
                const auto& g = entry.second;
                const long double delta = g.mean - mean;
                between += g.count * delta * delta;
            }
            const long double msb = between / (groups.size() - 1);
            const long double msw = within / df_within;
            const long double denom = msb + (r - 1) * msw;
            if (denom <= 0) { band_status[j] = 3; continue; }
            icc[j] = static_cast<double>((msb - msw) / denom);
            band_status[j] = 0;
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "repeatability allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "repeatability internal error"); }
}

N4M_API n4m_status_t n4m_preflight_summary_compute(
    n4m_context_t* ctx, const double* values, const uint8_t* valid_mask,
    int64_t n, double* out_eight) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!values || !out_eight) return fail(ctx, N4M_ERR_NULL_POINTER, "null summary buffer");
    if (n < 0) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "negative summary length");
    try {
        std::vector<double> v;
        DomainRange scan;
        for (int64_t i = 0; i < n; ++i) {
            if (valid_mask && !valid_mask[i]) continue;
            if (!std::isfinite(values[i]))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked summary input");
            scan.add(values[i]);
            v.push_back(values[i]);
        }
        // Summary: magnitude bound only. Mean, SD and quartiles are correct
        // for any spread inside it (lead arbitration of rule 13).
        if (!scan.magnitudes_inside())
            return fail(ctx, N4M_ERR_UNSUPPORTED, magnitude_domain_message);
        std::fill(out_eight, out_eight + 8, nan_value);
        out_eight[0] = static_cast<double>(v.size());
        if (v.empty()) return N4M_OK;
        const MeanSquares moments = mean_and_squares(v);
        std::sort(v.begin(), v.end());
        out_eight[1] = v.front(); out_eight[2] = v.back();
        out_eight[3] = moments.mean;
        if (v.size() >= 2)
            out_eight[4] = std::sqrt(moments.ss / static_cast<double>(v.size() - 1));
        out_eight[5] = quantile(v, 0.25);
        out_eight[6] = quantile(v, 0.5);
        out_eight[7] = quantile(v, 0.75);
        for (int i = 1; i < 8; ++i) {
            if (i == 4 && v.size() < 2) continue;
            if (!std::isfinite(out_eight[i]))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "summary result is not representable as f64");
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "summary allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "summary internal error"); }
}

N4M_API n4m_status_t n4m_preflight_nearest_neighbors(
    n4m_context_t* ctx, const n4m_matrix_view_t* query,
    const n4m_matrix_view_t* reference, const int64_t* query_groups,
    const int64_t* reference_groups, int32_t k, int64_t* indices,
    double* distances) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    for (const auto* v : {query, reference}) {
        const auto st = check_view(ctx, v);
        if (st != N4M_OK) return st;
    }
    if (!indices || !distances) return fail(ctx, N4M_ERR_NULL_POINTER, "null neighbor output");
    if (query->cols != reference->cols || query->rows < 1 || reference->rows < 1 ||
        query->cols < 1 || k < 1)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid neighbor dimensions");
    if ((query_groups == nullptr) != (reference_groups == nullptr))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "both neighbor group arrays are required");
    if (static_cast<uint64_t>(query->rows) >
        static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)) / static_cast<uint64_t>(k))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "neighbor output size overflow");
    try {
        for (const auto* v : {query, reference})
            for (int64_t i = 0; i < v->rows; ++i) {
                DomainRange scan;
                for (int64_t j = 0; j < v->cols; ++j) {
                    if (!std::isfinite(at(*v, i, j)))
                        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite neighbor input");
                    scan.add(at(*v, i, j));
                }
                // Magnitude bound only (lead decision): compensated distances
                // and the exact ordering near ties are correct for any spread,
                // e.g. rounding-level PCA score components next to large ones.
                if (!scan.magnitudes_inside())
                    return fail(ctx, N4M_ERR_UNSUPPORTED, magnitude_domain_message);
            }
        // Euclidean distance: compensated sum of squared differences (inside
        // the domain no difference or square overflows or underflows),
        // relative error below 4u. Candidates are ordered by distance, then
        // by reference index. When two distances are within 2^-45 (relative)
        // the order is decided exactly from sum_j (r_aj^2 - r_bj^2) -
        // 2 q_j (r_aj - r_bj) in an exact accumulator, so equal true distances
        // always tie-break by index and distinct ones never swap.
        const auto distance = [&](int64_t i, int64_t h) {
            double sum = 0.0, comp = 0.0;
            for (int64_t j = 0; j < query->cols; ++j) {
                const double d = at(*query, i, j) - at(*reference, h, j);
                const double sq = d * d, next = sum + sq;
                comp += sum >= sq ? (sum - next) + sq : (sq - next) + sum;
                sum = next;
            }
            return std::sqrt(sum + comp);
        };
        const auto closer = [&](int64_t i, int64_t a, double da, int64_t b, double db) {
            if (da < db * (1.0 - 0x1p-45)) return true;
            if (da > db * (1.0 + 0x1p-45)) return false;
            ::n4m::core::ExactSum diff;  // ||q - r_a||^2 - ||q - r_b||^2
            for (int64_t j = 0; j < query->cols; ++j) {
                const double q = at(*query, i, j), ra = at(*reference, a, j), rb = at(*reference, b, j);
                diff.add_product(ra, ra, false);
                diff.add_product(rb, rb, true);
                diff.add_product(q, ra, true);
                diff.add_product(q, ra, true);
                diff.add_product(q, rb, false);
                diff.add_product(q, rb, false);
            }
            const int sign = diff.sign();
            return sign < 0 || (sign == 0 && a < b);
        };
        for (int64_t i = 0; i < query->rows; ++i) {
            for (int32_t a = 0; a < k; ++a) {
                indices[i * k + a] = -1; distances[i * k + a] = nan_value;
            }
            for (int64_t h = 0; h < reference->rows; ++h) {
                if (query_groups && query_groups[i] == reference_groups[h]) continue;
                const double d = distance(i, h);
                int32_t pos = 0;
                while (pos < k && indices[i * k + pos] >= 0 &&
                       closer(i, indices[i * k + pos], distances[i * k + pos], h, d)) ++pos;
                if (pos == k) continue;
                for (int32_t a = k - 1; a > pos; --a) {
                    indices[i * k + a] = indices[i * k + a - 1];
                    distances[i * k + a] = distances[i * k + a - 1];
                }
                indices[i * k + pos] = h; distances[i * k + pos] = d;
            }
        }
        return N4M_OK;
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "neighbor internal error"); }
}

N4M_API n4m_status_t n4m_preflight_local_contrast(
    n4m_context_t* ctx, const double* targets, const int64_t* neighbor_indices,
    const int64_t* group_ids, int64_t n, int32_t k, int32_t categorical,
    double* out_four) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!targets || !neighbor_indices || !out_four)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null local contrast buffer");
    if (n < 1 || k < 1 || (categorical != 0 && categorical != 1) ||
        static_cast<uint64_t>(n) >
            static_cast<uint64_t>(PTRDIFF_MAX / sizeof(int64_t)) / static_cast<uint64_t>(k))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid local contrast dimensions");
    try {
        std::vector<int64_t> categories;
        if (categorical) categories.resize(static_cast<size_t>(n));
        constexpr double max_exact_category = 9007199254740991.0;
        DomainRange scan;
        for (int64_t i = 0; i < n; ++i) {
            const double y = targets[i];
            if (!std::isfinite(y))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite target");
            if (categorical) {
                if (std::abs(y) > max_exact_category || std::trunc(y) != y)
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                                "category codes must be exact integers");
                categories[static_cast<size_t>(i)] = static_cast<int64_t>(y);
            } else {
                scan.add(y);
            }
        }
        // Continuous targets follow the numerical domain; category codes are
        // labels compared for equality only.
        if (!categorical && !scan.inside())
            return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
        // Sum of squared deviations from a double-double mean (compensated).
        const double m2 = categorical ? 0.0
            : mean_and_squares(std::vector<double>(targets, targets + n)).ss;
        std::unordered_map<int64_t, int64_t> category_counts, group_counts;
        std::unordered_map<std::pair<int64_t, int64_t>, int64_t,
                           GroupCategoryHash> joint_counts;
        if (categorical) {
            for (int64_t i = 0; i < n; ++i) {
                const int64_t cat = categories[static_cast<size_t>(i)];
                ++category_counts[cat];
                if (group_ids) {
                    ++group_counts[group_ids[i]];
                    ++joint_counts[{group_ids[i], cat}];
                }
            }
        }
        // Neumaier sums in double (same result on every host); each target
        // difference is a single rounding, its square relative 2u.
        double row_sum = 0.0, row_comp = 0.0, expected_sum = 0.0, expected_comp = 0.0;
        const auto add = [](double& sum, double& comp, double x) {
            const double next = sum + x;
            comp += std::fabs(sum) >= std::fabs(x) ? (sum - next) + x : (x - next) + sum;
            sum = next;
        };
        int64_t eligible_rows = 0;
        for (int64_t i = 0; i < n; ++i) {
            double local = 0.0, local_comp = 0.0;
            int32_t found = 0;
            for (int32_t a = 0; a < k; ++a) {
                const int64_t j = neighbor_indices[i * k + a];
                if (j == -1) continue;
                if (j < 0 || j >= n || j == i ||
                    (group_ids && group_ids[i] == group_ids[j]))
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                                "invalid or nonindependent neighbor index");
                if (categorical) {
                    add(local, local_comp, categories[static_cast<size_t>(i)] !=
                                           categories[static_cast<size_t>(j)] ? 1.0 : 0.0);
                } else {
                    const double delta = targets[i] - targets[j];
                    add(local, local_comp, delta * delta);
                }
                ++found;
            }
            if (found > 0) {
                add(row_sum, row_comp, (local + local_comp) / found);
                ++eligible_rows;
                if (categorical) {
                    const int64_t cat = categories[static_cast<size_t>(i)];
                    const int64_t admissible = group_ids
                        ? n - group_counts.at(group_ids[i]) : n - 1;
                    const int64_t same = group_ids
                        ? category_counts.at(cat) - joint_counts.at({group_ids[i], cat})
                        : category_counts.at(cat) - 1;
                    if (admissible <= 0)
                        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                                    "neighbor has no admissible population");
                    add(expected_sum, expected_comp,
                        1.0 - static_cast<double>(same) / static_cast<double>(admissible));
                }
            }
        }
        out_four[0] = nan_value;
        out_four[1] = static_cast<double>(eligible_rows) / static_cast<double>(n);
        out_four[2] = nan_value;
        out_four[3] = nan_value;
        if (eligible_rows == 0) return N4M_OK;
        const double rows = static_cast<double>(eligible_rows);
        const double observed_total = row_sum + row_comp, expected_total = expected_sum + expected_comp;
        if (!categorical) {
            if (n < 2 || m2 <= 0) return N4M_OK;
            // 1 - observed / (2 s^2): the ratio has relative error ~6u, so the
            // result carries an absolute error of a few u times the ratio.
            const double variance = m2 / static_cast<double>(n - 1);
            out_four[0] = 1.0 - (observed_total / rows) / (2.0 * variance);
        } else {
            out_four[2] = 1.0 - observed_total / rows;
            out_four[3] = 1.0 - expected_total / rows;
            if (expected_total <= 0) return N4M_OK;
            out_four[0] = 1.0 - observed_total / expected_total;
        }
        if (!std::isfinite(out_four[0]))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "local contrast overflow");
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "local contrast allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "local contrast internal error"); }
}
N4M_API n4m_status_t n4m_cluster_hdbscan(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, int64_t min_cluster_size,
    int64_t min_samples, int32_t selection_method, int32_t allow_single_cluster,
    int64_t* labels, double* membership, double* outlier_scores, int64_t* n_clusters,
    double* cluster_persistence, int64_t* n_tree_nodes, int64_t* tree_parent,
    int64_t* tree_size, int64_t* tree_label, double* tree_lambda_birth,
    double* tree_lambda_death, double* tree_stability) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!labels || !membership || !outlier_scores || !n_clusters || !cluster_persistence ||
        !n_tree_nodes || !tree_parent || !tree_size || !tree_label || !tree_lambda_birth ||
        !tree_lambda_death || !tree_stability)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null HDBSCAN output");
    const int64_t n = X->rows, p = X->cols;
    if (n < 2 || p < 1)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "HDBSCAN needs at least two rows and one column");
    if (min_cluster_size < 2 || min_samples < 1 || min_samples > n)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "HDBSCAN needs min_cluster_size>=2 and 1<=min_samples<=rows");
    if ((selection_method != 0 && selection_method != 1) ||
        (allow_single_cluster != 0 && allow_single_cluster != 1))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid HDBSCAN selection flags");
    // Working memory: the data copy plus about 512 bytes per row of
    // distances, MST, linkage, condensed tree and per-point bookkeeping.
    if (static_cast<long double>(n) * n * p > max_hdbscan_products ||
        8.0L * static_cast<long double>(n) * p + 512.0L * static_cast<long double>(n) >
            max_working_bytes)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "HDBSCAN exceeds its 2.5e9 distance-product or 256 MiB memory budget");
    DomainRange domain;
    for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j) {
        const double x = at(*X, i, j);
        if (!std::isfinite(x)) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite HDBSCAN input");
        domain.add(x);
    }
    if (!domain.inside()) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        std::vector<double> data(static_cast<size_t>(n) * static_cast<size_t>(p));
        for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j)
            data[static_cast<size_t>(i) * static_cast<size_t>(p) + static_cast<size_t>(j)] =
                at(*X, i, j);
        ::n4m::core::hdbscan::Params params;
        params.min_cluster_size = min_cluster_size;
        params.min_samples = min_samples;
        params.selection = static_cast<::n4m::core::hdbscan::Selection>(selection_method);
        params.allow_single_cluster = allow_single_cluster != 0;
        const auto result = ::n4m::core::hdbscan::fit(data.data(), n, p, params);
        // Every output has X.rows capacity; check before writing anything.
        const auto rows = static_cast<size_t>(n);
        if (result.error != ::n4m::core::hdbscan::Error::none || result.labels.size() != rows || result.membership.size() != rows ||
            result.outlier_scores.size() != rows || result.persistence.size() > rows ||
            result.tree.empty() || result.tree.size() > rows)
            return fail(ctx, N4M_ERR_INTERNAL, "HDBSCAN result violates its size invariants");
        for (size_t k = 0; k < result.tree.size(); ++k) {
            const auto parent = result.tree[k].parent;
            if ((k == 0) != (parent < 0) || parent >= static_cast<int64_t>(k))
                return fail(ctx, N4M_ERR_INTERNAL, "HDBSCAN condensed tree is not ordered");
        }
        for (size_t i = 0; i < result.labels.size(); ++i) {
            labels[i] = result.labels[i];
            membership[i] = result.membership[i];
            outlier_scores[i] = result.outlier_scores[i];
        }
        *n_clusters = static_cast<int64_t>(result.persistence.size());
        std::copy(result.persistence.begin(), result.persistence.end(), cluster_persistence);
        *n_tree_nodes = static_cast<int64_t>(result.tree.size());
        for (size_t k = 0; k < result.tree.size(); ++k) {
            const auto& node = result.tree[k];
            tree_parent[k] = node.parent;
            tree_size[k] = node.size;
            tree_label[k] = node.label;
            tree_lambda_birth[k] = node.lambda_birth;
            tree_lambda_death[k] = node.lambda_death;
            tree_stability[k] = node.stability;
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "HDBSCAN allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "HDBSCAN internal error"); }
}

N4M_API n4m_status_t n4m_preflight_adjusted_rand_index(
    n4m_context_t* ctx, const int64_t* labels_a, const int64_t* labels_b, int64_t n,
    double* ari) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!labels_a || !labels_b || !ari) return fail(ctx, N4M_ERR_NULL_POINTER, "null ARI buffer");
    if (n < 0) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "negative ARI length");
    // Working memory: label pairs and two label copies, 32 bytes per element.
    if (32.0L * static_cast<long double>(n) > max_working_bytes)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "ARI exceeds its 256 MiB memory budget");
    try {
        *ari = nan_value;
        if (n < 2) return N4M_OK;
        std::vector<std::pair<int64_t, int64_t>> pairs(static_cast<size_t>(n));
        std::vector<int64_t> a(labels_a, labels_a + n), b(labels_b, labels_b + n);
        for (int64_t i = 0; i < n; ++i) pairs[static_cast<size_t>(i)] = {labels_a[i], labels_b[i]};
        // Sum of squared run lengths of a sorted sequence (exact: n <= 2^31).
        auto squared_runs = [](auto& values) {
            std::sort(values.begin(), values.end());
            uint64_t total = 0;
            for (size_t i = 0; i < values.size();) {
                size_t j = i;
                while (j < values.size() && values[j] == values[i]) ++j;
                total += static_cast<uint64_t>(j - i) * static_cast<uint64_t>(j - i);
                i = j;
            }
            return total;
        };
        // scikit-learn pair_confusion_matrix / adjusted_rand_score with exact
        // integer pair counts and an exact 128-bit numerator and denominator,
        // so a near-zero ARI never comes from cancelling rounded products.
        const uint64_t squares = squared_runs(pairs);
        const uint64_t rows = squared_runs(a), cols = squared_runs(b);
        const auto total = static_cast<uint64_t>(n);
        const uint64_t tp = squares - total, fp = cols - squares, fn = rows - squares;
        const uint64_t tn = total * total - fp - fn - squares;
        const U128 positive = mul_u64(tp, tn), negative = mul_u64(fn, fp);
        const U128 denominator = add_u128(mul_u64(tp + fn, fn + tn), mul_u64(tp + fp, fp + tn));
        if (denominator.hi == 0 && denominator.lo == 0) return N4M_OK;
        const bool sign = less_u128(positive, negative);
        const U128 numerator =
            sign ? sub_u128(negative, positive) : sub_u128(positive, negative);
        const long double ratio = 2.0L * to_long_double(numerator) / to_long_double(denominator);
        *ari = static_cast<double>(sign ? -ratio : ratio);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "ARI allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "ARI internal error"); }
}
N4M_API n4m_status_t n4m_cluster_jaccard_matrix(
    n4m_context_t* ctx, const int64_t* labels_a, const int64_t* labels_b, int64_t n,
    int64_t n_clusters_a, int64_t n_clusters_b, double* jaccard, int64_t* intersections,
    int64_t* sizes_a, int64_t* sizes_b) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (n < 0 || n_clusters_a < 0 || n_clusters_b < 0)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "negative Jaccard dimension");
    if (n > (int64_t{1} << 53))
        return fail(ctx, N4M_ERR_UNSUPPORTED, "Jaccard counts above 2^53 are not exact doubles");
    if (static_cast<long double>(n_clusters_a) * static_cast<long double>(n_clusters_b) >
        100000000.0L)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "Jaccard matrix exceeds 100 million cells");
    const auto ka = static_cast<size_t>(n_clusters_a), kb = static_cast<size_t>(n_clusters_b);
    if ((n > 0 && (!labels_a || !labels_b)) || (ka * kb > 0 && (!jaccard || !intersections)) ||
        (ka > 0 && !sizes_a) || (kb > 0 && !sizes_b))
        return fail(ctx, N4M_ERR_NULL_POINTER, "null Jaccard buffer");
    for (int64_t i = 0; i < n; ++i)
        if (labels_a[i] >= n_clusters_a || labels_b[i] >= n_clusters_b)
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "cluster label out of range");
    std::fill(intersections, intersections + ka * kb, int64_t{0});
    std::fill(sizes_a, sizes_a + ka, int64_t{0});
    std::fill(sizes_b, sizes_b + kb, int64_t{0});
    for (int64_t i = 0; i < n; ++i) {
        const int64_t a = labels_a[i], b = labels_b[i];
        if (a >= 0) ++sizes_a[a];
        if (b >= 0) ++sizes_b[b];
        if (a >= 0 && b >= 0)
            ++intersections[static_cast<size_t>(a) * kb + static_cast<size_t>(b)];
    }
    for (size_t i = 0; i < ka; ++i) for (size_t j = 0; j < kb; ++j) {
        const int64_t both = intersections[i * kb + j];
        const int64_t either = sizes_a[i] + sizes_b[j] - both;
        jaccard[i * kb + j] = either > 0
            ? static_cast<double>(both) / static_cast<double>(either) : nan_value;
    }
    return N4M_OK;
}

N4M_API n4m_status_t n4m_linear_sum_assignment(
    n4m_context_t* ctx, const n4m_matrix_view_t* cost, int32_t maximize, int64_t* row_ind,
    int64_t* col_ind, int64_t* n_assigned) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, cost);
    if (st != N4M_OK) return st;
    if (!row_ind || !col_ind || !n_assigned)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null assignment output");
    if (maximize != 0 && maximize != 1)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "maximize must be 0 or 1");
    const int64_t rows = cost->rows, cols = cost->cols;
    // Work: rows*cols*min(rows, cols) exact 192-bit additions and
    // comparisons, at most 1e8. Memory, checked before any allocation: two
    // copies of the costs (16 bytes per cell), three 32-byte exact values and
    // four 8-byte indices per row or column.
    const long double cells = static_cast<long double>(rows) * static_cast<long double>(cols);
    if (cells * static_cast<long double>(std::min(rows, cols)) > 100000000.0L ||
        16.0L * cells + 128.0L * static_cast<long double>(rows + cols) > max_working_bytes)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "assignment exceeds its 1e8 exact-operation or 256 MiB memory budget");
    // NaN and -inf (after negation when maximizing) are invalid as in SciPy;
    // +inf marks a forbidden pair and is not a numerical value.
    DomainRange domain;
    for (int64_t i = 0; i < rows; ++i) for (int64_t j = 0; j < cols; ++j) {
        const double v = at(*cost, i, j);
        if (std::isnan(v) || (std::isinf(v) && (v > 0) == (maximize != 0)))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "assignment cost contains NaN or -inf");
        if (std::isfinite(v)) domain.add(v);
    }
    if (!domain.inside()) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        std::vector<double> data(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int64_t i = 0; i < rows; ++i) for (int64_t j = 0; j < cols; ++j)
            data[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j)] =
                at(*cost, i, j);
        *n_assigned = 0;
        const auto status = ::n4m::core::lsap::solve(static_cast<size_t>(rows),
                                                      static_cast<size_t>(cols), data.data(),
                                                      maximize != 0, row_ind, col_ind);
        if (status == ::n4m::core::lsap::Status::invalid)
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "assignment cost contains NaN or -inf");
        if (status == ::n4m::core::lsap::Status::infeasible)
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "assignment cost matrix is infeasible");
        if (status == ::n4m::core::lsap::Status::overflow)
            return fail(ctx, N4M_ERR_INTERNAL, "assignment exact-arithmetic bound exceeded");
        *n_assigned = std::min(rows, cols);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "assignment allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "assignment internal error"); }
}

N4M_API n4m_status_t n4m_random_permutations(
    n4m_context_t* ctx, uint64_t seed, int64_t n, int64_t n_permutations, int64_t* out) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (n < 0 || n_permutations < 0)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "negative permutation dimension");
    if (static_cast<long double>(n) * n_permutations > max_moment_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "permutations exceed 250 million entries");
    if (n > 0 && n_permutations > 0 && !out)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null permutation output");
    n4m_rng_pcg64 rng{};
    n4m_pcg64_engine_seed(&rng, seed);
    for (int64_t b = 0; b < n_permutations; ++b)
        n4m_pcg64_engine_permutation(&rng, out + b * n, n);
    return N4M_OK;
}

N4M_API n4m_status_t n4m_random_multivariate_normal(
    n4m_context_t* ctx, uint64_t seed, const double* mean, const n4m_matrix_view_t* cov,
    int64_t n, double* out) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, cov);
    if (st != N4M_OK) return st;
    const int64_t d = cov->rows;
    if (d < 1 || cov->cols != d || n < 0)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "covariance must be square and n >= 0");
    if (!mean || (n > 0 && !out))
        return fail(ctx, N4M_ERR_NULL_POINTER, "null multivariate normal buffer");
    // Checked before any size_t product: the Cholesky factor (d*d doubles)
    // and its d^3/6 cost are bounded independently of n, then the draws.
    const long double dd = static_cast<long double>(d) * static_cast<long double>(d);
    const long double draws = static_cast<long double>(n) * static_cast<long double>(d);
    if (8.0L * (dd + draws) > max_working_bytes ||
        dd * static_cast<long double>(d) / 6.0L > max_hdbscan_products ||
        draws * static_cast<long double>(d) > max_hdbscan_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "multivariate normal exceeds its 256 MiB memory or 2.5e9 operation budget");
    // Mean and covariance (lower triangle) are each checked against the
    // rule-13 domain before any computation.
    DomainRange mean_domain, cov_domain;
    for (int64_t j = 0; j < d; ++j) {
        if (!std::isfinite(mean[j])) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite mean");
        mean_domain.add(mean[j]);
        for (int64_t k = 0; k <= j; ++k) {
            const double c = at(*cov, j, k);
            if (!std::isfinite(c)) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite covariance");
            cov_domain.add(c);
        }
    }
    if (!mean_domain.inside() || !cov_domain.inside())
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        const auto p = static_cast<size_t>(d);
        std::vector<double> L(p * p, 0.0);
        // The factor is computed on cov / 4^m, with 2^m near the largest
        // standard deviation, and the draws are rescaled by 2^m. Power-of-two
        // scaling is exact, so ordinary inputs give bit-identical draws, while
        // tiny or huge covariances neither underflow nor overflow inside the
        // factorization. Positive definiteness is tested in the scaled space.
        double largest_variance = 0.0;
        for (size_t j = 0; j < p; ++j)
            largest_variance = std::max(
                largest_variance, std::fabs(at(*cov, static_cast<int64_t>(j), static_cast<int64_t>(j))));
        int variance_exponent = 0;
        std::frexp(largest_variance, &variance_exponent);
        const int half = variance_exponent / 2;
        // Lower Cholesky factor (Cholesky-Banachiewicz) of the lower triangle.
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k <= j; ++k) {
                double sum = std::ldexp(
                    at(*cov, static_cast<int64_t>(j), static_cast<int64_t>(k)), -2 * half);
                for (size_t m = 0; m < k; ++m) sum -= L[j * p + m] * L[k * p + m];
                if (k == j) {
                    if (!(sum > 0.0))
                        return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                    "covariance is not positive definite");
                    L[j * p + j] = std::sqrt(sum);
                } else {
                    L[j * p + k] = sum / L[k * p + k];
                }
            }
        }
        const auto rows = static_cast<size_t>(n);
        std::vector<double> z(rows * p);
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        n4m_pcg64_engine_standard_normal_fill(&rng, z.data(), z.size());
        for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) sum += z[i * p + k] * L[j * p + k];
            out[i * p + j] = std::ldexp(sum, half) + mean[j];
            if (!std::isfinite(out[i * p + j]))
                return fail(ctx, N4M_ERR_INTERNAL, "multivariate normal draw is not finite");
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "multivariate normal allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "multivariate normal internal error"); }
}

N4M_API n4m_status_t n4m_preflight_noise_proxy_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const uint8_t* valid_mask,
    int32_t window_length, int32_t polyorder, double* band_noise,
    int64_t* band_counts, double* out_two) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!band_noise || !band_counts || !out_two)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null noise proxy output");
    // Validated domain: every odd window in [3, 1001] with polyorder <= 10
    // reproduces polynomial moments within 4e-12 (Methods 1.3.7 scan); higher
    // orders lose the reproduction (order 38, window 41: constant residual 0.83).
    if (window_length < 3 || window_length > max_noise_window || window_length % 2 == 0 ||
        polyorder < 0 || polyorder > max_noise_polyorder || polyorder > window_length - 2)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "noise proxy needs an odd window in [3, 1001] and polyorder <= min(10, window-2)");
    const int64_t n = X->rows, p = X->cols;
    if (n < 1 || p < window_length)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "noise proxy needs cols >= window length");
    if (static_cast<uint64_t>(n) > SIZE_MAX / sizeof(double) / static_cast<uint64_t>(p))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "noise proxy size overflow");
    n4m_pp_savgol_state_t* sg = n4m_pp_savgol_state_new(
        window_length, polyorder, 0, 1.0, N4M_PP_SAVGOL_NEAREST, 0.0);
    if (!sg) return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "Savitzky-Golay state allocation failed");
    struct Release { n4m_pp_savgol_state_t* s; ~Release() { n4m_pp_savgol_state_free(s); } } release{sg};
    try {
        const int64_t h = window_length / 2;
        const auto sz = [](int64_t v) { return static_cast<size_t>(v); };
        const auto valid = [&](int64_t i, int64_t j) {
            return !valid_mask || valid_mask[sz(i) * sz(p) + sz(j)];
        };
        // Domain pre-pass (rule 13) over the valid cells of each row.
        for (int64_t i = 0; i < n; ++i) {
            DomainRange scan;
            for (int64_t j = 0; j < p; ++j) {
                if (!valid(i, j)) continue;
                if (!std::isfinite(at(*X, i, j)))
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked noise proxy input");
                scan.add(at(*X, i, j));
            }
            if (!scan.inside()) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
        }
        // Smoothing weights w_k = sum_{m<=order} t_m(0) t_m(k) / ||t_m||^2 from
        // the Gram (discrete Chebyshev) polynomials on k = -h..h, N = 2h+1:
        // t_0 = 1, t_1 = k, t_{m+1} = k t_m - b_m t_{m-1},
        // b_m = m^2 (N^2 - m^2) / (4 (4 m^2 - 1)), ||t_m||^2 = b_m ||t_{m-1}||^2,
        // in double-double. Odd t_m vanish at 0, so w_{-k} = w_k exactly.
        const double nw = static_cast<double>(window_length);
        std::vector<DD> weight(sz(h + 1), dd_div({1.0, 0.0}, {nw, 0.0}));
        std::vector<DD> t_prev(sz(h + 1)), t_cur(sz(h + 1));
        for (int64_t k = 0; k <= h; ++k) {
            t_prev[sz(k)] = {1.0, 0.0};
            t_cur[sz(k)] = {static_cast<double>(k), 0.0};
        }
        DD norm = {nw, 0.0};
        for (int32_t m = 1; m <= polyorder; ++m) {
            const double md = static_cast<double>(m);
            const DD beta = dd_div({md * md * (nw * nw - md * md), 0.0},
                                                 {4.0 * (4.0 * md * md - 1.0), 0.0});
            norm = dd_mul(norm, beta);  // ||t_m||^2
            if (m % 2 == 0) {
                const DD scale = dd_div(t_cur[0], norm);
                for (int64_t k = 0; k <= h; ++k)
                    weight[sz(k)] = dd_add(weight[sz(k)], dd_mul(scale, t_cur[sz(k)]));
            }
            if (m == polyorder) break;
            // t_{m+1} = k t_m - beta_m t_{m-1}
            for (int64_t k = 0; k <= h; ++k) {
                const DD next = dd_add(dd_mul({static_cast<double>(k), 0.0}, t_cur[sz(k)]),
                                       dd_neg(dd_mul(beta, t_prev[sz(k)])));
                t_prev[sz(k)] = t_cur[sz(k)];
                t_cur[sz(k)] = next;
            }
        }
        // Cross-checks: the weights agree with the Methods Savitzky-Golay
        // engine's impulse response, which reproduces polynomials of degree
        // <= order (moments within 1e-9), and sum to 1 within 2^-60 (a
        // sanity check; the double-double construction is far more accurate).
        const int64_t centre = window_length - 1, impulse_cols = 2 * centre + 1;
        std::vector<double> impulse(sz(impulse_cols), 0.0), response(sz(impulse_cols));
        impulse[sz(centre)] = 1.0;
        const auto sst = n4m_pp_savgol_state_apply(sg, impulse.data(), 1, impulse_cols, response.data());
        if (sst != N4M_OK) return fail(ctx, sst, "Savitzky-Golay impulse response failed");
        for (int32_t order = 0; order <= polyorder; ++order) {
            long double moment = 0.0L;
            for (int64_t q = -h; q <= h; ++q)
                moment += static_cast<long double>(response[sz(centre + q)]) *
                          std::pow(static_cast<long double>(q) / static_cast<long double>(h),
                                   static_cast<long double>(order));
            if (!(std::fabs(moment - (order == 0 ? 1.0L : 0.0L)) <= moment_tolerance))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "Savitzky-Golay weights do not reproduce polynomial moments");
        }
        DD total = weight[0];
        for (int64_t k = 1; k <= h; ++k) {
            const DD w = weight[sz(k)];
            if (!(std::fabs(w.hi - response[sz(centre + k)]) <= 1e-9 &&
                  std::fabs(w.hi - response[sz(centre - k)]) <= 1e-9))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "Gram weights disagree with the Savitzky-Golay engine");
            total = dd_add(total, dd_add(w, w));
        }
        if (!(std::fabs(weight[0].hi - response[sz(centre)]) <= 1e-9) ||
            !(std::fabs((total.hi - 1.0) + total.lo) <= 0x1p-60))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "Savitzky-Golay weights do not sum to 1");
        // Residual gain for unit white noise: g^2 = (1 - w_0)^2 + 2 sum_{k>0} w_k^2.
        const DD one_minus = dd_add({1.0, 0.0}, dd_neg(weight[0]));
        DD gain2 = dd_mul(one_minus, one_minus);
        for (int64_t k = 1; k <= h; ++k) {
            const DD w2 = dd_mul(weight[sz(k)], weight[sz(k)]);
            gain2 = dd_add(gain2, dd_add(w2, w2));
        }
        const double gain = std::sqrt(gain2.hi + gain2.lo);
        if (!(gain > 0.0)) return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "zero residual gain");
        // Residual of an evaluable band j (its whole window valid), from the
        // window alone (values outside every evaluable window never enter):
        //   r_j = x_j - sum_k w_k x_{j+k} = sum_{k>0} w_k c_k,
        //   c_k = 2 x_j - x_{j+k} - x_{j-k}
        // (sum_k w_k = 1). Each c_k is kept exactly as three doubles (TwoSum,
        // no overflow inside the domain) and multiplied by the double-double
        // weight; the sum is double-double and rounded once to binary64.
        // Honest bound against the exact SG residual (weight errors and the
        // final rounding included): |r_hat - r| <= 2 W eps sum_k |c_k|
        // max_k |w_k|, W the window length, eps = 2^-52.
        std::vector<int64_t> invalid_prefix(sz(p) + 1);
        std::vector<DD> band_sum(sz(p));
        DD pooled_sum;
        int64_t pooled_count = 0;
        std::fill(band_counts, band_counts + p, int64_t{0});
        for (int64_t i = 0; i < n; ++i) {
            invalid_prefix[0] = 0;
            for (int64_t j = 0; j < p; ++j)
                invalid_prefix[sz(j) + 1] = invalid_prefix[sz(j)] + (valid(i, j) ? 0 : 1);
            for (int64_t j = h; j < p - h; ++j) {
                if (invalid_prefix[sz(j + h) + 1] != invalid_prefix[sz(j - h)]) continue;
                const double xj = at(*X, i, j);
                DD r{};
                for (int64_t q = 1; q <= h; ++q) {
                    const DD s1 = dd_two_sum(2.0 * xj, -at(*X, i, j + q));
                    const DD s2 = dd_two_sum(s1.hi, -at(*X, i, j - q));
                    const DD& w = weight[sz(q)];
                    r = dd_add(r, dd_mul(w, {s2.hi, 0.0}));
                    r = dd_add(r, dd_mul(w, {s2.lo, 0.0}));
                    r = dd_add(r, dd_mul(w, {s1.lo, 0.0}));
                }
                const double res = r.hi + r.lo;
                ++band_counts[j];
                ++pooled_count;
                band_sum[sz(j)] = dd_add(band_sum[sz(j)], res * res);
                pooled_sum = dd_add(pooled_sum, res * res);
            }
        }
        for (int64_t j = 0; j < p; ++j)
            band_noise[j] = band_counts[j]
                ? std::sqrt((band_sum[sz(j)].hi + band_sum[sz(j)].lo) / static_cast<double>(band_counts[j])) / gain
                : nan_value;
        out_two[0] = pooled_count
            ? std::sqrt((pooled_sum.hi + pooled_sum.lo) / static_cast<double>(pooled_count)) / gain
            : nan_value;
        out_two[1] = gain;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "noise proxy allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "noise proxy internal error"); }
}

N4M_API n4m_status_t n4m_preflight_snv_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    int64_t n_groups, const n4m_matrix_view_t* out, double* error_bounds) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    for (const auto* v : {X, out}) {
        const auto st = check_view(ctx, v);
        if (st != N4M_OK) return st;
    }
    if (!error_bounds) return fail(ctx, N4M_ERR_NULL_POINTER, "null SNV bound output");
    const int64_t n = X->rows, p = X->cols;
    if (n < 1 || p < 2) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "SNV needs rows >= 1 and cols >= 2");
    if (group_ids ? (n_groups < 1 || n_groups > n) : n_groups != n)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid SNV group count");
    if (out->rows != n_groups || out->cols != p)
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH, "SNV output shape mismatch");
    // rows*cols for the copy and the result, plus 2*groups*cols accumulators.
    if (static_cast<uint64_t>(n) > SIZE_MAX / sizeof(double) / 4 / static_cast<uint64_t>(p))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "SNV size overflow");
    n4m_pp_snv_state_t* state = n4m_pp_snv_state_new(1, 1, 0);
    if (!state) return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "SNV state allocation failed");
    struct Release { n4m_pp_snv_state_t* s; ~Release() { n4m_pp_snv_state_free(s); } } release{state};
    try {
        const auto sz = [](int64_t v) { return static_cast<size_t>(v); };
        std::vector<int64_t> members(sz(n_groups), 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group_ids ? group_ids[i] : i;
            if (g < 0 || g >= n_groups)
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "SNV group ID out of range");
            ++members[sz(g)];
        }
        if (std::find(members.begin(), members.end(), int64_t{0}) != members.end())
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "empty SNV group");
        std::vector<double> data(sz(n) * sz(p)), z(sz(n) * sz(p)), row_bound(sz(n));
        for (int64_t i = 0; i < n; ++i) {
            DomainRange scan;
            for (int64_t j = 0; j < p; ++j) {
                const double x = at(*X, i, j);
                if (!std::isfinite(x)) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite SNV input");
                scan.add(x);
                data[sz(i) * sz(p) + sz(j)] = x;
            }
            if (!scan.inside()) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
        }
        const auto st = n4m_pp_snv_apply_bounded(state, data.data(), n, p, z.data(),
                                                 row_bound.data());
        if (st != N4M_OK) return fail(ctx, st, "SNV is not representable for a row");
        // Equal-weight group means of the SNV rows, compensated per band.
        // Bound: max member bound + (4u + 2 r u^2) max member |z| for r > 1.
        constexpr double u = 0x1p-53;
        std::vector<double> sum(sz(n_groups) * sz(p), 0.0), comp(sum.size(), 0.0);
        std::vector<double> max_bound(sz(n_groups), 0.0), max_z(sz(n_groups), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            const auto g = sz(group_ids ? group_ids[i] : i);
            max_bound[g] = std::max(max_bound[g], row_bound[sz(i)]);
            for (int64_t j = 0; j < p; ++j) {
                const double v = z[sz(i) * sz(p) + sz(j)];
                max_z[g] = std::max(max_z[g], std::fabs(v));
                double& s0 = sum[g * sz(p) + sz(j)];
                double& c0 = comp[g * sz(p) + sz(j)];
                const double t = s0 + v;
                c0 += std::fabs(s0) >= std::fabs(v) ? (s0 - t) + v : (v - t) + s0;
                s0 = t;
            }
        }
        for (int64_t g = 0; g < n_groups; ++g) {
            const double r = static_cast<double>(members[sz(g)]);
            for (int64_t j = 0; j < p; ++j) {
                const size_t c = sz(g) * sz(p) + sz(j);
                put(*out, g, j, (sum[c] + comp[c]) / r);
            }
            error_bounds[g] = max_bound[sz(g)] +
                (r > 1.0 ? (4.0 * u + 2.0 * r * u * u) * max_z[sz(g)] : 0.0);
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "SNV allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "SNV internal error"); }
}

N4M_API n4m_status_t n4m_preflight_pairs_within_tolerance(
    n4m_context_t* ctx, const n4m_matrix_view_t* query,
    const n4m_matrix_view_t* reference, const int64_t* query_groups,
    const int64_t* reference_groups, double tolerance, int64_t max_pairs,
    int64_t* pair_indices, double* pair_distances, uint8_t* query_flags,
    uint8_t* reference_flags, int64_t* out_counts) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const bool self = reference == nullptr;
    auto st = check_view(ctx, query);
    if (st != N4M_OK) return st;
    if (!self) {
        st = check_view(ctx, reference);
        if (st != N4M_OK) return st;
    }
    const n4m_matrix_view_t& R = self ? *query : *reference;
    if (!query_flags || !out_counts || (!self && !reference_flags) ||
        (max_pairs > 0 && (!pair_indices || !pair_distances)))
        return fail(ctx, N4M_ERR_NULL_POINTER, "null pair output");
    if (!self && (query_groups == nullptr) != (reference_groups == nullptr))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "both pair group arrays are required");
    if (query->rows < 1 || query->cols < 1 || R.rows < 1 || R.cols != query->cols ||
        max_pairs < 0 || !std::isfinite(tolerance) || !(tolerance > 0.0))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid pair dimensions or tolerance");
    const int64_t* rgroups = self ? query_groups : reference_groups;
    for (const auto* v : {query, &R})
        for (int64_t i = 0; i < v->rows; ++i) {
            DomainRange scan;
            for (int64_t j = 0; j < v->cols; ++j) {
                if (!std::isfinite(at(*v, i, j)))
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite pair input");
                scan.add(at(*v, i, j));
            }
            if (!scan.inside()) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
        }
    std::fill(query_flags, query_flags + query->rows, uint8_t{0});
    if (!self) std::fill(reference_flags, reference_flags + R.rows, uint8_t{0});
    uint8_t* rflags = self ? query_flags : reference_flags;
    // RMS distance and decision, independent of the tolerance. Inside the
    // domain no difference q_j - r_j overflows or underflows to zero.
    //   fast estimate: d_j = q_j - r_j rescaled by an exact power of two 2^-e
    //   of max|d|, rho = sqrt(sum t^2 / cols) with a compensated sum; RMS =
    //   rho 2^e, and tau = tol 2^-e is exact (or decides by its magnitude).
    //   rho carries a relative error below 8u, so |rho - tau| > 2^-45 tau
    //   decides. Otherwise the decision is exact: the sign of
    //   sum_j (q_j^2 - 2 q_j r_j + r_j^2) - cols tol^2 in an exact fixed-point
    //   accumulator (core/common/exact_sum.hpp).
    // An accepted pair publishes min(rho 2^e, tol).
    const double inv_cols = 1.0 / static_cast<double>(query->cols);
    const double margin = 0x1p-45;
    const auto exact_within = [&](int64_t i, int64_t k) {
        ::n4m::core::ExactSum sum;
        for (int64_t j = 0; j < query->cols; ++j) {
            const double a = at(*query, i, j), b = at(R, k, j);
            sum.add_product(a, a, false);
            sum.add_product(a, b, true);
            sum.add_product(a, b, true);
            sum.add_product(b, b, false);
        }
        for (int64_t c = 0; c < query->cols; ++c) sum.add_product(tolerance, tolerance, true);
        return sum.sign() <= 0;
    };
    const auto rms_within = [&](int64_t i, int64_t k, double& distance) {
        double dmax = 0.0;
        for (int64_t j = 0; j < query->cols; ++j)
            dmax = std::max(dmax, std::fabs(at(*query, i, j) - at(R, k, j)));
        // Distinct doubles never round to a zero difference.
        if (dmax == 0.0) { distance = 0.0; return true; }
        const int e = std::ilogb(dmax);
        const double tau = std::ldexp(tolerance, -e);
        const double far2 = tau * tau * (1.0 + 0x1p-40);
        double sum = 0.0, comp = 0.0;  // Neumaier
        for (int64_t j = 0; j < query->cols; ++j) {
            const double t = std::ldexp(std::fabs(at(*query, i, j) - at(R, k, j)), -e);
            const double sq = t * t, next = sum + sq;
            comp += sum >= sq ? (sum - next) + sq : (sq - next) + sum;
            sum = next;
            if (sum * inv_cols > far2) return false;
        }
        const double rho = std::sqrt((sum + comp) * inv_cols);
        bool within;
        if (rho > tau * (1.0 + margin)) within = false;
        else if (rho < tau * (1.0 - margin)) within = true;
        else within = exact_within(i, k);
        if (!within) return false;
        distance = std::min(std::ldexp(rho, e), tolerance);
        return true;
    };
    int64_t found = 0, candidates = 0;
    for (int64_t i = 0; i < query->rows; ++i) {
        for (int64_t k = self ? i + 1 : 0; k < R.rows; ++k) {
            if (query_groups && query_groups[i] == rgroups[k]) continue;
            ++candidates;
            double distance = 0.0;
            if (!rms_within(i, k, distance)) continue;
            query_flags[i] = 1;
            rflags[k] = 1;
            if (found < max_pairs) {
                pair_indices[2 * found] = i;
                pair_indices[2 * found + 1] = k;
                pair_distances[found] = distance;
            }
            ++found;
        }
    }
    out_counts[0] = found;
    out_counts[1] = candidates;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_preflight_factor_association(
    n4m_context_t* ctx, const int64_t* factor_codes, const double* targets,
    int64_t n, int32_t categorical, int32_t n_permutations, uint64_t seed,
    double* out_eight, int64_t* out_two) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!out_eight || !out_two) return fail(ctx, N4M_ERR_NULL_POINTER, "null association buffer");
    if (n_permutations < 0 || n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid association dimensions");
    try {
        FactorProblem p;
        const n4m_status_t st = prepare_factor(ctx, factor_codes, targets, n, categorical, p);
        if (st != N4M_OK) return st;
        const auto g = static_cast<int64_t>(p.level_n.size());
        const auto v = static_cast<int64_t>(p.value_n.size());
        const long double work =
            p.cost() * (static_cast<long double>(n_permutations) + 1.0L);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "association permutations exceed 250 million operation budget");
        for (int i = 0; i < 8; ++i) out_eight[i] = nan_value;
        out_two[0] = 0;
        out_two[1] = 0;
        out_eight[4] = static_cast<double>(g);
        out_eight[5] = static_cast<double>(v);
        out_eight[6] = std::min(log10_factorial_ratio(n, p.level_n),
                                log10_factorial_ratio(n, p.value_n));
        if (!p.defined()) return N4M_OK;
        p.allocate();
        const double observed = p.statistic(p.level.data(), p.value.data());
        if (!p.categorical) {
            out_eight[0] = observed;
            // SSW from explicit within-level residuals (never SST - SSB). Each
            // level is scaled by its own power of two and centered on its own
            // values (shifted origin), from the original targets, so no
            // global scale or centre can erase a within-level spread. Level
            // sums of squares q_l * 2^(2 e_l) are combined by exponents.
            std::vector<std::vector<double>> by_level(static_cast<size_t>(g));
            for (size_t i = 0; i < p.level.size(); ++i)
                by_level[static_cast<size_t>(p.level[i])].push_back(p.raw[i]);
            std::vector<double> q_level;
            std::vector<int> e_level;
            bool varies = false;
            for (auto& members : by_level) {
                const bool distinct = std::any_of(members.begin(), members.end(),
                                                  [&members](double x) { return x != members[0]; });
                varies |= distinct;
                if (!distinct) continue;
                const int e1 = bounding_exponent(members);
                for (double& x : members) x = std::ldexp(x, -e1);
                center_in_place(members);
                int e2 = 0;
                const double q = scaled_sum_squares(members, e2);
                if (!(q > 0.0))
                    return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                "within-level variation lost to rounding");
                q_level.push_back(q);
                e_level.push_back(e1 + e2);
            }
            if (n > g && varies) {
                const int e_max = *std::max_element(e_level.begin(), e_level.end());
                double q_within = 0.0;
                for (size_t l = 0; l < q_level.size(); ++l)
                    q_within += std::ldexp(q_level[l], 2 * (e_level[l] - e_max));
                // SSB = eta^2 * sst * 2^(2 (scale + variation)), SSW = q * 2^(2 e_max).
                const double f = std::ldexp(
                    (observed * p.sst / static_cast<double>(g - 1)) *
                        static_cast<double>(n - g) / q_within,
                    2 * (p.scale + p.variation - e_max));
                if (!std::isfinite(f) || (observed > 0.0 && !(f >= DBL_MIN)))
                    return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                "ANOVA F is not representable as a normal double");
                out_eight[1] = f;
            }
            out_eight[2] = static_cast<double>(g - 1);
            out_eight[3] = static_cast<double>(n - g);
        } else {
            out_eight[0] = std::sqrt(observed);
            out_eight[1] = observed * static_cast<double>(n) *
                           static_cast<double>(std::min(g, v) - 1);
            out_eight[2] = static_cast<double>((g - 1) * (v - 1));
        }
        if (n_permutations == 0) return N4M_OK;
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(static_cast<size_t>(n)), permuted(static_cast<size_t>(n));
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            for (size_t i = 0; i < perm.size(); ++i)
                permuted[i] = p.value[static_cast<size_t>(perm[i])];
            if (p.statistic(p.level.data(), permuted.data()) >= observed - tolerance) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_eight[7] = static_cast<double>(1 + at_least) /
                       static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "association allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "association internal error"); }
}

N4M_API n4m_status_t n4m_stats_factor_association_exact(
    n4m_context_t* ctx, const int64_t* factor_codes, const double* targets,
    int64_t n, int32_t categorical, int64_t max_arrangements, double* out_four) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!out_four) return fail(ctx, N4M_ERR_NULL_POINTER, "null exact association buffer");
    try {
        FactorProblem p;
        const n4m_status_t st = prepare_factor(ctx, factor_codes, targets, n, categorical, p);
        if (st != N4M_OK) return st;
        const auto cap = static_cast<uint64_t>(std::max<int64_t>(max_arrangements, 1));
        const auto setup = [&p]() { p.allocate(); };
        // Enumerate the side with fewer distinct arrangements; the other stays fixed.
        if (arrangement_count(p.level_n, cap) <= arrangement_count(p.value_n, cap))
            return exact_distribution(ctx, p.level, p.level_n, max_arrangements, p.defined(),
                                      p.cost(), setup,
                                      [&p](const int64_t* L) {
                                          return p.statistic(L, p.value.data());
                                      },
                                      out_four);
        return exact_distribution(ctx, p.value, p.value_n, max_arrangements, p.defined(),
                                  p.cost(), setup,
                                  [&p](const int64_t* Y) {
                                      return p.statistic(p.level.data(), Y);
                                  },
                                  out_four);
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "exact association allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "exact association internal error"); }
}

N4M_API n4m_status_t n4m_preflight_holm_adjust(
    n4m_context_t* ctx, const double* p_values, int64_t m, double* adjusted) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!p_values || !adjusted)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null Holm buffer");
    if (m < 1 || static_cast<uint64_t>(m) >
                     static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid Holm family size");
    try {
        for (int64_t i = 0; i < m; ++i)
            if (!(p_values[i] >= 0.0 && p_values[i] <= 1.0))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "p-values must lie in [0, 1]");
        if (!p_values_in_domain(p_values, static_cast<size_t>(m)))
            return fail(ctx, N4M_ERR_UNSUPPORTED, p_value_domain_message);
        std::vector<int64_t> order(static_cast<size_t>(m));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [p_values](int64_t a, int64_t b) { return p_values[a] < p_values[b]; });
        double running = 0.0;
        for (int64_t rank = 0; rank < m; ++rank) {
            const int64_t i = order[static_cast<size_t>(rank)];
            running = std::max(running, static_cast<double>(m - rank) * p_values[i]);
            adjusted[i] = std::min(1.0, running);
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "Holm allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "Holm internal error"); }
}
N4M_API n4m_status_t n4m_stats_correlation_test(
    n4m_context_t* ctx, const double* x, const double* y, int64_t n,
    int32_t method, int32_t n_permutations, uint64_t seed,
    double* out_five, int64_t* out_two) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!out_five || !out_two) return fail(ctx, N4M_ERR_NULL_POINTER, "null correlation buffer");
    if (n_permutations < 0 || n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid correlation dimensions or method");
    try {
        CorrelationProblem p;
        const n4m_status_t st = prepare_correlation(ctx, x, y, n, method, p);
        if (st != N4M_OK) return st;
        const long double work = static_cast<long double>(n) *
                                 (static_cast<long double>(n_permutations) + 1.0L);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "correlation permutations exceed 250 million operation budget");
        for (int i = 0; i < 5; ++i) out_five[i] = nan_value;
        out_two[0] = 0;
        out_two[1] = 0;
        out_five[1] = static_cast<double>(p.x_n.size());
        out_five[2] = static_cast<double>(p.y_n.size());
        out_five[3] = std::min(log10_factorial_ratio(n, p.x_n), log10_factorial_ratio(n, p.y_n));
        if (!p.defined()) return N4M_OK;
        const double observed = p.statistic(p.x_code.data(), p.y_code.data());
        out_five[0] = observed;
        if (n_permutations == 0) return N4M_OK;
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(static_cast<size_t>(n)), permuted(static_cast<size_t>(n));
        const double threshold =
            std::abs(observed) - std::sqrt(std::numeric_limits<double>::epsilon());
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            for (size_t i = 0; i < perm.size(); ++i)
                permuted[i] = p.y_code[static_cast<size_t>(perm[i])];
            if (std::abs(p.statistic(p.x_code.data(), permuted.data())) >= threshold) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_five[4] = static_cast<double>(1 + at_least) /
                      static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "correlation allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "correlation internal error"); }
}

N4M_API n4m_status_t n4m_stats_correlation_exact(
    n4m_context_t* ctx, const double* x, const double* y, int64_t n, int32_t method,
    int64_t max_arrangements, double* out_four) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!out_four) return fail(ctx, N4M_ERR_NULL_POINTER, "null exact correlation buffer");
    try {
        CorrelationProblem p;
        const n4m_status_t st = prepare_correlation(ctx, x, y, n, method, p);
        if (st != N4M_OK) return st;
        const auto cap = static_cast<uint64_t>(std::max<int64_t>(max_arrangements, 1));
        const auto cost = static_cast<long double>(n);
        const auto setup = []() {};
        if (arrangement_count(p.x_n, cap) <= arrangement_count(p.y_n, cap))
            return exact_distribution(ctx, p.x_code, p.x_n, max_arrangements, p.defined(), cost,
                                      setup,
                                      [&p](const int64_t* X) {
                                          return std::abs(p.statistic(X, p.y_code.data()));
                                      },
                                      out_four);
        return exact_distribution(ctx, p.y_code, p.y_n, max_arrangements, p.defined(), cost,
                                  setup,
                                  [&p](const int64_t* Y) {
                                      return std::abs(p.statistic(p.x_code.data(), Y));
                                  },
                                  out_four);
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "exact correlation allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "exact correlation internal error"); }
}

N4M_API n4m_status_t n4m_stats_bh_adjust(
    n4m_context_t* ctx, const double* p_values, int64_t m, double* adjusted) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!p_values || !adjusted)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null Benjamini-Hochberg buffer");
    if (m < 1 || static_cast<uint64_t>(m) >
                     static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid Benjamini-Hochberg family size");
    try {
        for (int64_t i = 0; i < m; ++i)
            if (!(p_values[i] >= 0.0 && p_values[i] <= 1.0))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "p-values must lie in [0, 1]");
        if (!p_values_in_domain(p_values, static_cast<size_t>(m)))
            return fail(ctx, N4M_ERR_UNSUPPORTED, p_value_domain_message);
        std::vector<int64_t> order(static_cast<size_t>(m));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [p_values](int64_t a, int64_t b) { return p_values[a] < p_values[b]; });
        double running = 1.0;
        for (int64_t rank = m; rank >= 1; --rank) {
            const int64_t i = order[static_cast<size_t>(rank - 1)];
            running = std::min(running, static_cast<double>(m) / static_cast<double>(rank) *
                                            p_values[i]);
            adjusted[i] = running;
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "Benjamini-Hochberg allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "Benjamini-Hochberg internal error"); }
}

N4M_API n4m_status_t n4m_stats_permanova(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* term_codes,
    const int64_t* strata_codes, int32_t n_permutations, uint64_t seed,
    double* out_fifteen, int64_t* out_two, int32_t* ss_exponent) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!term_codes || !out_fifteen || !out_two || !ss_exponent)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null PERMANOVA buffer");
    const int64_t n = X->rows, k = X->cols;
    if (n < 2 || k < 1 || n_permutations < 0 || n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PERMANOVA dimensions");
    const DomainRange x_range = domain_of({X});
    if (x_range.finite && !x_range.inside())
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        const auto nn = static_cast<size_t>(n), kk = static_cast<size_t>(k);
        std::vector<int64_t> level_n, stratum_n;
        const auto level = dense_codes(std::vector<int64_t>(term_codes, term_codes + n), level_n);
        const auto stratum = dense_codes(
            strata_codes ? std::vector<int64_t>(strata_codes, strata_codes + n)
                         : std::vector<int64_t>(nn, 0),
            stratum_n);
        const size_t a = level_n.size(), s = stratum_n.size();
        // Sizes and budgets before any allocation proportional to them.
        if (a > 2048)
            return fail(ctx, N4M_ERR_UNSUPPORTED, "PERMANOVA term exceeds 2048 levels");
        const long double work =
            (2.0L * static_cast<long double>(n) * static_cast<long double>(k + 1) +
             2.0L * static_cast<long double>(a) * static_cast<long double>(a) *
                 static_cast<long double>(k)) *
                (static_cast<long double>(n_permutations) + 1.0L) +
            static_cast<long double>(a) * static_cast<long double>(n);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "PERMANOVA permutations exceed 250 million operation budget");
        std::vector<double> y;
        int exponent = 0;
        long double range_ss = 0.0L;
        if (!centred_rows(*X, y, exponent, range_ss))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PERMANOVA input");
        // Stratum means, total and between-strata SS, within-strata centred
        // rows; every sum of squares is a sum of squared terms.
        std::vector<long double> smean(s * kk, 0.0L);
        for (size_t i = 0; i < nn; ++i)
            for (size_t j = 0; j < kk; ++j)
                smean[static_cast<size_t>(stratum[i]) * kk + j] += ld(y[i * kk + j]);
        for (size_t g = 0; g < s; ++g)
            for (size_t j = 0; j < kk; ++j)
                smean[g * kk + j] /= static_cast<long double>(stratum_n[g]);
        long double ss_total = 0.0L, ss_strata = 0.0L, within = 0.0L;
        std::vector<double> xw(nn * kk);
        for (size_t i = 0; i < nn; ++i)
            for (size_t j = 0; j < kk; ++j) {
                const long double w = ld(y[i * kk + j]) - smean[static_cast<size_t>(stratum[i]) * kk + j];
                ss_total += ld(y[i * kk + j]) * ld(y[i * kk + j]);
                within += w * w;
                xw[i * kk + j] = static_cast<double>(w);
            }
        for (size_t g = 0; g < s; ++g)
            for (size_t j = 0; j < kk; ++j)
                ss_strata += static_cast<long double>(stratum_n[g]) * smean[g * kk + j] *
                             smean[g * kk + j];
        // Occupied (stratum, level) cells and their counts.
        std::vector<std::pair<int64_t, int64_t>> cell(nn);
        for (size_t i = 0; i < nn; ++i) cell[i] = {stratum[i], level[i]};
        std::sort(cell.begin(), cell.end());
        std::vector<std::pair<std::pair<int64_t, int64_t>, int64_t>> cells;
        for (const auto& c : cell) {
            if (cells.empty() || cells.back().first != c) cells.push_back({c, 0});
            ++cells.back().second;
        }
        // Connected components of the bipartite level-stratum graph; the
        // identifiable term df is levels - components.
        std::vector<size_t> parent(a + s);
        std::iota(parent.begin(), parent.end(), size_t{0});
        const std::function<size_t(size_t)> root = [&](size_t u) {
            while (parent[u] != u) u = parent[u] = parent[parent[u]];
            return u;
        };
        for (const auto& c : cells) {
            const size_t u = root(static_cast<size_t>(c.first.second));
            const size_t v = root(a + static_cast<size_t>(c.first.first));
            if (u != v) parent[std::max(u, v)] = std::min(u, v);
        }
        PermanovaKernel kernel{level, stratum, xw, kk, std::vector<int64_t>(a, -1), {}, 0,
                               std::vector<std::vector<std::pair<int64_t, int64_t>>>(s),
                               stratum_n, {}, {}, {}, {}};
        for (const auto& c : cells)
            kernel.stratum_cells[static_cast<size_t>(c.first.first)].push_back(
                {c.first.second, c.second});
        std::vector<bool> component_seen(a + s, false);
        for (size_t l = 0; l < a; ++l) {
            const size_t c = root(l);
            if (!component_seen[c]) {
                component_seen[c] = true;  // the lowest level is the reference
                continue;
            }
            kernel.reduced[l] = static_cast<int64_t>(kernel.r++);
        }
        const size_t r = kernel.r;
        const int64_t df_term = static_cast<int64_t>(r);
        const int64_t df_res = n - static_cast<int64_t>(s) - df_term;
        double log10_distinct = 0.0;
        {
            std::vector<long double> m(r * r, 0.0L);
            for (size_t g = 0; g < s; ++g) {
                const auto ns = static_cast<long double>(stratum_n[g]);
                log10_distinct += std::lgamma(static_cast<double>(stratum_n[g]) + 1.0);
                for (const auto& [lu, cu] : kernel.stratum_cells[g]) {
                    log10_distinct -= std::lgamma(static_cast<double>(cu) + 1.0);
                    const int64_t ru = kernel.reduced[static_cast<size_t>(lu)];
                    if (ru < 0) continue;
                    for (const auto& [lv, cv] : kernel.stratum_cells[g]) {
                        const int64_t rv = kernel.reduced[static_cast<size_t>(lv)];
                        if (rv < 0) continue;
                        m[static_cast<size_t>(ru) * r + static_cast<size_t>(rv)] -=
                            static_cast<long double>(cu) * static_cast<long double>(cv) / ns;
                    }
                }
            }
            log10_distinct /= std::log(10.0);
            for (size_t l = 0; l < a; ++l)
                if (kernel.reduced[l] >= 0) {
                    const auto u = static_cast<size_t>(kernel.reduced[l]);
                    m[u * r + u] += static_cast<long double>(level_n[l]);
                }
            kernel.chol.assign(r * r, 0.0);
            for (size_t j = 0; j < r; ++j) {
                long double d = m[j * r + j];
                for (size_t l = 0; l < j; ++l)
                    d -= ld(kernel.chol[j * r + l]) * ld(kernel.chol[j * r + l]);
                if (!(d > 1e-12L * m[j * r + j]))
                    return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                "PERMANOVA design is numerically singular");
                const long double pivot = std::sqrt(d);
                kernel.chol[j * r + j] = static_cast<double>(pivot);
                for (size_t i = j + 1; i < r; ++i) {
                    long double v = m[i * r + j];
                    for (size_t l = 0; l < j; ++l)
                        v -= ld(kernel.chol[i * r + l]) * ld(kernel.chol[j * r + l]);
                    kernel.chol[i * r + j] = static_cast<double>(v / pivot);
                }
            }
        }
        kernel.sums.assign(a * kk, 0.0);
        kernel.rhs.assign(r, 0.0);
        kernel.effect.assign(a * kk, 0.0);
        kernel.stratum_effect.assign(s * kk, 0.0);
        const auto [ss_term, ss_res] =
            r > 0 ? kernel(nullptr) : std::pair<long double, long double>{0.0L, within};
        for (int i = 0; i < 15; ++i) out_fifteen[i] = nan_value;
        if (!publish({ss_total, ss_strata, ss_term, ss_res}, 2 * exponent,
                     {&out_fifteen[0], &out_fifteen[1], &out_fifteen[2], &out_fifteen[3]},
                     *ss_exponent))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                        "PERMANOVA sum of squares below 2^-1074 of the total");
        // Ratios and the test statistic stay in the scaled unit. The partial
        // R^2 denominator is the term + residual SS of the same projection.
        if (df_term > 0 && ss_total > 0.0L)
            out_fifteen[4] = static_cast<double>(std::min(1.0L, ss_term / ss_total));
        if (df_term > 0 && ss_term + ss_res > 0.0L)
            out_fifteen[5] = static_cast<double>(ss_term / (ss_term + ss_res));
        const long double f_scale =
            df_term > 0 ? static_cast<long double>(df_res) / static_cast<long double>(df_term) : 0.0L;
        // F is undefined only for an exactly zero residual SS; below the
        // error estimate it is published and flagged numerically unresolved.
        if (df_term > 0 && df_res > 0) {
            if (ss_res > 0.0L) out_fifteen[6] = static_cast<double>(ss_term / ss_res * f_scale);
            write_resolution(residual_error_estimate(range_ss, s + r, nn, kk), ss_res,
                             &out_fifteen[13], &out_fifteen[14]);
        }
        out_fifteen[7] = static_cast<double>(df_term);
        out_fifteen[8] = static_cast<double>(df_res);
        out_fifteen[9] = static_cast<double>(a);
        out_fifteen[10] = static_cast<double>(s);
        out_fifteen[11] = log10_distinct;
        out_two[0] = 0;
        out_two[1] = 0;
        // As vegan anova.cca: count pseudo-F_perm >= F_obs - sqrt(eps); a
        // permuted residual SS that is exactly zero is an infinite F.
        const double f_observed = out_fifteen[6];
        if (n_permutations == 0 || !std::isfinite(f_observed)) return N4M_OK;
        // Strata in ascending code order, members in ascending row order.
        std::vector<std::vector<int64_t>> members(s);
        for (size_t i = 0; i < nn; ++i)
            members[static_cast<size_t>(stratum[i])].push_back(static_cast<int64_t>(i));
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(nn), q(nn);
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            for (const auto& idx : members) {
                const auto ns = static_cast<int64_t>(idx.size());
                n4m_pcg64_engine_permutation(&rng, q.data(), ns);
                for (size_t j = 0; j < idx.size(); ++j)
                    perm[static_cast<size_t>(idx[j])] = idx[static_cast<size_t>(q[j])];
            }
            const auto [term, rest] = kernel(perm.data());
            const double f = rest > 0.0L ? static_cast<double>(term / rest * f_scale)
                                         : std::numeric_limits<double>::infinity();
            if (f >= f_observed - tolerance) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_fifteen[12] = static_cast<double>(1 + at_least) /
                           static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PERMANOVA allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PERMANOVA internal error"); }
}

N4M_API n4m_status_t n4m_stats_permanova_freedman_lane(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* term_codes,
    const int64_t* condition_codes, int32_t n_conditions, int32_t n_permutations,
    uint64_t seed, double* out_fourteen, int64_t* out_two, int32_t* ss_exponent) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!term_codes || !out_fourteen || !out_two || !ss_exponent ||
        (n_conditions > 0 && !condition_codes))
        return fail(ctx, N4M_ERR_NULL_POINTER, "null Freedman-Lane PERMANOVA buffer");
    const int64_t n = X->rows, k = X->cols;
    if (n < 2 || k < 1 || n_conditions < 0 || n_conditions > 64 || n_permutations < 0 ||
        n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid Freedman-Lane PERMANOVA dimensions");
    const DomainRange x_range = domain_of({X});
    if (x_range.finite && !x_range.inside())
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        const auto nn = static_cast<size_t>(n), kk = static_cast<size_t>(k);
        const auto nc = static_cast<size_t>(n_conditions);
        // Level counts first: the design is generated column by column only
        // after its size, memory and work are checked.
        std::vector<std::vector<int64_t>> condition_dense(nc);
        std::vector<size_t> condition_levels(nc);
        long double columns = 1.0L;
        for (size_t c = 0; c < nc; ++c) {
            std::vector<int64_t> codes(nn), counts;
            for (size_t i = 0; i < nn; ++i) codes[i] = condition_codes[i * nc + c];
            condition_dense[c] = dense_codes(codes, counts);
            condition_levels[c] = counts.size();
            columns += static_cast<long double>(counts.size());
        }
        std::vector<int64_t> term_counts;
        const auto term_dense =
            dense_codes(std::vector<int64_t>(term_codes, term_codes + n), term_counts);
        columns += static_cast<long double>(term_counts.size());
        const long double rows = static_cast<long double>(n);
        if (columns > 4096.0L)
            return fail(ctx, N4M_ERR_UNSUPPORTED, "Freedman-Lane design exceeds 4096 columns");
        if (rows * columns > 32.0L * 1024.0L * 1024.0L)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "Freedman-Lane design basis exceeds 32 Mi values (256 MiB)");
        const long double work =
            2.0L * rows * columns * columns +
            (4.0L * rows * static_cast<long double>(k) * (columns + 1.0L)) *
                (static_cast<long double>(n_permutations) + 1.0L);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "Freedman-Lane permutations exceed 250 million operation budget");
        std::vector<double> y;
        int exponent = 0;
        long double range_ss = 0.0L;
        if (!centred_rows(*X, y, exponent, range_ss))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PERMANOVA input");
        // Reduced design: intercept and the indicators of every condition
        // factor (additive main effects, no interaction); full design adds the
        // term indicators. Numerical ranks (Gram-Schmidt, 1e-9 relative).
        std::vector<std::vector<double>> basis;
        size_t rank_reduced = extend_basis(basis, std::vector<double>(nn, 1.0));
        for (size_t c = 0; c < nc; ++c)
            rank_reduced += extend_with_indicators(basis, condition_dense[c], condition_levels[c]);
        const size_t df_term = extend_with_indicators(basis, term_dense, term_counts.size());
        const size_t rank_full = rank_reduced + df_term;
        const int64_t df_res = n - static_cast<int64_t>(rank_full);
        // Reduced-model residuals e, projected off the reduced design twice
        // (the second pass removes what rounding left of large fitted columns,
        // "twice is enough"); y is centred, so the intercept projection is
        // zero. The term and residual SS then come from one projection of e
        // on the full basis, and the partial R^2 denominator is their sum.
        const std::vector<std::vector<double>> reduced_basis(
            basis.begin(), basis.begin() + static_cast<std::ptrdiff_t>(rank_reduced));
        std::vector<double> e, scratch;
        const long double ss_reduced = project(reduced_basis, 0, y, kk, scratch).first;
        project(reduced_basis, 0, scratch, kk, e);
        long double ss_total = 0.0L;
        for (double v : y) ss_total += ld(v) * ld(v);
        const auto [ss_term, ss_res] = project(basis, rank_reduced, e, kk, scratch);
        for (int i = 0; i < 14; ++i) out_fourteen[i] = nan_value;
        if (!publish({ss_total, ss_reduced, df_term > 0 ? ss_term : 0.0L, ss_res}, 2 * exponent,
                     {&out_fourteen[0], &out_fourteen[1], &out_fourteen[2], &out_fourteen[3]},
                     *ss_exponent))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                        "PERMANOVA sum of squares below 2^-1074 of the total");
        if (df_term > 0 && ss_total > 0.0L)
            out_fourteen[4] = static_cast<double>(std::min(1.0L, ss_term / ss_total));
        if (df_term > 0 && ss_term + ss_res > 0.0L)
            out_fourteen[5] = static_cast<double>(ss_term / (ss_term + ss_res));
        const long double f_scale = df_term > 0 ? static_cast<long double>(df_res) /
                                                      static_cast<long double>(df_term)
                                                : 0.0L;
        if (df_term > 0 && df_res > 0) {
            if (ss_res > 0.0L) out_fourteen[6] = static_cast<double>(ss_term / ss_res * f_scale);
            write_resolution(residual_error_estimate(range_ss, rank_full, nn, kk), ss_res,
                             &out_fourteen[12], &out_fourteen[13]);
        }
        out_fourteen[7] = static_cast<double>(df_term);
        out_fourteen[8] = static_cast<double>(df_res);
        out_fourteen[9] = static_cast<double>(rank_reduced);
        out_fourteen[10] = std::lgamma(static_cast<double>(n) + 1.0) / std::log(10.0);
        out_two[0] = 0;
        out_two[1] = 0;
        const double f_observed = out_fourteen[6];
        if (n_permutations == 0 || !std::isfinite(f_observed)) return N4M_OK;
        // Freedman-Lane: the permuted data fitted + e[perm] has, given the
        // reduced design, the same term SS and residual SS as e[perm]
        // projected on the full basis; both come from explicit projections.
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(nn);
        std::vector<double> ep(nn * kk);
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            for (size_t i = 0; i < nn; ++i)
                for (size_t j = 0; j < kk; ++j)
                    ep[i * kk + j] = e[static_cast<size_t>(perm[i]) * kk + j];
            const auto [term, rest] = project(basis, rank_reduced, ep, kk, scratch);
            const double f = rest > 0.0L ? static_cast<double>(term / rest * f_scale)
                                         : std::numeric_limits<double>::infinity();
            if (f >= f_observed - tolerance) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_fourteen[11] = static_cast<double>(1 + at_least) / static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "Freedman-Lane PERMANOVA allocation failed");
    } catch (...) {
        return fail(ctx, N4M_ERR_INTERNAL, "Freedman-Lane PERMANOVA internal error");
    }
}

N4M_API n4m_status_t n4m_stats_permdisp(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int32_t n_permutations, uint64_t seed, double* distances,
    double* group_mean_distances, double* out_eight, int64_t* out_two,
    int32_t* distance_exponent) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_codes || !distances || !group_mean_distances || !out_eight || !out_two ||
        !distance_exponent)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null PERMDISP buffer");
    const int64_t n = X->rows, k = X->cols;
    if (n < 2 || k < 1 || n_permutations < 0 || n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PERMDISP dimensions");
    const DomainRange x_range = domain_of({X});
    if (x_range.finite && !x_range.inside())
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    const long double work =
        static_cast<long double>(n) * (2.0L * static_cast<long double>(k) +
                                       3.0L * (static_cast<long double>(n_permutations) + 1.0L));
    if (work > max_moment_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "PERMDISP permutations exceed 250 million operation budget");
    try {
        const auto nn = static_cast<size_t>(n), kk = static_cast<size_t>(k);
        std::vector<int64_t> group_n;
        const auto group = dense_codes(std::vector<int64_t>(group_codes, group_codes + n), group_n);
        const size_t g = group_n.size();
        std::vector<double> y;
        int exponent = 0;
        long double range_ss = 0.0L;
        if (!centred_rows(*X, y, exponent, range_ss))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PERMDISP input");
        std::vector<long double> centroid(g * kk, 0.0L);
        for (size_t i = 0; i < nn; ++i)
            for (size_t j = 0; j < kk; ++j)
                centroid[static_cast<size_t>(group[i]) * kk + j] += ld(y[i * kk + j]);
        for (size_t c = 0; c < g; ++c)
            for (size_t j = 0; j < kk; ++j)
                centroid[c * kk + j] /= static_cast<long double>(group_n[c]);
        std::vector<long double> z(nn);
        std::vector<long double> zmean(g, 0.0L);
        for (size_t i = 0; i < nn; ++i) {
            long double d2 = 0.0L;
            for (size_t j = 0; j < kk; ++j) {
                const long double d = ld(y[i * kk + j]) - centroid[static_cast<size_t>(group[i]) * kk + j];
                d2 += d * d;
            }
            z[i] = std::sqrt(d2);
            zmean[static_cast<size_t>(group[i])] += z[i];
        }
        for (size_t c = 0; c < g; ++c) zmean[c] /= static_cast<long double>(group_n[c]);
        long double grand = 0.0L;
        for (size_t c = 0; c < g; ++c) grand += static_cast<long double>(group_n[c]) * zmean[c];
        grand /= static_cast<long double>(n);
        // Between and within SS of the distances, each a sum of squares.
        long double between = 0.0L, residual = 0.0L;
        std::vector<long double> resid(nn);
        for (size_t c = 0; c < g; ++c)
            between += static_cast<long double>(group_n[c]) * (zmean[c] - grand) * (zmean[c] - grand);
        for (size_t i = 0; i < nn; ++i) {
            resid[i] = z[i] - zmean[static_cast<size_t>(group[i])];
            residual += resid[i] * resid[i];
        }
        std::vector<long double> published(z);
        std::vector<double*> targets;
        for (size_t i = 0; i < nn; ++i) targets.push_back(&distances[i]);
        for (size_t c = 0; c < g; ++c) {
            published.push_back(zmean[c]);
            targets.push_back(&group_mean_distances[c]);
        }
        if (!publish(published, exponent, targets, *distance_exponent))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                        "PERMDISP distance below 2^-1074 of the largest");
        for (int i = 0; i < 8; ++i) out_eight[i] = nan_value;
        out_eight[1] = static_cast<double>(g) - 1.0;
        out_eight[2] = static_cast<double>(n) - static_cast<double>(g);
        out_eight[3] = static_cast<double>(g);
        out_eight[4] = log10_factorial_ratio(n, group_n);
        out_two[0] = 0;
        out_two[1] = 0;
        // F is undefined with one group, n == groups or an exactly zero
        // within SS; below the error estimate it is published and flagged.
        if (g < 2 || nn <= g) return N4M_OK;
        write_resolution(residual_error_estimate(range_ss, g, nn, kk), residual, &out_eight[6],
                         &out_eight[7]);
        if (!(residual > 0.0L)) return N4M_OK;
        const long double f_scale =
            static_cast<long double>(nn - g) / static_cast<long double>(g - 1);
        out_eight[0] = static_cast<double>(between / residual * f_scale);
        if (n_permutations == 0) return N4M_OK;
        // vegan permutest.betadisper: permute the ANOVA residuals of the
        // distances, refit the one-way model and count F_perm >= F_obs -
        // sqrt(eps). Between and within SS of each draw are explicit sums of
        // squares around its own group means, never a total minus a part.
        const double f_observed = out_eight[0];
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(nn);
        std::vector<long double> means(g);
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            std::fill(means.begin(), means.end(), 0.0L);
            long double mean_all = 0.0L;
            for (size_t i = 0; i < nn; ++i) {
                const long double v = resid[static_cast<size_t>(perm[i])];
                means[static_cast<size_t>(group[i])] += v;
                mean_all += v;
            }
            mean_all /= static_cast<long double>(n);
            for (size_t c = 0; c < g; ++c) means[c] /= static_cast<long double>(group_n[c]);
            long double bp = 0.0L, wp = 0.0L;
            for (size_t c = 0; c < g; ++c)
                bp += static_cast<long double>(group_n[c]) * (means[c] - mean_all) * (means[c] - mean_all);
            for (size_t i = 0; i < nn; ++i) {
                const long double d =
                    resid[static_cast<size_t>(perm[i])] - means[static_cast<size_t>(group[i])];
                wp += d * d;
            }
            const double f = wp > 0.0L ? static_cast<double>(bp / wp * f_scale)
                                       : std::numeric_limits<double>::infinity();
            if (f >= f_observed - tolerance) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_eight[5] = static_cast<double>(1 + at_least) / static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PERMDISP allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PERMDISP internal error"); }
}
N4M_API n4m_status_t n4m_stats_energy_distance(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int32_t n_groups, int32_t n_permutations, uint64_t seed, double* energy,
    double* energy_bounds, double* mean_distances, double* out_five, int64_t* out_three) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_codes || !energy || !energy_bounds || !mean_distances || !out_five || !out_three)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null energy distance buffer");
    const int64_t n = X->rows, p = X->cols;
    if (n < 2 || p < 1 || n_groups < 2 || n_groups > n || n_permutations < 0 ||
        n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid energy distance dimensions");
    // Every size is checked before any allocation.
    const long double pairs = static_cast<long double>(n) * static_cast<long double>(n - 1) / 2.0L;
    if (pairs > max_energy_pairs ||
        static_cast<long double>(n) * static_cast<long double>(p) > max_shift_cells ||
        static_cast<long double>(n_groups) * static_cast<long double>(n_groups) >
            max_energy_table ||
        pairs * (static_cast<long double>(p) + static_cast<long double>(n_permutations) + 1.0L) >
            max_energy_work)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "energy distance exceeds 20 million pairs, 50 million cells, 10 million "
                    "group pairs or 2.5 billion operations");
    if (!domain_of({X}).inside())
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        const auto g = static_cast<size_t>(n_groups), k = static_cast<size_t>(p);
        std::vector<int64_t> group_n(g, 0);
        for (int64_t i = 0; i < n; ++i) {
            if (group_codes[i] < 0 || group_codes[i] >= n_groups)
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "group code outside [0, n_groups)");
            ++group_n[static_cast<size_t>(group_codes[i])];
        }
        if (std::find(group_n.begin(), group_n.end(), int64_t{0}) != group_n.end())
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "every group needs at least one row");
        // Distances come from local pairwise differences of the input values:
        // no common centre is subtracted, so no difference is lost to another
        // row's magnitude.
        std::vector<double> x(static_cast<size_t>(n) * k);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < p; ++j)
                x[static_cast<size_t>(i) * k + static_cast<size_t>(j)] = at(*X, i, j);
        std::vector<double> dist(static_cast<size_t>(pairs));
        size_t idx = 0;
        for (int64_t i = 0; i < n; ++i)
            for (int64_t h = i + 1; h < n; ++h, ++idx)
                dist[idx] = difference_norm(x.data() + static_cast<size_t>(i) * k,
                                            x.data() + static_cast<size_t>(h) * k, k);
        EnergyKernel kernel{dist, group_n, n, std::vector<DD>(g * g)};
        kernel.accumulate(group_codes);
        for (size_t a = 0; a < g; ++a) for (size_t b = 0; b < g; ++b) {
            const DD m = kernel.mean_distance(a, b);
            mean_distances[a * g + b] = m.hi + m.lo;
            if (a == b) {
                energy[a * g + b] = 0.0;
                energy_bounds[a * g + b] = 0.0;
                continue;
            }
            // E >= 0 exactly; a computed value below -bound would mean the
            // bound does not hold, which is reported, never clipped.
            const DD e = kernel.energy(a, b);
            const double value = e.hi + e.lo, bound = kernel.energy_bound(a, b);
            if (value < -bound)
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "energy distance below minus its rounding bound");
            energy[a * g + b] = std::max(value, 0.0);
            energy_bounds[a * g + b] = bound;
        }
        const auto [observed, statistic_bound] = kernel.statistic();
        const double t_obs = observed.hi + observed.lo;
        if (t_obs < -statistic_bound)
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                        "energy statistic below minus its rounding bound");
        out_five[0] = std::max(t_obs, 0.0);
        out_five[1] = statistic_bound;
        out_five[2] = log10_factorial_ratio(n, group_n);
        out_five[3] = nan_value;
        out_five[4] = nan_value;
        out_three[0] = 0;
        out_three[1] = 0;
        out_three[2] = 0;
        for (size_t i = 0; i < g * g; ++i)
            if (!std::isfinite(energy[i]) || !std::isfinite(mean_distances[i]) ||
                !std::isfinite(energy_bounds[i]))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "energy distance not finite");
        if (n_permutations == 0) return N4M_OK;
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(static_cast<size_t>(n)), labels(static_cast<size_t>(n));
        // Declared tie rule: a draw counts when T_perm >= T - sqrt(eps)|T|.
        // A draw whose difference to that threshold is within 2 bound(T)
        // (the rounding of both statistics) is also counted as undecidable;
        // the conservative p-value counts every undecidable draw as a tie.
        const double threshold = -std::sqrt(std::numeric_limits<double>::epsilon()) * std::abs(t_obs);
        const double band = 2.0 * statistic_bound;
        int64_t at_least = 0, undecidable = 0, conservative = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            for (int64_t i = 0; i < n; ++i)
                labels[static_cast<size_t>(i)] = group_codes[perm[static_cast<size_t>(i)]];
            kernel.accumulate(labels.data());
            const DD diff = dd_add(kernel.statistic().first, dd_neg(observed));
            const double d = diff.hi + diff.lo;
            if (d >= threshold) ++at_least;
            if (std::abs(d - threshold) <= band) ++undecidable;
            if (d >= threshold - band) ++conservative;
        }
        out_three[0] = at_least;
        out_three[1] = n_permutations;
        out_three[2] = undecidable;
        const auto draws = static_cast<double>(1 + n_permutations);
        out_five[3] = static_cast<double>(1 + at_least) / draws;
        out_five[4] = static_cast<double>(1 + conservative) / draws;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "energy distance allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "energy distance internal error"); }
}

N4M_API n4m_status_t n4m_stats_pca_monitor(
    n4m_context_t* ctx, const n4m_matrix_view_t* reference, const n4m_matrix_view_t* monitored,
    int32_t max_components, int32_t n_folds, double alpha, double* eigenvalues,
    double* reference_t2, double* reference_q, double* monitored_t2, double* monitored_q,
    int32_t* effective_components, int64_t* out_counts, double* out_thirteen) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    for (const auto* v : {reference, monitored}) {
        const auto st = check_view(ctx, v);
        if (st != N4M_OK) return st;
    }
    if (!eigenvalues || !reference_t2 || !reference_q || !monitored_t2 || !monitored_q ||
        !effective_components || !out_counts || !out_thirteen)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null PCA monitor buffer");
    const int64_t nr = reference->rows, nm = monitored->rows, p = reference->cols;
    if (nr < 3 || nm < 1 || p < 1 || monitored->cols != p || n_folds < 2 || n_folds > nr ||
        !(alpha > 0.0 && alpha < 1.0))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PCA monitor dimensions");
    const int64_t largest_fold = (nr + n_folds - 1) / n_folds;
    const int64_t smallest_train = nr - largest_fold;
    if (max_components < 1 || max_components > p || max_components > smallest_train - 1)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "max_components must not exceed the columns or the smallest training "
                    "fold minus one");
    const long double m = static_cast<long double>(std::min(nr, p));
    const long double work =
        (static_cast<long double>(n_folds) + 1.0L) * m * m *
            static_cast<long double>(std::max(nr, p)) +
        static_cast<long double>(nr + nm) * static_cast<long double>(max_components) *
            static_cast<long double>(p) * 4.0L;
    if (work > max_monitor_work ||
        static_cast<long double>(nr + nm) * static_cast<long double>(p) > max_shift_cells)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "PCA monitor exceeds 10 billion operations or 50 million cells");
    if (!domain_of({reference, monitored}).inside())
        return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    try {
        const auto P = static_cast<size_t>(p), NR = static_cast<size_t>(nr),
                   NM = static_cast<size_t>(nm);
        std::vector<double> ref(NR * P), mon(NM * P);
        for (size_t i = 0; i < NR; ++i)
            for (size_t j = 0; j < P; ++j)
                ref[i * P + j] = at(*reference, static_cast<int64_t>(i), static_cast<int64_t>(j));
        for (size_t i = 0; i < NM; ++i)
            for (size_t j = 0; j < P; ++j)
                mon[i * P + j] = at(*monitored, static_cast<int64_t>(i), static_cast<int64_t>(j));
        const int64_t kmax = max_components;
        // Fits on each training fold (row i in fold i mod n_folds) and on all
        // reference rows, each centred on its own double-double mean; the
        // effective k is the smallest number of resolved components.
        std::vector<std::unique_ptr<ReferencePca>> folds;
        int64_t k = kmax;
        for (int32_t f = 0; f < n_folds; ++f) {
            std::vector<size_t> train;
            for (size_t i = 0; i < NR; ++i)
                if (static_cast<int64_t>(i) % n_folds != f) train.push_back(i);
            folds.push_back(std::make_unique<ReferencePca>());
            const auto st = folds.back()->compute(ref, P, train, kmax);
            if (st != N4M_OK) return fail(ctx, st, "PCA monitor fold fit failed");
            k = std::min(k, folds.back()->resolved_components());
        }
        std::vector<size_t> all(NR);
        std::iota(all.begin(), all.end(), size_t{0});
        ReferencePca full;
        auto st = full.compute(ref, P, all, kmax);
        if (st != N4M_OK) return fail(ctx, st, "PCA monitor reference fit failed");
        k = std::min(k, full.resolved_components());
        *effective_components = static_cast<int32_t>(k);
        // Conservative relative tolerance of T^2 and Q (not a proven bound):
        // tau = 16 m eps cond, m = max(n_r, p), cond = lambda_1 / lambda_k
        // of the full fit (1 when k == 0).
        const double m_dim = static_cast<double>(std::max(nr, p));
        const double cond = k > 0 ? full.fit.explained_var[0] / full.fit.explained_var[k - 1] : 1.0;
        const double tau = 16.0 * m_dim * std::numeric_limits<double>::epsilon() * cond;
        std::vector<double> scores(static_cast<size_t>(kmax)), residual, xc(P);
        std::vector<RowStatistics> rs(NR);
        double scale = 0.0;
        for (int32_t f = 0; f < n_folds; ++f)
            for (size_t i = static_cast<size_t>(f); i < NR; i += static_cast<size_t>(n_folds)) {
                const auto& fold = *folds[static_cast<size_t>(f)];
                centre(ref.data() + i * P, fold.mean, xc.data());
                rs[i] = fold.evaluate(xc.data(), k, scores.data(), residual);
                scale = std::max(scale, rs[i].norm2);
            }
        auto quantile_of = [alpha, &rs](double RowStatistics::*field) {
            std::vector<double> values;
            for (const auto& r : rs) values.push_back(r.*field);
            std::sort(values.begin(), values.end());
            return quantile(values, 1.0 - alpha);
        };
        const double t2_limit = k > 0 ? quantile_of(&RowStatistics::t2) : nan_value;
        const double q_limit = quantile_of(&RowStatistics::q);
        int64_t counts[6] = {0, 0, 0, 0, 0, 0};
        double q_scale = scale;
        for (size_t i = 0; i < NM; ++i) {
            centre(mon.data() + i * P, full.mean, xc.data());
            const auto s = full.evaluate(xc.data(), k, scores.data(), residual);
            monitored_t2[i] = s.t2;
            monitored_q[i] = s.q;
            // Q is compared on the scale of the larger of this row's and the
            // reference rows' squared centred norms.
            const double row_scale = std::max(scale, s.norm2);
            q_scale = std::max(q_scale, s.norm2);
            const int over_t2 = k > 0 ? compare_with_tolerance(
                                            s.t2, t2_limit, tau * std::max(s.t2, t2_limit))
                                      : 0;
            const int over_q = compare_with_tolerance(s.q, q_limit, tau * row_scale);
            const int either = (over_t2 == 1 || over_q == 1) ? 1
                               : (over_t2 == 0 && over_q == 0) ? 0 : -1;
            counts[0] += over_t2 == 1;
            counts[1] += over_q == 1;
            counts[2] += either == 1;
            counts[3] += over_t2 == -1;
            counts[4] += over_q == -1;
            counts[5] += either == -1;
        }
        for (int i = 0; i < 6; ++i) out_counts[i] = counts[i];
        for (size_t i = 0; i < NR; ++i) {
            reference_t2[i] = rs[i].t2;
            reference_q[i] = rs[i].q;
        }
        // Dispersions from the raw data, each set about its own centre.
        auto trace = [P](const std::vector<double>& x, size_t n) {
            std::vector<size_t> rows(n);
            std::iota(rows.begin(), rows.end(), size_t{0});
            const auto mean = dd_means(x, P, rows);
            std::vector<double> c(P);
            DD sum;
            for (size_t i = 0; i < n; ++i) {
                centre(x.data() + i * P, mean, c.data());
                sum = dd_add(sum, squared_norm(c.data(), P));
            }
            return (sum.hi + sum.lo) / static_cast<double>(n - 1);
        };
        const double trace_ref = trace(ref, NR);
        const double trace_mon = nm >= 2 ? trace(mon, NM) : nan_value;
        // Centroid shift: difference of the double-double means.
        std::vector<size_t> mon_rows(NM);
        std::iota(mon_rows.begin(), mon_rows.end(), size_t{0});
        const auto mon_mean = dd_means(mon, P, mon_rows);
        std::vector<double> diff(P);
        for (size_t j = 0; j < P; ++j) {
            const DD d = dd_add(mon_mean[j], dd_neg(full.mean[j]));
            diff[j] = d.hi + d.lo;
        }
        const auto c = full.evaluate(diff.data(), k, scores.data(), residual);
        double subspace2 = 0.0, retained = 0.0;
        for (int64_t a = 0; a < k; ++a) {
            subspace2 += scores[static_cast<size_t>(a)] * scores[static_cast<size_t>(a)];
            retained += full.fit.explained_var[a];
            eigenvalues[a] = full.fit.explained_var[a];
        }
        for (int64_t a = k; a < kmax; ++a) eigenvalues[a] = nan_value;
        out_thirteen[0] = t2_limit;
        out_thirteen[1] = q_limit;
        out_thirteen[2] = trace_ref;
        out_thirteen[3] = trace_mon;
        out_thirteen[4] = std::sqrt(squared_norm(diff.data(), P));
        out_thirteen[5] = std::sqrt(subspace2);
        out_thirteen[6] = std::sqrt(c.q);
        out_thirteen[7] = c.t2;
        out_thirteen[8] = trace_ref > 0.0 ? retained / trace_ref : nan_value;
        out_thirteen[9] = tau;
        out_thirteen[10] = cond;
        out_thirteen[11] = q_scale;
        out_thirteen[12] = full.resolution();
        bool finite = true;
        for (int i : {1, 2, 4, 5, 6, 9, 10, 11, 12}) finite = finite && std::isfinite(out_thirteen[i]);
        for (size_t i = 0; i < NR; ++i) finite = finite && std::isfinite(reference_q[i]);
        for (size_t i = 0; i < NM; ++i) finite = finite && std::isfinite(monitored_q[i]);
        if (k > 0) {
            for (int i : {0, 7}) finite = finite && std::isfinite(out_thirteen[i]);
            for (size_t i = 0; i < NR; ++i) finite = finite && std::isfinite(reference_t2[i]);
            for (size_t i = 0; i < NM; ++i) finite = finite && std::isfinite(monitored_t2[i]);
        }
        if (!finite) return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "PCA monitor result not finite");
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PCA monitor allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PCA monitor internal error"); }
}

N4M_API n4m_status_t n4m_stats_group_means(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int64_t n_groups, double* out_means, int64_t* out_counts) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_codes || !out_means || !out_counts)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null group-means buffer");
    const int64_t n = X->rows, k = X->cols;
    if (n < 1 || k < 1 || n_groups < 1 || n_groups > n)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid group-means dimensions");
    if (static_cast<uint64_t>(n_groups) > std::numeric_limits<size_t>::max() / sizeof(DD) /
                                                static_cast<uint64_t>(k))
        return fail(ctx, N4M_ERR_UNSUPPORTED, "group-means output exceeds the address space");
    try {
        const auto g = static_cast<size_t>(n_groups), kk = static_cast<size_t>(k);
        std::vector<int64_t> count(g, 0);
        for (int64_t i = 0; i < n; ++i) {
            if (group_codes[i] < 0 || group_codes[i] >= n_groups)
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "group code outside [0, n_groups)");
            ++count[static_cast<size_t>(group_codes[i])];
        }
        if (std::find(count.begin(), count.end(), 0) != count.end())
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "every group needs at least one row");
        DomainRange scan;
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < k; ++j) scan.add(at(*X, i, j));
        if (!scan.finite)
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite group-means input");
        // A mean is a summary: only the magnitude bounds apply (rule 15).
        if (!scan.magnitudes_inside())
            return fail(ctx, N4M_ERR_UNSUPPORTED, magnitude_domain_message);
        // Inside D no partial sum can overflow (n * 2^256); the double-double
        // sum and one division give a mean whose relative error is small
        // (order n eps) for same-sign data and whose absolute error is order
        // n eps max|x| otherwise.
        std::vector<DD> sum(g * kk);
        for (int64_t i = 0; i < n; ++i) {
            const auto c = static_cast<size_t>(group_codes[i]);
            for (size_t j = 0; j < kk; ++j)
                sum[c * kk + j] = dd_add(sum[c * kk + j], at(*X, i, static_cast<int64_t>(j)));
        }
        for (size_t c = 0; c < g; ++c) {
            out_counts[c] = count[c];
            for (size_t j = 0; j < kk; ++j)
                out_means[c * kk + j] = dd_div(sum[c * kk + j], static_cast<double>(count[c])).hi;
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "group-means allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "group-means internal error"); }
}

N4M_API n4m_status_t n4m_stats_regression_scores(
    n4m_context_t* ctx, const double* y_true, const double* y_pred, int64_t n,
    double* out_four) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!y_true || !y_pred || !out_four)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null regression-scores buffer");
    if (n < 1) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "regression scores need n >= 1");
    try {
        DomainRange truth, pred;
        for (int64_t i = 0; i < n; ++i) {
            truth.add(y_true[i]);
            pred.add(y_pred[i]);
        }
        if (!truth.finite || !pred.finite)
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite regression-scores input");
        if (!truth.inside() || !pred.inside()) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
        // Residual e_i = y_i - yhat_i: |e| <= 2^257, e^2 <= 2^514, so the
        // double-double sums of e^2 and |e| never overflow inside D.
        DD sse, sae, sum_y;
        for (int64_t i = 0; i < n; ++i) {
            const double e = y_true[i] - y_pred[i];
            sse = dd_add(sse, e * e);
            sae = dd_add(sae, std::abs(e));
            sum_y = dd_add(sum_y, y_true[i]);
        }
        const double nd = static_cast<double>(n);
        // Mean of y as an unevaluated sum hi + lo; deviations are explicit.
        const double mean_hi = dd_div(sum_y, nd).hi;
        DD mean_err;
        for (int64_t i = 0; i < n; ++i) mean_err = dd_add(mean_err, y_true[i] - mean_hi);
        const double mean_lo = dd_div(mean_err, nd).hi;
        // With spread <= 2^32 the deviations keep their difference (no
        // cancellation by offset); RMSE, MAE, SSE and SST have relative errors
        // of order n eps, R^2 an absolute error of order n eps (1 + SSE/SST).
        DD sst;
        for (int64_t i = 0; i < n; ++i) {
            const double d = (y_true[i] - mean_hi) - mean_lo;
            sst = dd_add(sst, d * d);
        }
        const double sse_v = sse.hi + sse.lo, sst_v = sst.hi + sst.lo;
        out_four[0] = std::sqrt(dd_div(sse, nd).hi);
        out_four[1] = dd_div(sae, nd).hi;
        // R^2 = 1 - SSE/SST as significand * 2^exponent. The ratio of two
        // finite sums can exceed the double range inside D (SSE ~ 2^514,
        // SST ~ 2^-511); it is then published scaled, never as -Inf.
        out_four[2] = std::numeric_limits<double>::quiet_NaN();
        out_four[3] = 0.0;
        if (sst_v > 0.0) {
            int es = 0, et = 0;
            const double ms = std::frexp(sse_v, &es), mt = std::frexp(sst_v, &et);
            const int exponent = es - et;
            if (sse_v == 0.0 || exponent < 1000) {
                out_four[2] = 1.0 - sse_v / sst_v;
            } else {
                // 1 is below the resolution of SSE/SST >= 2^998: R^2 = -SSE/SST.
                int e = 0;
                out_four[2] = -std::frexp(ms / mt, &e);
                out_four[3] = static_cast<double>(exponent + e);
            }
        }
        return N4M_OK;
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "regression-scores internal error"); }
}
}  // extern "C"
