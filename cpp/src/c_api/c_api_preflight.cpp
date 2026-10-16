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

#include "core/common/context.hpp"
#include "core/common/matrix_view.hpp"
#include "core/clustering/_vendored/hdbscan.hpp"
#include "core/common/pca_helper.h"
#include "core/common/rng_pcg64.h"
#include "core/preprocessing/derivatives/savitzky_golay.h"

namespace {
using ::n4m::core::Context;
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr long double max_moment_products = 250000000.0L;
constexpr long double max_hdbscan_products = 2500000000.0L;
// Contingency-table budget of the categorical association kernels (cells).
constexpr size_t max_table_cells = 10000000;

// Supported numerical domain D (preflight robustness rules 13 and 15): finite values,
// every nonzero |x| in [2^-256, 2^256] and, inside one input array,
// max nonzero |x| / min nonzero |x| <= 2^N4M_DOMAIN_MAX_SPREAD_LOG2 (metrics.h).
// Exact zeros are always allowed.
// Inputs outside D are refused before any computation.
constexpr double domain_low = 0x1p-256;
constexpr double domain_high = 0x1p256;
constexpr double domain_spread = static_cast<double>(1ULL << N4M_DOMAIN_MAX_SPREAD_LOG2);
constexpr const char* domain_message =
    "input outside the supported numerical domain (nonzero |x| in [2^-256, 2^256], "
    "max/min nonzero |x| <= 2^32)";
static_assert(N4M_DOMAIN_MAX_SPREAD_LOG2 == 32, "update domain_message with the spread");

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

// Smallest and largest nonzero |x| of finite values, tested against D.
struct DomainRange {
    double low = domain_high, high = 0.0;
    void add(double x) noexcept {
        const double a = std::abs(x);
        if (a == 0.0) return;
        low = std::min(low, a);
        high = std::max(high, a);
    }
    bool magnitudes_inside() const noexcept {
        return high == 0.0 || (low >= domain_low && high <= domain_high);
    }
    bool inside() const noexcept {
        return magnitudes_inside() && (high == 0.0 || high <= low * domain_spread);
    }
};

// One O(n) pre-pass; inputs are already known to be finite.
bool in_numerical_domain(const double* values, size_t n) noexcept {
    DomainRange range;
    for (size_t i = 0; i < n; ++i) range.add(values[i]);
    return range.inside();
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

double at(const n4m_matrix_view_t& v, int64_t i, int64_t j) noexcept {
    return static_cast<const double*>(v.data)[i * v.row_stride + j * v.col_stride];
}
void put(const n4m_matrix_view_t& v, int64_t i, int64_t j, double x) noexcept {
    static_cast<double*>(v.data)[i * v.row_stride + j * v.col_stride] = x;
}

double quantile(const std::vector<double>& sorted, double q) noexcept {
    const double h = static_cast<double>(sorted.size() - 1) * q;
    const auto lo = static_cast<size_t>(h);
    const auto hi = std::min(lo + 1, sorted.size() - 1);
    const long double weight = static_cast<long double>(h) -
                               static_cast<long double>(lo);
    return static_cast<double>((1.0L - weight) * static_cast<long double>(sorted[lo]) +
                               weight * static_cast<long double>(sorted[hi]));
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

// Exponent e of a power of two bounding max|v|: max|v| * 2^-e in [1/2, 1).
int bounding_exponent(const std::vector<double>& values) noexcept {
    double largest = 0.0;
    for (double v : values) largest = std::max(largest, std::abs(v));
    int exponent = 0;
    if (largest > 0.0) std::frexp(largest, &exponent);
    return exponent;
}

// In-place centering without cancellation of a large common offset: shift
// by the first value (exact for nearby values), subtract the mean, then the
// residual mean.
void center_in_place(std::vector<double>& v) noexcept {
    if (v.empty()) return;
    const double origin = v[0];
    double mean = 0.0;
    for (double& x : v) {
        x -= origin;
        mean += x;
    }
    mean /= static_cast<double>(v.size());
    double residual = 0.0;
    for (double& x : v) {
        x -= mean;
        residual += x;
    }
    residual /= static_cast<double>(v.size());
    for (double& x : v) x -= residual;
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

// D for a coordinate matrix: one input array, so the spread is over all its
// entries. One O(n k) pre-pass without a copy; nonfinite values are left to
// the caller's own validation.
bool in_numerical_domain(const n4m_matrix_view_t& v) noexcept {
    DomainRange range;
    for (int64_t j = 0; j < v.cols; ++j)
        for (int64_t i = 0; i < v.rows; ++i) {
            const double x = at(v, i, j);
            if (!std::isfinite(x)) return true;
            range.add(x);
        }
    return range.inside();
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

// Difference domain of the domain-shift kernels (rule 15) over every input
// array together; nonfinite values are outside it.
bool in_difference_domain(std::initializer_list<const n4m_matrix_view_t*> views) noexcept {
    DomainRange range;
    for (const auto* v : views)
        for (int64_t i = 0; i < v->rows; ++i)
            for (int64_t j = 0; j < v->cols; ++j) {
                const double x = at(*v, i, j);
                if (!std::isfinite(x)) return false;
                range.add(x);
            }
    return range.inside();
}

// Double-double arithmetic (Dekker / Knuth error-free transformations).
struct DD {
    double hi = 0.0, lo = 0.0;
};
DD two_sum(double a, double b) noexcept {
    const double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
DD quick_two_sum(double a, double b) noexcept {
    const double s = a + b;
    return {s, b - (s - a)};
}
DD dd_add(DD x, double y) noexcept {
    DD s = two_sum(x.hi, y);
    s.lo += x.lo;
    return quick_two_sum(s.hi, s.lo);
}
DD dd_add(DD x, DD y) noexcept {
    DD s = two_sum(x.hi, y.hi);
    const DD t = two_sum(x.lo, y.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}
DD dd_neg(DD x) noexcept { return {-x.hi, -x.lo}; }
DD dd_mul(DD x, double y) noexcept {
    const double p = x.hi * y;
    const double e = std::fma(x.hi, y, -p);
    return quick_two_sum(p, e + x.lo * y);
}
DD dd_div(DD x, double y) noexcept {
    const double q1 = x.hi / y;
    const DD r = dd_add(x, dd_neg(dd_mul(DD{q1, 0.0}, y)));
    return quick_two_sum(q1, r.hi / y);
}

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

extern "C" {
N4M_API n4m_status_t n4m_preflight_pca_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, int32_t k,
    const n4m_matrix_view_t* scores, const n4m_matrix_view_t* loadings,
    double* eigenvalues, double* full_moments) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    if (!eigenvalues || !full_moments)
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
    const long double m = static_cast<long double>(std::min(n, p));
    const long double products = m * m * static_cast<long double>(std::max(n, p));
    if (products > max_moment_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "full PCA moments exceed 250 million product budget");
    try {
        if (static_cast<long double>(n) * p > static_cast<long double>(SIZE_MAX / sizeof(double)))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "PCA matrix size overflow");
        std::vector<double> data(static_cast<size_t>(n) * static_cast<size_t>(p));
        std::vector<double> means(static_cast<size_t>(p), 0.0);
        for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j) {
            const double x = at(*X, i, j);
            if (!std::isfinite(x))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PCA input");
            data[static_cast<size_t>(i) * static_cast<size_t>(p) + static_cast<size_t>(j)] = x;
            means[static_cast<size_t>(j)] += x / static_cast<double>(n);
        }
        n4m_pca_fit_t fit{};
        const auto st = n4m_pca_fit(data.data(), n, p, k, 1, &fit);
        if (st != N4M_OK) return fail(ctx, st, "PCA fit failed");
        for (int64_t a = 0; a < k; ++a) {
            if (!std::isfinite(fit.explained_var[a])) {
                n4m_pca_fit_free(&fit);
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "nonfinite PCA eigenvalue");
            }
        }
        for (int64_t i = 0; i < n; ++i) for (int64_t a = 0; a < k; ++a)
            put(*scores, i, a, fit.scores[i * k + a]);
        for (int64_t a = 0; a < k; ++a) {
            eigenvalues[a] = fit.explained_var[a];
            for (int64_t j = 0; j < p; ++j) put(*loadings, a, j, fit.components[a * p + j]);
        }
        n4m_pca_fit_free(&fit);
        for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j)
            data[static_cast<size_t>(i) * static_cast<size_t>(p) +
                 static_cast<size_t>(j)] -= means[static_cast<size_t>(j)];
        long double sum1 = 0.0L, sum2 = 0.0L;
        for (double x : data)
            sum1 += static_cast<long double>(x) * static_cast<long double>(x);
        // trace(C^2) equals the squared Frobenius norm of either Gram matrix.
        if (n <= p) {
            for (int64_t i = 0; i < n; ++i) for (int64_t h = 0; h <= i; ++h) {
                long double dot = 0.0L;
                for (int64_t j = 0; j < p; ++j)
                    dot += static_cast<long double>(data[static_cast<size_t>(i) *
                                                           static_cast<size_t>(p) +
                                                       static_cast<size_t>(j)]) *
                           static_cast<long double>(
                               data[static_cast<size_t>(h) * static_cast<size_t>(p) +
                                    static_cast<size_t>(j)]);
                sum2 += (i == h ? 1.0L : 2.0L) * dot * dot;
            }
        } else {
            for (int64_t j = 0; j < p; ++j) for (int64_t h = 0; h <= j; ++h) {
                long double dot = 0.0L;
                for (int64_t i = 0; i < n; ++i)
                    dot += static_cast<long double>(data[static_cast<size_t>(i) *
                                                           static_cast<size_t>(p) +
                                                       static_cast<size_t>(j)]) *
                           static_cast<long double>(
                               data[static_cast<size_t>(i) * static_cast<size_t>(p) +
                                    static_cast<size_t>(h)]);
                sum2 += (j == h ? 1.0L : 2.0L) * dot * dot;
            }
        }
        const long double d = static_cast<long double>(n - 1);
        full_moments[0] = static_cast<double>(sum1 / d);
        full_moments[1] = static_cast<double>(sum2 / (d * d));
        if (!std::isfinite(full_moments[0]) || !std::isfinite(full_moments[1]))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "PCA moments overflow");
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PCA allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PCA internal error"); }
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
        for (int64_t i = 0; i < n; ++i) {
            if (valid_mask && !valid_mask[i]) continue;
            if (!std::isfinite(values[i]))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked summary input");
            v.push_back(values[i]);
        }
        std::fill(out_eight, out_eight + 8, nan_value);
        out_eight[0] = static_cast<double>(v.size());
        if (v.empty()) return N4M_OK;
        double scale = 0.0;
        for (double x : v) scale = std::max(scale, std::abs(x));
        if (scale == 0.0) scale = 1.0;
        long double mean = 0.0L, ss = 0.0L;
        for (size_t i = 0; i < v.size(); ++i) {
            const long double x = static_cast<long double>(v[i]) /
                                  static_cast<long double>(scale);
            mean += (x - mean) / (i + 1);
        }
        for (double x : v) {
            const long double d = static_cast<long double>(x) /
                                  static_cast<long double>(scale) - mean;
            ss += d * d;
        }
        std::sort(v.begin(), v.end());
        out_eight[1] = v.front(); out_eight[2] = v.back();
        out_eight[3] = static_cast<double>(static_cast<long double>(scale) * mean);
        if (v.size() >= 2)
            out_eight[4] = static_cast<double>(
                static_cast<long double>(scale) * std::sqrt(ss / (v.size() - 1)));
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
            for (int64_t i = 0; i < v->rows; ++i)
                for (int64_t j = 0; j < v->cols; ++j)
                    if (!std::isfinite(at(*v, i, j)))
                        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite neighbor input");
        // Keep only k candidates per query; ties resolve to smaller index.
        for (int64_t i = 0; i < query->rows; ++i) {
            for (int32_t a = 0; a < k; ++a) {
                indices[i * k + a] = -1; distances[i * k + a] = nan_value;
            }
            for (int64_t h = 0; h < reference->rows; ++h) {
                if (query_groups && query_groups[i] == reference_groups[h]) continue;
                long double scale = 0.0L, ssq = 1.0L;
                for (int64_t j = 0; j < query->cols; ++j) {
                    const double q = at(*query, i, j), r = at(*reference, h, j);
                    const long double d = static_cast<long double>(q) -
                                          static_cast<long double>(r);
                    const long double magnitude = std::abs(d);
                    if (!std::isfinite(magnitude))
                        return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                    "neighbor difference is not representable");
                    if (magnitude > scale) {
                        const long double ratio = scale / magnitude;
                        ssq = 1.0L + ssq * ratio * ratio;
                        scale = magnitude;
                    } else if (magnitude > 0.0L) {
                        const long double ratio = magnitude / scale;
                        ssq += ratio * ratio;
                    }
                }
                const double d = static_cast<double>(scale * std::sqrt(ssq));
                if (!std::isfinite(d))
                    return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "neighbor distance overflow");
                int32_t pos = 0;
                while (pos < k && indices[i * k + pos] >= 0 &&
                       (distances[i * k + pos] < d ||
                        (distances[i * k + pos] == d && indices[i * k + pos] < h))) ++pos;
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
        long double mean = 0, m2 = 0;
        constexpr double max_exact_category = 9007199254740991.0;
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
                const long double delta = static_cast<long double>(y) - mean;
                mean += delta / (i + 1);
                m2 += delta * (static_cast<long double>(y) - mean);
            }
        }
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
        long double row_sum = 0, expected_sum = 0;
        int64_t eligible_rows = 0;
        for (int64_t i = 0; i < n; ++i) {
            long double local = 0;
            int32_t found = 0;
            for (int32_t a = 0; a < k; ++a) {
                const int64_t j = neighbor_indices[i * k + a];
                if (j == -1) continue;
                if (j < 0 || j >= n || j == i ||
                    (group_ids && group_ids[i] == group_ids[j]))
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                                "invalid or nonindependent neighbor index");
                if (categorical) {
                    local += categories[static_cast<size_t>(i)] !=
                             categories[static_cast<size_t>(j)] ? 1.0L : 0.0L;
                } else {
                    const long double delta = static_cast<long double>(targets[i]) -
                                              static_cast<long double>(targets[j]);
                    local += delta * delta;
                }
                ++found;
            }
            if (found > 0) {
                row_sum += local / found;
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
                    expected_sum += 1.0L - static_cast<long double>(same) / admissible;
                }
            }
        }
        out_four[0] = nan_value;
        out_four[1] = static_cast<double>(eligible_rows) / static_cast<double>(n);
        out_four[2] = nan_value;
        out_four[3] = nan_value;
        if (eligible_rows == 0) return N4M_OK;
        const long double observed = row_sum / eligible_rows;
        if (!categorical) {
            if (n < 2 || m2 <= 0) return N4M_OK;
            out_four[0] = static_cast<double>(1.0L - observed / (2.0L * m2 / (n - 1)));
        } else {
            out_four[2] = static_cast<double>(1.0L - observed);
            out_four[3] = static_cast<double>(1.0L - expected_sum / eligible_rows);
            if (expected_sum <= 0) return N4M_OK;
            out_four[0] = static_cast<double>(1.0L - row_sum / expected_sum);
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
    if (static_cast<long double>(n) * n * p > max_hdbscan_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED, "HDBSCAN exceeds 2.5 billion distance-product budget");
    try {
        std::vector<double> data(static_cast<size_t>(n) * static_cast<size_t>(p));
        for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < p; ++j) {
            const double x = at(*X, i, j);
            if (!std::isfinite(x)) return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite HDBSCAN input");
            data[static_cast<size_t>(i) * static_cast<size_t>(p) + static_cast<size_t>(j)] = x;
        }
        ::n4m::core::hdbscan::Params params;
        params.min_cluster_size = min_cluster_size;
        params.min_samples = min_samples;
        params.selection = static_cast<::n4m::core::hdbscan::Selection>(selection_method);
        params.allow_single_cluster = allow_single_cluster != 0;
        const auto result = ::n4m::core::hdbscan::fit(data.data(), n, p, params);
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
    try {
        *ari = nan_value;
        if (n < 2) return N4M_OK;
        std::vector<std::pair<int64_t, int64_t>> pairs(static_cast<size_t>(n));
        std::vector<int64_t> a(labels_a, labels_a + n), b(labels_b, labels_b + n);
        for (int64_t i = 0; i < n; ++i) pairs[static_cast<size_t>(i)] = {labels_a[i], labels_b[i]};
        // Sum of squared run lengths of a sorted sequence.
        auto squared_runs = [](auto& values) {
            std::sort(values.begin(), values.end());
            long double total = 0.0L;
            for (size_t i = 0; i < values.size();) {
                size_t j = i;
                while (j < values.size() && values[j] == values[i]) ++j;
                total += static_cast<long double>(j - i) * static_cast<long double>(j - i);
                i = j;
            }
            return total;
        };
        // scikit-learn pair_confusion_matrix / adjusted_rand_score.
        const long double squares = squared_runs(pairs);
        const long double rows = squared_runs(a), cols = squared_runs(b);
        const long double total = static_cast<long double>(n);
        const long double tp = squares - total;
        const long double fp = cols - squares;
        const long double fn = rows - squares;
        const long double tn = total * total - fp - fn - squares;
        const long double denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn);
        if (denominator == 0.0L) return N4M_OK;
        *ari = static_cast<double>(2.0L * (tp * tn - fn * fp) / denominator);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "ARI allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "ARI internal error"); }
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
    if (window_length < 3 || window_length % 2 == 0 || polyorder < 0 ||
        polyorder > window_length - 2)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "noise proxy needs an odd window >= 3 and polyorder <= window-2");
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
        // Residual gain from the filter's own impulse response: interior
        // outputs of a centred unit impulse of length 2w-1 are the weights.
        const int64_t impulse_cols = 2 * static_cast<int64_t>(window_length) - 1;
        std::vector<double> impulse(sz(impulse_cols), 0.0), response(sz(impulse_cols));
        impulse[sz(window_length - 1)] = 1.0;
        auto sst = n4m_pp_savgol_state_apply(sg, impulse.data(), 1, impulse_cols, response.data());
        if (sst != N4M_OK) return fail(ctx, sst, "Savitzky-Golay impulse response failed");
        long double gain2 = 0.0L;
        for (int64_t j = h; j < impulse_cols - h; ++j) {
            const long double d = static_cast<long double>(impulse[sz(j)]) -
                                  static_cast<long double>(response[sz(j)]);
            gain2 += d * d;
        }
        const double gain = static_cast<double>(std::sqrt(gain2));
        if (!(gain > 0.0) || !std::isfinite(gain))
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "zero residual gain");
        std::vector<double> data(sz(n) * sz(p)), smooth(sz(n) * sz(p));
        std::vector<int64_t> invalid_prefix(sz(p) + 1);
        std::vector<uint8_t> evaluable(sz(n) * sz(p), 0);
        for (int64_t i = 0; i < n; ++i) {
            invalid_prefix[0] = 0;
            for (int64_t j = 0; j < p; ++j) {
                const bool valid = !valid_mask || valid_mask[sz(i) * sz(p) + sz(j)];
                double x = 0.0;
                if (valid) {
                    x = at(*X, i, j);
                    if (!std::isfinite(x))
                        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked noise proxy input");
                }
                data[sz(i) * sz(p) + sz(j)] = x;
                invalid_prefix[sz(j) + 1] = invalid_prefix[sz(j)] + (valid ? 0 : 1);
            }
            for (int64_t j = h; j < p - h; ++j)
                evaluable[sz(i) * sz(p) + sz(j)] =
                    invalid_prefix[sz(j + h) + 1] == invalid_prefix[sz(j - h)] ? 1 : 0;
        }
        sst = n4m_pp_savgol_state_apply(sg, data.data(), n, p, smooth.data());
        if (sst != N4M_OK) return fail(ctx, sst, "Savitzky-Golay smoothing failed");
        double scale = 0.0;
        for (size_t c = 0; c < data.size(); ++c) {
            if (!evaluable[c]) continue;
            smooth[c] = data[c] - smooth[c];
            if (!std::isfinite(smooth[c]))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "noise proxy residual overflow");
            scale = std::max(scale, std::abs(smooth[c]));
        }
        if (scale == 0.0) scale = 1.0;
        const long double lscale = static_cast<long double>(scale);
        const long double lgain = static_cast<long double>(gain);
        long double pooled = 0.0L;
        int64_t pooled_count = 0;
        for (int64_t j = 0; j < p; ++j) {
            long double sum = 0.0L;
            int64_t count = 0;
            for (int64_t i = 0; i < n; ++i) {
                const size_t c = sz(i) * sz(p) + sz(j);
                if (!evaluable[c]) continue;
                const long double r = static_cast<long double>(smooth[c]) / lscale;
                sum += r * r;
                ++count;
            }
            band_counts[j] = count;
            band_noise[j] = count ? static_cast<double>(
                lscale * std::sqrt(sum / static_cast<long double>(count)) / lgain) : nan_value;
            pooled += sum;
            pooled_count += count;
        }
        out_two[0] = pooled_count ? static_cast<double>(
            lscale * std::sqrt(pooled / static_cast<long double>(pooled_count)) / lgain) : nan_value;
        out_two[1] = gain;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "noise proxy allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "noise proxy internal error"); }
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
        for (int64_t i = 0; i < v->rows; ++i)
            for (int64_t j = 0; j < v->cols; ++j)
                if (!std::isfinite(at(*v, i, j)))
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite pair input");
    std::fill(query_flags, query_flags + query->rows, uint8_t{0});
    if (!self) std::fill(reference_flags, reference_flags + R.rows, uint8_t{0});
    uint8_t* rflags = self ? query_flags : reference_flags;
    // Compare sum(((q-r)/tolerance)^2) with cols: overflow means "far".
    const double limit = static_cast<double>(query->cols);
    int64_t found = 0, candidates = 0;
    for (int64_t i = 0; i < query->rows; ++i) {
        for (int64_t k = self ? i + 1 : 0; k < R.rows; ++k) {
            if (query_groups && query_groups[i] == rgroups[k]) continue;
            ++candidates;
            double s = 0.0;
            for (int64_t j = 0; j < query->cols && s <= limit; ++j) {
                const double t = (at(*query, i, j) - at(R, k, j)) / tolerance;
                s += t * t;
            }
            if (!(s <= limit)) continue;
            query_flags[i] = 1;
            rflags[k] = 1;
            if (found < max_pairs) {
                pair_indices[2 * found] = i;
                pair_indices[2 * found + 1] = k;
                pair_distances[found] = tolerance * std::sqrt(s / limit);
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
    if (!in_numerical_domain(*X)) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
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
    if (!in_numerical_domain(*X)) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
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
    if (!in_numerical_domain(*X)) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
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
    if (!in_difference_domain({X}))
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
    if (!in_difference_domain({reference, monitored}))
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
}  // extern "C"
