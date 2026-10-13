// SPDX-License-Identifier: CECILL-2.1
// Reusable, caller-buffer diagnostics for Preflight and other C ABI clients.
#include "n4m/metrics.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
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

// Supported numerical domain D (preflight robustness rule 13): finite values,
// every nonzero |x| in [2^-256, 2^256] and, inside one input array,
// max nonzero |x| / min nonzero |x| <= 2^64. Exact zeros are always allowed.
// Inputs outside D are refused before any computation.
constexpr double domain_low = 0x1p-256;
constexpr double domain_high = 0x1p256;
constexpr double domain_spread = 0x1p64;
constexpr const char* domain_message =
    "input outside the supported numerical domain (nonzero |x| in [2^-256, 2^256], "
    "max/min nonzero |x| <= 2^64)";

// One O(n) pre-pass; inputs are already known to be finite.
bool in_numerical_domain(const double* values, size_t n) noexcept {
    double low = domain_high, high = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double a = std::abs(values[i]);
        if (a == 0.0) continue;
        low = std::min(low, a);
        high = std::max(high, a);
    }
    if (high == 0.0) return true;
    return low >= domain_low && high <= domain_high && high <= low * domain_spread;
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
// Rows of X as a row-major n x k copy, scaled by the exact power of two
// 2^-exponent with max|x| = m 2^exponent, m in [0.5, 1), then centred per
// column (shift to the first row, mean, residual-mean correction), so a large
// common offset does not cost precision. Sums of squares of the copy are
// mapped back with scalbn(ss, 2 exponent), never through scale^2.
// False on nonfinite X; exponent 0 when X is all zero.
bool scaled_centered_rows(const n4m_matrix_view_t& v, std::vector<double>& out, int& exponent) {
    double max_abs = 0.0;
    for (int64_t i = 0; i < v.rows; ++i)
        for (int64_t j = 0; j < v.cols; ++j) {
            const double x = at(v, i, j);
            if (!std::isfinite(x)) return false;
            max_abs = std::max(max_abs, std::abs(x));
        }
    exponent = 0;
    if (max_abs > 0.0) std::frexp(max_abs, &exponent);
    const auto n = static_cast<size_t>(v.rows), k = static_cast<size_t>(v.cols);
    out.resize(n * k);
    for (size_t j = 0; j < k; ++j) {
        const auto jj = static_cast<int64_t>(j);
        const double origin = std::ldexp(at(v, 0, jj), -exponent);
        long double mean = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            out[i * k + j] = std::ldexp(at(v, static_cast<int64_t>(i), jj), -exponent) - origin;
            mean += out[i * k + j];
        }
        mean /= static_cast<long double>(n);
        long double residual = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            out[i * k + j] = static_cast<double>(out[i * k + j] - mean);
            residual += out[i * k + j];
        }
        residual /= static_cast<long double>(n);
        for (size_t i = 0; i < n; ++i)
            out[i * k + j] = static_cast<double>(out[i * k + j] - residual);
    }
    return true;
}

// Partial sum of squares of one categorical term given strata, for rows of
// within-strata centered data Xw. With D the term dummies and S the strata
// dummies, the reduced normal equations M = D'(I - P_S)D have entries
// M_ab = delta_ab n_a - sum_s n_as n_bs / n_s. Dropping one reference level
// per connected component of the level-stratum graph leaves M_r positive
// definite, and SS = sum_j t_j' M_r^-1 t_j with t_j the per-level sums of
// column j of Xw (D'(I - P_S)Xw = D'Xw because Xw has zero stratum sums).
struct PermanovaKernel {
    const std::vector<int64_t>& level;
    const std::vector<double>& xw;
    size_t k;
    std::vector<int64_t> reduced;   // level -> row of M_r, -1 for references
    std::vector<double> chol;       // lower Cholesky factor of M_r, r x r
    size_t r = 0;
    std::vector<double> sums, rhs;

    double operator()(const int64_t* perm) {
        std::fill(sums.begin(), sums.end(), 0.0);
        for (size_t i = 0; i < level.size(); ++i) {
            const size_t src = static_cast<size_t>(perm ? perm[i] : static_cast<int64_t>(i));
            double* row = &sums[static_cast<size_t>(level[i]) * k];
            for (size_t j = 0; j < k; ++j) row[j] += xw[src * k + j];
        }
        long double ss = 0.0L;
        for (size_t j = 0; j < k; ++j) {
            for (size_t a = 0; a < reduced.size(); ++a)
                if (reduced[a] >= 0) rhs[static_cast<size_t>(reduced[a])] = sums[a * k + j];
            for (size_t u = 0; u < r; ++u) {
                long double v = rhs[u];
                for (size_t w = 0; w < u; ++w)
                    v -= static_cast<long double>(chol[u * r + w]) * rhs[w];
                rhs[u] = static_cast<double>(v / chol[u * r + u]);
                ss += static_cast<long double>(rhs[u]) * rhs[u];
            }
        }
        return static_cast<double>(ss);
    }
};
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
        if (!in_numerical_domain(p_values, static_cast<size_t>(m)))
            return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
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
        if (!in_numerical_domain(p_values, static_cast<size_t>(m)))
            return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
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
    double* out_thirteen, int64_t* out_two) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!term_codes || !out_thirteen || !out_two)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null PERMANOVA buffer");
    const int64_t n = X->rows, k = X->cols;
    if (n < 2 || k < 1 || n_permutations < 0 || n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PERMANOVA dimensions");
    try {
        const auto nn = static_cast<size_t>(n), kk = static_cast<size_t>(k);
        std::vector<int64_t> level_n, stratum_n;
        const auto level = dense_codes(std::vector<int64_t>(term_codes, term_codes + n), level_n);
        const auto stratum = dense_codes(
            strata_codes ? std::vector<int64_t>(strata_codes, strata_codes + n)
                         : std::vector<int64_t>(nn, 0),
            stratum_n);
        const size_t a = level_n.size(), s = stratum_n.size();
        if (a > 2048)
            return fail(ctx, N4M_ERR_UNSUPPORTED, "PERMANOVA term exceeds 2048 levels");
        const long double work =
            (static_cast<long double>(n) * static_cast<long double>(k + 1) +
             static_cast<long double>(a) * static_cast<long double>(a) *
                 static_cast<long double>(k)) *
                (static_cast<long double>(n_permutations) + 1.0L) +
            static_cast<long double>(a) * static_cast<long double>(n);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "PERMANOVA permutations exceed 250 million operation budget");
        std::vector<double> x;
        int exponent = 0;
        if (!scaled_centered_rows(*X, x, exponent))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PERMANOVA input");
        // Grand and stratum means, total and between-strata SS, centered Xw.
        std::vector<long double> grand(kk, 0.0L), smean(s * kk, 0.0L);
        for (size_t i = 0; i < nn; ++i)
            for (size_t j = 0; j < kk; ++j) {
                grand[j] += x[i * kk + j];
                smean[static_cast<size_t>(stratum[i]) * kk + j] += x[i * kk + j];
            }
        for (size_t j = 0; j < kk; ++j) grand[j] /= static_cast<long double>(n);
        for (size_t g = 0; g < s; ++g)
            for (size_t j = 0; j < kk; ++j)
                smean[g * kk + j] /= static_cast<long double>(stratum_n[g]);
        long double ss_total = 0.0L, ss_strata = 0.0L, within = 0.0L;
        std::vector<double> xw(nn * kk);
        for (size_t i = 0; i < nn; ++i)
            for (size_t j = 0; j < kk; ++j) {
                const long double d = x[i * kk + j] - grand[j];
                const long double w = x[i * kk + j] - smean[static_cast<size_t>(stratum[i]) * kk + j];
                ss_total += d * d;
                within += w * w;
                xw[i * kk + j] = static_cast<double>(w);
            }
        for (size_t g = 0; g < s; ++g)
            for (size_t j = 0; j < kk; ++j) {
                const long double d = smean[g * kk + j] - grand[j];
                ss_strata += static_cast<long double>(stratum_n[g]) * d * d;
            }
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
        PermanovaKernel kernel{level, xw, kk, std::vector<int64_t>(a, -1), {}, 0, {}, {}};
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
            size_t begin = 0;
            while (begin < cells.size()) {
                size_t end = begin;
                const int64_t g = cells[begin].first.first;
                while (end < cells.size() && cells[end].first.first == g) ++end;
                const auto ns = static_cast<long double>(stratum_n[static_cast<size_t>(g)]);
                log10_distinct += std::lgamma(static_cast<double>(stratum_n[static_cast<size_t>(g)]) + 1.0);
                for (size_t u = begin; u < end; ++u) {
                    log10_distinct -= std::lgamma(static_cast<double>(cells[u].second) + 1.0);
                    const int64_t ru = kernel.reduced[static_cast<size_t>(cells[u].first.second)];
                    if (ru < 0) continue;
                    for (size_t v = begin; v < end; ++v) {
                        const int64_t rv = kernel.reduced[static_cast<size_t>(cells[v].first.second)];
                        if (rv < 0) continue;
                        m[static_cast<size_t>(ru) * r + static_cast<size_t>(rv)] -=
                            static_cast<long double>(cells[u].second) *
                            static_cast<long double>(cells[v].second) / ns;
                    }
                }
                begin = end;
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
                    d -= static_cast<long double>(kernel.chol[j * r + l]) * kernel.chol[j * r + l];
                if (!(d > 1e-12L * m[j * r + j]))
                    return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                                "PERMANOVA design is numerically singular");
                const long double pivot = std::sqrt(d);
                kernel.chol[j * r + j] = static_cast<double>(pivot);
                for (size_t i = j + 1; i < r; ++i) {
                    long double v = m[i * r + j];
                    for (size_t l = 0; l < j; ++l)
                        v -= static_cast<long double>(kernel.chol[i * r + l]) * kernel.chol[j * r + l];
                    kernel.chol[i * r + j] = static_cast<double>(v / pivot);
                }
            }
        }
        kernel.sums.assign(a * kk, 0.0);
        kernel.rhs.assign(r, 0.0);
        const double ss_term = r > 0 ? kernel(nullptr) : 0.0;
        const double w = static_cast<double>(within);
        const double ss_res = std::max(0.0, w - ss_term);
        for (int i = 0; i < 13; ++i) out_thirteen[i] = nan_value;
        out_thirteen[0] = std::scalbn(static_cast<double>(ss_total), 2 * exponent);
        out_thirteen[1] = std::scalbn(static_cast<double>(ss_strata), 2 * exponent);
        out_thirteen[2] = std::scalbn(ss_term, 2 * exponent);
        out_thirteen[3] = std::scalbn(ss_res, 2 * exponent);
        for (int i = 0; i < 4; ++i)
            if (!std::isfinite(out_thirteen[i]))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "PERMANOVA sums of squares are not representable");
        const bool defined = df_term > 0 && w > 0.0;
        const double partial = defined ? std::min(1.0, ss_term / w) : nan_value;
        if (df_term > 0 && ss_total > 0.0L)
            out_thirteen[4] = std::min(1.0, ss_term / static_cast<double>(ss_total));
        out_thirteen[5] = partial;
        if (defined && df_res > 0 && ss_res > 0.0)
            out_thirteen[6] = (ss_term / static_cast<double>(df_term)) /
                              (ss_res / static_cast<double>(df_res));
        out_thirteen[7] = static_cast<double>(df_term);
        out_thirteen[8] = static_cast<double>(df_res);
        out_thirteen[9] = static_cast<double>(a);
        out_thirteen[10] = static_cast<double>(s);
        out_thirteen[11] = log10_distinct;
        out_two[0] = 0;
        out_two[1] = 0;
        // As vegan anova.cca / permutest: count pseudo-F_perm >= F_obs - sqrt(eps).
        const double f_observed = out_thirteen[6];
        if (n_permutations == 0 || !std::isfinite(f_observed)) return N4M_OK;
        const long double f_scale = static_cast<long double>(df_res) / static_cast<long double>(df_term);
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
            const long double ss = kernel(perm.data());
            const long double rest = static_cast<long double>(w) - ss;
            const double f = rest > 0.0L ? static_cast<double>(ss / rest * f_scale)
                                         : std::numeric_limits<double>::infinity();
            if (f >= f_observed - tolerance) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_thirteen[12] = static_cast<double>(1 + at_least) /
                           static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PERMANOVA allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PERMANOVA internal error"); }
}

N4M_API n4m_status_t n4m_stats_permdisp(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int32_t n_permutations, uint64_t seed, double* distances,
    double* group_mean_distances, double* out_six, int64_t* out_two) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_view(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_codes || !distances || !group_mean_distances || !out_six || !out_two)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null PERMDISP buffer");
    const int64_t n = X->rows, k = X->cols;
    if (n < 2 || k < 1 || n_permutations < 0 || n_permutations > 1000000)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid PERMDISP dimensions");
    const long double work =
        static_cast<long double>(n) * (2.0L * static_cast<long double>(k) +
                                       static_cast<long double>(n_permutations) + 1.0L);
    if (work > max_moment_products)
        return fail(ctx, N4M_ERR_UNSUPPORTED,
                    "PERMDISP permutations exceed 250 million operation budget");
    try {
        const auto nn = static_cast<size_t>(n), kk = static_cast<size_t>(k);
        std::vector<int64_t> group_n;
        const auto group = dense_codes(std::vector<int64_t>(group_codes, group_codes + n), group_n);
        const size_t g = group_n.size();
        std::vector<double> x;
        int exponent = 0;
        if (!scaled_centered_rows(*X, x, exponent))
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite PERMDISP input");
        std::vector<long double> centroid(g * kk, 0.0L);
        for (size_t i = 0; i < nn; ++i)
            for (size_t j = 0; j < kk; ++j)
                centroid[static_cast<size_t>(group[i]) * kk + j] += x[i * kk + j];
        for (size_t c = 0; c < g; ++c)
            for (size_t j = 0; j < kk; ++j)
                centroid[c * kk + j] /= static_cast<long double>(group_n[c]);
        std::vector<double> z(nn);
        std::vector<long double> zmean(g, 0.0L);
        long double grand = 0.0L;
        for (size_t i = 0; i < nn; ++i) {
            long double d2 = 0.0L;
            for (size_t j = 0; j < kk; ++j) {
                const long double d = x[i * kk + j] - centroid[static_cast<size_t>(group[i]) * kk + j];
                d2 += d * d;
            }
            z[i] = static_cast<double>(std::sqrt(d2));
            zmean[static_cast<size_t>(group[i])] += z[i];
            grand += z[i];
        }
        grand /= static_cast<long double>(n);
        for (size_t c = 0; c < g; ++c) zmean[c] /= static_cast<long double>(group_n[c]);
        long double between = 0.0L, residual = 0.0L, resid_sum = 0.0L;
        std::vector<double> resid(nn);
        for (size_t c = 0; c < g; ++c)
            between += static_cast<long double>(group_n[c]) * (zmean[c] - grand) * (zmean[c] - grand);
        for (size_t i = 0; i < nn; ++i) {
            const long double e = z[i] - zmean[static_cast<size_t>(group[i])];
            resid[i] = static_cast<double>(e);
            residual += e * e;
            resid_sum += e;
            distances[i] = std::scalbn(z[i], exponent);
        }
        for (size_t c = 0; c < g; ++c)
            group_mean_distances[c] = std::scalbn(static_cast<double>(zmean[c]), exponent);
        for (size_t i = 0; i < nn; ++i)
            if (!std::isfinite(distances[i]))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "PERMDISP distances are not representable");
        for (int i = 0; i < 6; ++i) out_six[i] = nan_value;
        out_six[1] = static_cast<double>(g) - 1.0;
        out_six[2] = static_cast<double>(n) - static_cast<double>(g);
        out_six[3] = static_cast<double>(g);
        out_six[4] = log10_factorial_ratio(n, group_n);
        out_two[0] = 0;
        out_two[1] = 0;
        const bool defined = g >= 2 && nn > g && residual > 0.0L;
        if (!defined) return N4M_OK;
        out_six[0] = static_cast<double>((between / static_cast<long double>(g - 1)) /
                                         (residual / static_cast<long double>(nn - g)));
        if (n_permutations == 0) return N4M_OK;
        // vegan permutest.betadisper: permute the ANOVA residuals of the
        // distances, refit the one-way model and count
        // F_perm = (B_perm/(g-1)) / ((total - B_perm)/(n-g)) >= F_obs - sqrt(eps).
        const long double mean_r = resid_sum / static_cast<long double>(n);
        long double total = 0.0L;
        for (size_t i = 0; i < nn; ++i) total += (resid[i] - mean_r) * (resid[i] - mean_r);
        const double f_observed = out_six[0];
        const long double f_scale =
            static_cast<long double>(nn - g) / static_cast<long double>(g - 1);
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(nn);
        std::vector<long double> sums(g);
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            std::fill(sums.begin(), sums.end(), 0.0L);
            for (size_t i = 0; i < nn; ++i)
                sums[static_cast<size_t>(group[i])] += resid[static_cast<size_t>(perm[i])];
            long double bp = -resid_sum * resid_sum / static_cast<long double>(n);
            for (size_t c = 0; c < g; ++c)
                bp += sums[c] * sums[c] / static_cast<long double>(group_n[c]);
            const long double rest = total - bp;
            const double f = rest > 0.0L ? static_cast<double>(bp / rest * f_scale)
                                         : std::numeric_limits<double>::infinity();
            if (f >= f_observed - tolerance) ++at_least;
        }
        out_two[0] = at_least;
        out_two[1] = n_permutations;
        out_six[5] = static_cast<double>(1 + at_least) / static_cast<double>(1 + n_permutations);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "PERMDISP allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "PERMDISP internal error"); }
}
}  // extern "C"
