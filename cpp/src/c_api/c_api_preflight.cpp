// SPDX-License-Identifier: CECILL-2.1
// Reusable, caller-buffer diagnostics for Preflight and other C ABI clients.
#include "n4m/metrics.h"

#include <algorithm>
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

// Permutation statistic over targets[perm[i]] assigned to unit i:
// eta^2 = sum_g S_g^2/n_g / SST on centered targets, or Cramer's V^2.
struct AssociationKernel {
    const std::vector<int64_t>& level;
    const std::vector<int64_t>& level_n;
    const std::vector<double>& centered;
    const std::vector<int64_t>& cls;
    const std::vector<int64_t>& class_n;
    bool categorical;
    double sst;
    std::vector<double> sums;
    std::vector<int64_t> table;

    double operator()(const int64_t* perm) {
        const size_t n = level.size(), g = level_n.size();
        if (!categorical) {
            std::fill(sums.begin(), sums.end(), 0.0);
            for (size_t i = 0; i < n; ++i)
                sums[static_cast<size_t>(level[i])] +=
                    centered[static_cast<size_t>(perm ? perm[i] : static_cast<int64_t>(i))];
            double ssb = 0.0;
            for (size_t a = 0; a < g; ++a)
                ssb += sums[a] * sums[a] / static_cast<double>(level_n[a]);
            return std::min(1.0, ssb / sst);
        }
        const size_t c = class_n.size();
        std::fill(table.begin(), table.end(), 0);
        for (size_t i = 0; i < n; ++i)
            ++table[static_cast<size_t>(level[i]) * c +
                    static_cast<size_t>(cls[static_cast<size_t>(
                        perm ? perm[i] : static_cast<int64_t>(i))])];
        double sum = 0.0;
        for (size_t a = 0; a < g; ++a) for (size_t b = 0; b < c; ++b) {
            const double cell = static_cast<double>(table[a * c + b]);
            if (cell > 0.0)
                sum += cell * cell /
                       (static_cast<double>(level_n[a]) * static_cast<double>(class_n[b]));
        }
        return std::max(0.0, (sum - 1.0) / static_cast<double>(std::min(g, c) - 1));
    }
};

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

// Values divided by max|v| (sums stay representable; r is scale invariant),
// then centered. Returns the centered sum of squares.
double scaled_centered(std::vector<double>& values) noexcept {
    double scale = 0.0;
    for (double v : values) scale = std::max(scale, std::abs(v));
    if (!(scale > 0.0)) return 0.0;
    double mean = 0.0;
    for (double& v : values) {
        v /= scale;
        mean += v;
    }
    mean /= static_cast<double>(values.size());
    double ss = 0.0;
    for (double& v : values) {
        v -= mean;
        ss += v * v;
    }
    return ss;
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
    if (!factor_codes || !targets || !out_eight || !out_two)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null association buffer");
    if (n < 2 || (categorical != 0 && categorical != 1) || n_permutations < 0 ||
        n_permutations > 1000000 ||
        static_cast<uint64_t>(n) > static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid association dimensions");
    try {
        constexpr double max_exact_category = 9007199254740991.0;
        double scale = 0.0;
        std::vector<int64_t> category(categorical ? static_cast<size_t>(n) : 0);
        for (int64_t i = 0; i < n; ++i) {
            const double y = targets[i];
            if (!std::isfinite(y))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite target");
            if (categorical) {
                if (std::abs(y) > max_exact_category || std::trunc(y) != y)
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                                "category codes must be exact integers");
                category[static_cast<size_t>(i)] = static_cast<int64_t>(y);
            }
            scale = std::max(scale, std::abs(y));
        }
        std::vector<int64_t> level_n, class_n, value_n;
        const auto level = dense_codes(std::vector<int64_t>(factor_codes, factor_codes + n),
                                       level_n);
        std::vector<int64_t> cls;
        if (categorical) cls = dense_codes(category, class_n);
        // +0.0 replaces -0.0 so both zero encodings share one value.
        std::vector<double> exact(targets, targets + n);
        for (double& y : exact) y += 0.0;
        dense_codes(exact, value_n);
        const auto g = static_cast<int64_t>(level_n.size());
        const auto v = static_cast<int64_t>(value_n.size());
        const auto c = static_cast<int64_t>(class_n.size());
        const long double work =
            (static_cast<long double>(n) + static_cast<long double>(g) * c) *
            (static_cast<long double>(n_permutations) + 1.0L);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "association permutations exceed 250 million operation budget");
        for (int i = 0; i < 8; ++i) out_eight[i] = nan_value;
        out_two[0] = 0;
        out_two[1] = 0;
        out_eight[4] = static_cast<double>(g);
        out_eight[5] = static_cast<double>(v);
        out_eight[6] = std::min(log10_factorial_ratio(n, level_n),
                                log10_factorial_ratio(n, value_n));
        if (g < 2 || v < 2) return N4M_OK;
        // Scaling by max|y| keeps every sum representable; eta^2 and F are
        // scale invariant.
        std::vector<double> centered(static_cast<size_t>(n));
        double mean = 0.0;
        for (int64_t i = 0; i < n; ++i) mean += targets[i] / scale;
        mean /= static_cast<double>(n);
        double sst = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            centered[static_cast<size_t>(i)] = targets[i] / scale - mean;
            sst += centered[static_cast<size_t>(i)] * centered[static_cast<size_t>(i)];
        }
        if (!categorical && !(sst > 0.0)) return N4M_OK;
        AssociationKernel kernel{level, level_n, centered, cls, class_n,
                                 categorical != 0, sst,
                                 std::vector<double>(static_cast<size_t>(g), 0.0),
                                 std::vector<int64_t>(categorical
                                     ? static_cast<size_t>(g) * static_cast<size_t>(c) : 0)};
        const double observed = kernel(nullptr);
        if (!categorical) {
            out_eight[0] = observed;
            std::vector<double> level_mean(static_cast<size_t>(g), 0.0);
            for (int64_t i = 0; i < n; ++i)
                level_mean[static_cast<size_t>(level[static_cast<size_t>(i)])] +=
                    centered[static_cast<size_t>(i)];
            for (int64_t a = 0; a < g; ++a)
                level_mean[static_cast<size_t>(a)] /=
                    static_cast<double>(level_n[static_cast<size_t>(a)]);
            double ssw = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = centered[static_cast<size_t>(i)] -
                                 level_mean[static_cast<size_t>(level[static_cast<size_t>(i)])];
                ssw += d * d;
            }
            const double ssb = observed * sst;
            if (n > g && ssw > 0.0)
                out_eight[1] = (ssb / static_cast<double>(g - 1)) /
                               (ssw / static_cast<double>(n - g));
            out_eight[2] = static_cast<double>(g - 1);
            out_eight[3] = static_cast<double>(n - g);
        } else {
            out_eight[0] = std::sqrt(observed);
            out_eight[1] = observed * static_cast<double>(n) *
                           static_cast<double>(std::min(g, c) - 1);
            out_eight[2] = static_cast<double>((g - 1) * (c - 1));
        }
        if (n_permutations == 0) return N4M_OK;
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(static_cast<size_t>(n));
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            if (kernel(perm.data()) >= observed - tolerance) ++at_least;
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
    if (!x || !y || !out_five || !out_two)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null correlation buffer");
    if (n < 2 || (method != 0 && method != 1) || n_permutations < 0 ||
        n_permutations > 1000000 ||
        static_cast<uint64_t>(n) > static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double)))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid correlation dimensions or method");
    try {
        for (int64_t i = 0; i < n; ++i)
            if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite correlation input");
        const long double work = static_cast<long double>(n) *
                                 (static_cast<long double>(n_permutations) + 1.0L);
        if (work > max_moment_products)
            return fail(ctx, N4M_ERR_UNSUPPORTED,
                        "correlation permutations exceed 250 million operation budget");
        // +0.0 replaces -0.0 so both zero encodings share one value.
        std::vector<double> xs(x, x + n), ys(y, y + n);
        for (double& v : xs) v += 0.0;
        for (double& v : ys) v += 0.0;
        std::vector<int64_t> x_n, y_n;
        dense_codes(xs, x_n);
        dense_codes(ys, y_n);
        for (int i = 0; i < 5; ++i) out_five[i] = nan_value;
        out_two[0] = 0;
        out_two[1] = 0;
        out_five[1] = static_cast<double>(x_n.size());
        out_five[2] = static_cast<double>(y_n.size());
        out_five[3] = std::min(log10_factorial_ratio(n, x_n), log10_factorial_ratio(n, y_n));
        if (x_n.size() < 2 || y_n.size() < 2) return N4M_OK;
        if (method == 1) {
            xs = average_ranks(xs);
            ys = average_ranks(ys);
        }
        const double sxx = scaled_centered(xs), syy = scaled_centered(ys);
        if (!(sxx > 0.0) || !(syy > 0.0)) return N4M_OK;
        const double denominator = std::sqrt(sxx) * std::sqrt(syy);
        const auto size = static_cast<size_t>(n);
        const auto correlation = [&](const int64_t* perm) {
            double sxy = 0.0;
            for (size_t i = 0; i < size; ++i)
                sxy += xs[i] * ys[perm ? static_cast<size_t>(perm[i]) : i];
            return std::max(-1.0, std::min(1.0, sxy / denominator));
        };
        const double observed = correlation(nullptr);
        out_five[0] = observed;
        if (n_permutations == 0) return N4M_OK;
        n4m_rng_pcg64 rng{};
        n4m_pcg64_engine_seed(&rng, seed);
        std::vector<int64_t> perm(size);
        const double threshold =
            std::abs(observed) - std::sqrt(std::numeric_limits<double>::epsilon());
        int64_t at_least = 0;
        for (int32_t b = 0; b < n_permutations; ++b) {
            n4m_pcg64_engine_permutation(&rng, perm.data(), n);
            if (std::abs(correlation(perm.data())) >= threshold) ++at_least;
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
}  // extern "C"
