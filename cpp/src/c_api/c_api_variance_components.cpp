// SPDX-License-Identifier: CECILL-2.1
// One-way random-effects variance components: REML estimates and exact or
// cluster-bootstrap confidence intervals, per feature, into caller buffers.
#include "n4m/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <vector>

#include "core/augmentation/aug_rng_utils.h"
#include "core/common/centring.hpp"
#include "core/common/context.hpp"
#include "core/common/distributions.hpp"
#include "core/common/matrix_view.hpp"
#include "core/common/rng_pcg64.h"
#include "core/variance_components.hpp"

namespace {
using ::n4m::core::Context;
namespace vc = ::n4m::core::variance_components;
namespace dist = ::n4m::core::distributions;
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr long double max_bootstrap_work = 2500000000.0L;
// Per replicate and feature: the group scan plus the grid and bisections.
constexpr long double bootstrap_fit_work = 1024.0L;
constexpr int32_t max_bootstrap = 100000;
// Supported numerical domain (lead rule 13): every nonzero |x| in
// [2^-256, 2^256], and max/min nonzero |x| <= 2^64 over the raw values and
// over the centred values of each feature. Outside it the call is refused.
constexpr int domain_exponent = 256;
constexpr int domain_spread = 64;
constexpr const char* domain_message =
    "input outside the supported numerical domain (nonzero |x| in [2^-256, 2^256], "
    "max/min nonzero |x| <= 2^64)";


size_t sz(int64_t v) noexcept { return static_cast<size_t>(v); }

n4m_status_t fail(n4m_context_t* ctx, n4m_status_t code, const char* message) noexcept {
    static_cast<Context*>(ctx)->set_error(message);
    return code;
}

n4m_status_t check_input(n4m_context_t* ctx, const n4m_matrix_view_t* X) noexcept {
    if (!X) return fail(ctx, N4M_ERR_NULL_POINTER, "null matrix view");
    const auto st = ::n4m::core::validate_nonnull_view(*X);
    if (st != N4M_OK) return fail(ctx, st, "invalid matrix view");
    if (X->dtype != N4M_DTYPE_F64) return fail(ctx, N4M_ERR_DTYPE_MISMATCH, "matrix must be f64");
    if (X->rows < 1 || X->cols < 1)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "empty variance-components matrix");
    const auto max_index = static_cast<uint64_t>(PTRDIFF_MAX / sizeof(double));
    const auto row = static_cast<uint64_t>(X->rows - 1), col = static_cast<uint64_t>(X->cols - 1);
    const auto rs = static_cast<uint64_t>(X->row_stride), cs = static_cast<uint64_t>(X->col_stride);
    if ((rs && row > max_index / rs) || (cs && col > max_index / cs) ||
        row * rs > max_index - col * cs ||
        static_cast<uint64_t>(X->rows) > SIZE_MAX / static_cast<uint64_t>(X->cols))
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "matrix view offset overflow");
    return N4M_OK;
}

// Dense group index per row, groups ordered by ascending ID.
std::vector<int64_t> dense(const int64_t* ids, int64_t n, int64_t& levels) {
    std::vector<int64_t> sorted(ids, ids + n);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    levels = static_cast<int64_t>(sorted.size());
    std::vector<int64_t> out(sz(n));
    for (int64_t i = 0; i < n; ++i)
        out[sz(i)] = std::lower_bound(sorted.begin(), sorted.end(), ids[i]) - sorted.begin();
    return out;
}

// One feature on an exact power-of-two scale chosen from its centred
// variation: values are prescaled by 2^-e1 (raw magnitude), centred without
// cancellation, then divided by 2^e2 bounding the centred values, so |c| < 1
// and x - mean(x) = c 2^exponent. Within-group residuals are centred per
// group from the ORIGINAL values (same prescale, then 2^-e2), so a within
// spread far below the between spread keeps full relative precision.
struct Centred {
    std::vector<double> c, residual;
    std::vector<int64_t> owner;
    int exponent = 0;
};

Centred centre_column(const n4m_matrix_view_t& X, int64_t j, const std::vector<int64_t>& group,
                      int64_t n_groups, const uint8_t* mask) {
    const auto* data = static_cast<const double*>(X.data);
    Centred out;
    for (int64_t i = 0; i < X.rows; ++i) {
        if (mask && mask[sz(i) * sz(X.cols) + sz(j)] == 0) continue;
        out.c.push_back(data[i * X.row_stride + j * X.col_stride]);
        out.owner.push_back(group[sz(i)]);
    }
    const int e1 = ::n4m::core::bounding_exponent(out.c);
    for (double& v : out.c) v = std::ldexp(v, -e1);
    std::vector<std::vector<size_t>> members(sz(n_groups));
    for (size_t i = 0; i < out.c.size(); ++i) members[sz(out.owner[i])].push_back(i);
    out.residual.assign(out.c.size(), 0.0);
    std::vector<double> block;
    for (const auto& rows : members) {
        block.clear();
        for (size_t i : rows) block.push_back(out.c[i]);
        ::n4m::core::center_in_place(block);
        for (size_t k = 0; k < rows.size(); ++k) out.residual[rows[k]] = block[k];
    }
    ::n4m::core::center_in_place(out.c);
    const int e2 = ::n4m::core::bounding_exponent(out.c);
    for (double& v : out.c) v = std::ldexp(v, -e2);
    for (double& v : out.residual) v = std::ldexp(v, -e2);
    out.exponent = e1 + e2;
    return out;
}

// Groups observed in one feature: means from the centred values, within
// sums of squares m2 = sum (res 2^-ew)^2 from the per-group residuals with
// their own exponent ew. Sorted by (r, group).
struct Column {
    std::vector<vc::Group> groups;
    std::vector<int64_t> index;  // dense group index of each entry
    int exponent = 0;
    int within_exponent = 0;
};

void scan_column(const n4m_matrix_view_t& X, int64_t j, const std::vector<int64_t>& group,
                 int64_t n_groups, const uint8_t* mask, Column& out) {
    const auto cen = centre_column(X, j, group, n_groups, mask);
    out.exponent = cen.exponent;
    std::vector<vc::Group> all(sz(n_groups));
    for (size_t i = 0; i < cen.c.size(); ++i) {
        auto& g = all[sz(cen.owner[i])];
        ++g.r;
        g.mean += static_cast<long double>(cen.c[i]);
    }
    for (auto& g : all)
        if (g.r > 0) g.mean /= static_cast<long double>(g.r);
    out.within_exponent = ::n4m::core::bounding_exponent(cen.residual);
    for (size_t i = 0; i < cen.c.size(); ++i) {
        const auto r = static_cast<long double>(std::ldexp(cen.residual[i], -out.within_exponent));
        all[sz(cen.owner[i])].m2 += r * r;
    }
    out.index.clear();
    for (int64_t k = 0; k < n_groups; ++k)
        if (all[sz(k)].r > 0) out.index.push_back(k);
    std::stable_sort(out.index.begin(), out.index.end(),
                     [&all](int64_t a, int64_t b) { return all[sz(a)].r < all[sz(b)].r; });
    out.groups.clear();
    for (int64_t k : out.index) out.groups.push_back(all[sz(k)]);
}

// Spread max/min of the nonzero magnitudes within 2^64; zeros are allowed.
bool spread_ok(const std::vector<double>& a, const std::vector<double>& b) noexcept {
    double lo = 0.0, hi = 0.0;
    for (const auto* values : {&a, &b})
        for (double v : *values) {
            const double m = std::abs(v);
            if (m == 0.0) continue;
            lo = lo == 0.0 ? m : std::min(lo, m);
            hi = std::max(hi, m);
        }
    return hi == 0.0 || hi <= std::ldexp(lo, domain_spread);
}

// Rule 13 pre-pass over every feature, before any computation: unmasked
// values must be finite, every nonzero |x| in [2^-256, 2^256], and the
// spread max/min of nonzero magnitudes <= 2^64 over the raw values and over
// the differences the statistic uses (column-centred values together with
// the per-group residuals).
n4m_status_t check_domain(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                          const std::vector<int64_t>& group, int64_t n_groups,
                          const uint8_t* mask) {
    if (X.rows > (int64_t{1} << 31))
        return fail(ctx, N4M_ERR_UNSUPPORTED, "more than 2^31 rows (quantile domain)");
    const auto* data = static_cast<const double*>(X.data);
    const double lowest = std::ldexp(1.0, -domain_exponent);
    const double highest = std::ldexp(1.0, domain_exponent);
    const std::vector<double> none;
    std::vector<double> values;
    for (int64_t j = 0; j < X.cols; ++j) {
        values.clear();
        for (int64_t i = 0; i < X.rows; ++i) {
            if (mask && mask[sz(i) * sz(X.cols) + sz(j)] == 0) continue;
            const double x = data[i * X.row_stride + j * X.col_stride];
            if (!std::isfinite(x))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked input");
            const double a = std::abs(x);
            if (a != 0.0 && (a < lowest || a > highest))
                return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
            values.push_back(x);
        }
        if (!spread_ok(values, none)) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
        const auto cen = centre_column(X, j, group, n_groups, mask);
        if (!spread_ok(cen.c, cen.residual)) return fail(ctx, N4M_ERR_UNSUPPORTED, domain_message);
    }
    return N4M_OK;
}

// A non-negative value v 2^shift as a double: false when it overflows, or
// when a positive v underflows to zero (not reachable inside the domain).
bool unscale(long double v, int shift, double& out) noexcept {
    out = static_cast<double>(std::scalbn(v, shift));
    return std::isfinite(out) && !(v > 0 && !(out > 0.0));
}

double type7(std::vector<double>& v, double q) {
    std::sort(v.begin(), v.end());
    const double h = static_cast<double>(v.size() - 1) * q;
    const auto lo = static_cast<size_t>(h);
    const auto hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

bool usable(int32_t status) noexcept {
    return status == vc::interior || status == vc::between_boundary ||
           status == vc::within_boundary;
}
}  // namespace

extern "C" {
N4M_API n4m_status_t n4m_stats_variance_components_reml(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    const uint8_t* valid_mask, double* out, int32_t* band_status) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_input(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_ids || !out || !band_status)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null variance-components buffer");
    try {
        int64_t n_groups = 0;
        const auto group = dense(group_ids, X->rows, n_groups);
        const auto domain = check_domain(ctx, *X, group, n_groups, valid_mask);
        if (domain != N4M_OK) return domain;
        Column column;
        constexpr long double ln2 = 0.693147180559945309417232121458176568L;
        for (int64_t j = 0; j < X->cols; ++j) {
            scan_column(*X, j, group, n_groups, valid_mask, column);
            const auto fit = vc::fit_reml(column.groups, nullptr, column.within_exponent);
            double* row = out + 8 * j;
            std::fill(row, row + 8, nan_value);
            row[4] = static_cast<double>(fit.n_groups);
            row[5] = static_cast<double>(fit.n_obs);
            row[6] = static_cast<double>(fit.candidates);
            row[7] = static_cast<double>(fit.iterations);
            band_status[j] = fit.status;
            if (!usable(fit.status)) continue;
            const int e = 2 * column.exponent;
            if (!unscale(fit.sigma2_between, e, row[0]) ||
                !unscale(fit.sigma2_within, e + fit.within_shift, row[1]))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "variance component is not representable as f64");
            row[2] = fit.icc;
            row[3] = static_cast<double>(
                fit.loglik - static_cast<long double>(fit.n_obs - 1) * column.exponent * ln2);
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "variance-components allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "variance-components internal error"); }
}

N4M_API n4m_status_t n4m_stats_variance_components_intervals(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    const int64_t* cluster_ids, const uint8_t* valid_mask, double confidence,
    int32_t n_bootstrap, uint64_t seed, double* out, int64_t* n_valid_replicates) {
    if (!ctx) return N4M_ERR_NULL_POINTER;
    const auto st = check_input(ctx, X);
    if (st != N4M_OK) return st;
    if (!group_ids || !out || !n_valid_replicates)
        return fail(ctx, N4M_ERR_NULL_POINTER, "null variance-components buffer");
    // Any double confidence < 1 leaves half tails >= 2^-54, inside the
    // quantile domain [2^-64, 1/2].
    if (!(confidence > 0.0 && confidence < 1.0) || n_bootstrap < 0 ||
        n_bootstrap > max_bootstrap)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid confidence or bootstrap count");
    try {
        int64_t n_groups = 0, n_units = 0;
        const auto group = dense(group_ids, X->rows, n_groups);
        const auto domain = check_domain(ctx, *X, group, n_groups, valid_mask);
        if (domain != N4M_OK) return domain;
        std::vector<int64_t> unit_of(sz(n_groups));
        if (cluster_ids) {
            const auto cluster = dense(cluster_ids, X->rows, n_units);
            std::vector<int64_t> seen(sz(n_groups), -1);
            for (int64_t i = 0; i < X->rows; ++i) {
                auto& u = seen[sz(group[sz(i)])];
                if (u >= 0 && u != cluster[sz(i)])
                    return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "a group spans two clusters");
                u = cluster[sz(i)];
            }
            unit_of = seen;
        } else {
            n_units = n_groups;
            std::iota(unit_of.begin(), unit_of.end(), int64_t{0});
        }
        const long double work = static_cast<long double>(n_bootstrap) *
                                 static_cast<long double>(X->cols) *
                                 (static_cast<long double>(n_groups) + bootstrap_fit_work);
        if (work > max_bootstrap_work)
            return fail(ctx, N4M_ERR_UNSUPPORTED, "bootstrap exceeds 2.5e9 operation budget");
        const double alpha = 1.0 - confidence;
        std::map<double, std::pair<double, double>> chi;
        std::map<std::pair<double, double>, double> f_upper;
        auto chi_quantiles = [&](double df) {
            auto it = chi.find(df);
            if (it == chi.end())
                it = chi.emplace(df, std::make_pair(dist::chi2_quantile(alpha / 2, df, false),
                                                    dist::chi2_quantile(alpha / 2, df, true)))
                         .first;
            return it->second;
        };
        auto f_quantile = [&](double d1, double d2) {
            const auto key = std::make_pair(d1, d2);
            auto it = f_upper.find(key);
            if (it == f_upper.end())
                it = f_upper.emplace(key, dist::f_quantile(alpha / 2, d1, d2, true)).first;
            return it->second;
        };
        Column column;
        std::vector<int64_t> counts(sz(n_units)), multiplicity;
        std::vector<double> sd_draws, icc_draws;
        for (int64_t j = 0; j < X->cols; ++j) {
            scan_column(*X, j, group, n_groups, valid_mask, column);
            double* row = out + 8 * j;
            std::fill(row, row + 8, nan_value);
            n_valid_replicates[j] = 0;
            bool representable = true;
            auto publish = [&](long double v, int shift, double& slot) {
                if (!unscale(v, shift, slot)) {
                    slot = nan_value;
                    representable = false;
                }
            };
            const auto& gs = column.groups;
            const auto g = static_cast<int64_t>(gs.size());
            const int ew = column.within_exponent, e = column.exponent;
            int64_t n = 0;
            long double ssw_q = 0;  // units 2^(2 ew)
            for (const auto& grp : gs) {
                n += grp.r;
                ssw_q += grp.m2;
            }
            if (n - g >= 1) {
                const auto df = static_cast<double>(n - g);
                const auto q = chi_quantiles(df);
                if (!std::isfinite(q.first) || !std::isfinite(q.second)) {
                    representable = false;
                } else {
                    publish(std::sqrt(ssw_q / static_cast<long double>(q.second)), ew + e, row[0]);
                    publish(std::sqrt(ssw_q / static_cast<long double>(q.first)), ew + e, row[1]);
                }
                const int64_t r = gs.front().r;
                if (g >= 2 && r >= 2 && gs.back().r == r) {
                    long double mean = 0, ssb = 0;
                    for (const auto& grp : gs) mean += grp.mean / static_cast<long double>(g);
                    for (const auto& grp : gs) ssb += (grp.mean - mean) * (grp.mean - mean);
                    const auto rr = static_cast<long double>(r);
                    const long double msb = rr * ssb / static_cast<long double>(g - 1);
                    const long double msw_q = ssw_q / static_cast<long double>(n - g);
                    const auto d1 = static_cast<double>(g - 1);
                    if (msw_q <= 0) {
                        if (msb > 0) row[2] = row[3] = 1.0;
                    } else {
                        // log2 F, then F only when it cannot overflow; beyond
                        // 2^200 both bounds round to 1.
                        const long double log2_f = msb > 0
                            ? std::log2(msb) - std::log2(msw_q) - 2.0L * ew
                            : -std::numeric_limits<long double>::infinity();
                        const double fl = f_quantile(d1, df), fu = f_quantile(df, d1);
                        if (!std::isfinite(fl) || !std::isfinite(fu)) {
                            representable = false;
                        } else if (log2_f > 200) {
                            row[2] = row[3] = 1.0;
                        } else {
                            const long double F = std::scalbn(msb / msw_q, -2 * ew);
                            const long double lower = F / static_cast<long double>(fl);
                            const long double upper = F * static_cast<long double>(fu);
                            row[2] = static_cast<double>((lower - 1) / (lower + rr - 1));
                            row[3] = static_cast<double>((upper - 1) / (upper + rr - 1));
                        }
                    }
                }
            }
            if (n_bootstrap > 0) {
                n4m_rng_pcg64 rng{};
                n4m_pcg64_engine_seed(&rng, seed);
                multiplicity.assign(gs.size(), 0);
                sd_draws.clear();
                icc_draws.clear();
                for (int32_t b = 0; b < n_bootstrap; ++b) {
                    std::fill(counts.begin(), counts.end(), 0);
                    for (int64_t u = 0; u < n_units; ++u)
                        ++counts[sz(static_cast<int64_t>(
                            n4m_aug_rng_bounded(&rng, static_cast<uint64_t>(n_units - 1))))];
                    long double rssw = 0, rn = 0, rg = 0;
                    for (size_t k = 0; k < gs.size(); ++k) {
                        const int64_t m = counts[sz(unit_of[sz(column.index[k])])];
                        multiplicity[k] = m;
                        rssw += static_cast<long double>(m) * gs[k].m2;
                        rn += static_cast<long double>(m * gs[k].r);
                        rg += static_cast<long double>(m);
                    }
                    const auto fit = vc::fit_reml(gs, multiplicity.data(), ew);
                    if (!usable(fit.status)) continue;
                    // Pooled SD in units of 2^ew, mapped back after the percentiles.
                    sd_draws.push_back(static_cast<double>(std::sqrt(rssw / (rn - rg))));
                    icc_draws.push_back(fit.icc);
                }
                n_valid_replicates[j] = static_cast<int64_t>(sd_draws.size());
                if (!sd_draws.empty()) {
                    publish(static_cast<long double>(type7(sd_draws, alpha / 2)), ew + e, row[4]);
                    publish(static_cast<long double>(type7(sd_draws, 1 - alpha / 2)), ew + e, row[5]);
                    row[6] = type7(icc_draws, alpha / 2);
                    row[7] = type7(icc_draws, 1 - alpha / 2);
                }
            }
            if (!representable)
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "interval bound or quantile not representable as f64");
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "variance-components allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "variance-components internal error"); }
}
}  // extern "C"
