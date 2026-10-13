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
constexpr int32_t max_bootstrap = 100000;

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

// Groups observed in one feature, scaled by max |x|, sorted by (r, group).
struct Column {
    std::vector<vc::Group> groups;
    std::vector<int64_t> index;  // dense group index of each entry
    double scale = 1.0;
};

int scan_column(const n4m_matrix_view_t& X, int64_t j, const std::vector<int64_t>& group,
                int64_t n_groups, const uint8_t* mask, Column& out) {
    const auto* data = static_cast<const double*>(X.data);
    auto valid = [&](int64_t i) {
        return !mask || mask[sz(i) * sz(X.cols) + sz(j)] != 0;
    };
    double scale = 0.0;
    for (int64_t i = 0; i < X.rows; ++i) {
        if (!valid(i)) continue;
        const double x = data[i * X.row_stride + j * X.col_stride];
        if (!std::isfinite(x)) return 1;
        scale = std::max(scale, std::abs(x));
    }
    out.scale = scale > 0.0 ? scale : 1.0;
    std::vector<vc::Group> all(sz(n_groups));
    for (int64_t i = 0; i < X.rows; ++i) {
        if (!valid(i)) continue;
        auto& g = all[sz(group[sz(i)])];
        const long double x = static_cast<long double>(data[i * X.row_stride + j * X.col_stride]) /
                              static_cast<long double>(out.scale);
        ++g.r;
        const long double delta = x - g.mean;
        g.mean += delta / static_cast<long double>(g.r);
        g.m2 += delta * (x - g.mean);
    }
    out.index.clear();
    for (int64_t k = 0; k < n_groups; ++k)
        if (all[sz(k)].r > 0) out.index.push_back(k);
    std::stable_sort(out.index.begin(), out.index.end(),
                     [&all](int64_t a, int64_t b) { return all[sz(a)].r < all[sz(b)].r; });
    out.groups.clear();
    for (int64_t k : out.index) out.groups.push_back(all[sz(k)]);
    return 0;
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
        Column column;
        for (int64_t j = 0; j < X->cols; ++j) {
            if (scan_column(*X, j, group, n_groups, valid_mask, column))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked input");
            const auto fit = vc::fit_reml(column.groups, nullptr);
            const double s2 = column.scale * column.scale;
            double* row = out + 8 * j;
            row[0] = fit.sigma2_between * s2;
            row[1] = fit.sigma2_within * s2;
            row[2] = fit.icc;
            row[3] = fit.loglik - static_cast<double>(fit.n_obs - 1) * std::log(column.scale);
            row[4] = static_cast<double>(fit.n_groups);
            row[5] = static_cast<double>(fit.n_obs);
            row[6] = static_cast<double>(fit.local_minima);
            row[7] = static_cast<double>(fit.iterations);
            band_status[j] = fit.status;
            if (usable(fit.status) && (!std::isfinite(row[0]) || !std::isfinite(row[1])))
                return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                            "variance component is not representable as f64");
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
    if (!(confidence > 0.0 && confidence < 1.0) || n_bootstrap < 0 || n_bootstrap > max_bootstrap)
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "invalid confidence or bootstrap count");
    try {
        int64_t n_groups = 0, n_units = 0;
        const auto group = dense(group_ids, X->rows, n_groups);
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
                                 (static_cast<long double>(n_groups) + 512.0L);
        if (work > max_bootstrap_work)
            return fail(ctx, N4M_ERR_UNSUPPORTED, "bootstrap exceeds 2.5e9 operation budget");
        const double alpha = 1.0 - confidence;
        std::map<double, double> chi_lo, chi_hi;
        std::map<std::pair<double, double>, double> f_upper;
        auto cached = [](auto& cache, auto key, auto compute) {
            auto it = cache.find(key);
            if (it == cache.end()) it = cache.emplace(key, compute()).first;
            return it->second;
        };
        Column column;
        std::vector<int64_t> counts(sz(n_units)), multiplicity;
        std::vector<double> sd_draws, icc_draws;
        for (int64_t j = 0; j < X->cols; ++j) {
            if (scan_column(*X, j, group, n_groups, valid_mask, column))
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "nonfinite unmasked input");
            double* row = out + 8 * j;
            std::fill(row, row + 8, nan_value);
            n_valid_replicates[j] = 0;
            const auto& gs = column.groups;
            const auto g = static_cast<int64_t>(gs.size());
            int64_t n = 0;
            long double ssw = 0;
            for (const auto& grp : gs) {
                n += grp.r;
                ssw += std::max(0.0L, grp.m2);
            }
            const long double scale = static_cast<long double>(column.scale);
            if (n - g >= 1) {
                const auto df = static_cast<double>(n - g);
                const double qlo = cached(chi_lo, df, [&] { return dist::chi2_quantile(alpha / 2, df); });
                const double qhi = cached(chi_hi, df, [&] { return dist::chi2_quantile(1 - alpha / 2, df); });
                row[0] = static_cast<double>(scale * std::sqrt(ssw / static_cast<long double>(qhi)));
                row[1] = static_cast<double>(scale * std::sqrt(ssw / static_cast<long double>(qlo)));
                const int64_t r = gs.front().r;
                if (g >= 2 && r >= 2 && gs.back().r == r) {
                    long double mean = 0, ssb = 0;
                    for (const auto& grp : gs) mean += grp.mean / static_cast<long double>(g);
                    for (const auto& grp : gs) ssb += (grp.mean - mean) * (grp.mean - mean);
                    const long double msb = static_cast<long double>(r) * ssb / static_cast<long double>(g - 1);
                    const long double msw = ssw / static_cast<long double>(n - g);
                    const auto d1 = static_cast<double>(g - 1);
                    if (msw > 0) {
                        const long double F = msb / msw;
                        const long double lower = F / static_cast<long double>(cached(
                            f_upper, std::make_pair(d1, df),
                            [&] { return dist::f_quantile(1 - alpha / 2, d1, df); }));
                        const long double upper = F * static_cast<long double>(cached(
                            f_upper, std::make_pair(df, d1),
                            [&] { return dist::f_quantile(1 - alpha / 2, df, d1); }));
                        const auto rr = static_cast<long double>(r);
                        row[2] = static_cast<double>((lower - 1) / (lower + rr - 1));
                        row[3] = static_cast<double>((upper - 1) / (upper + rr - 1));
                    } else if (msb > 0) {
                        row[2] = row[3] = 1.0;
                    }
                }
            }
            if (n_bootstrap == 0) continue;
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
                    rssw += static_cast<long double>(m) * std::max(0.0L, gs[k].m2);
                    rn += static_cast<long double>(m * gs[k].r);
                    rg += static_cast<long double>(m);
                }
                const auto fit = vc::fit_reml(gs, multiplicity.data());
                if (!usable(fit.status)) continue;
                sd_draws.push_back(static_cast<double>(scale * std::sqrt(rssw / (rn - rg))));
                icc_draws.push_back(fit.icc);
            }
            n_valid_replicates[j] = static_cast<int64_t>(sd_draws.size());
            if (sd_draws.empty()) continue;
            row[4] = type7(sd_draws, alpha / 2);
            row[5] = type7(sd_draws, 1 - alpha / 2);
            row[6] = type7(icc_draws, alpha / 2);
            row[7] = type7(icc_draws, 1 - alpha / 2);
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return fail(ctx, N4M_ERR_OUT_OF_MEMORY, "variance-components allocation failed");
    } catch (...) { return fail(ctx, N4M_ERR_INTERNAL, "variance-components internal error"); }
}
}  // extern "C"
