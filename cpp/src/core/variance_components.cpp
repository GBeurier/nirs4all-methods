// SPDX-License-Identifier: CECILL-2.1
#include "core/variance_components.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace n4m::core::variance_components {
namespace {
using ld = long double;
constexpr ld log_two_pi = 1.837877066409345483560659472811235279722794947275566825634L;
constexpr double grid_low = -8.0;
constexpr double grid_step = 0.125;
constexpr int max_bisections = 200;
constexpr ld asymptotic_log2_f = 64.0L;

// Group means of one group size r, each counted with its multiplicity.
struct Class {
    ld r = 0, count = 0, mean = 0, ss = 0;
};

struct Profile {
    const std::vector<Class>& classes;
    ld ssw, n;

    struct Value { ld f, slope, q; };
    Value operator()(ld gamma) const {
        ld sw = 0, swm = 0, sw2 = 0, slog = 0;
        for (const auto& c : classes) {
            const ld a = gamma + 1.0L / c.r, w = 1.0L / a;
            sw += c.count * w;
            swm += c.count * w * c.mean;
            sw2 += c.count * w * w;
            slog += c.count * std::log(a);
        }
        const ld mu = swm / sw;
        ld q = 0, q2 = 0;
        for (const auto& c : classes) {
            const ld w = 1.0L / (gamma + 1.0L / c.r), d = c.mean - mu;
            const ld dev = c.ss + c.count * d * d;
            q += w * dev;
            q2 += w * w * dev;
        }
        const ld total = ssw + q;
        return {(n - 1) * std::log(total / (n - 1)) + slog + std::log(sw),
                -(n - 1) * q2 / total + sw - sw2 / sw, q};
    }
};
}  // namespace

Fit fit_reml(const std::vector<Group>& groups, const int64_t* multiplicity,
             int within_exponent) {
    Fit fit;
    std::vector<Class> classes;
    ld ssw_q = 0, n = 0, g = 0, sum_log_r = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        const ld m = static_cast<ld>(multiplicity ? multiplicity[i] : 1);
        const auto& grp = groups[i];
        if (m <= 0 || grp.r <= 0) continue;
        const ld r = static_cast<ld>(grp.r);
        if (classes.empty() || classes.back().r != r) classes.push_back(Class{r, 0, 0, 0});
        auto& c = classes.back();
        const ld total = c.count + m, delta = grp.mean - c.mean;
        c.mean += delta * m / total;
        c.ss += m * delta * (grp.mean - c.mean);
        c.count = total;
        ssw_q += m * std::max(0.0L, grp.m2);
        n += m * r;
        g += m;
        sum_log_r += m * std::log(r);
    }
    fit.n_groups = static_cast<int64_t>(g);
    fit.n_obs = static_cast<int64_t>(n);
    const ld nan = std::numeric_limits<ld>::quiet_NaN();
    auto finish = [&fit](int32_t status, ld sb, ld sw, int shift, ld icc, ld loglik) {
        fit.status = status;
        fit.sigma2_between = sb;
        fit.sigma2_within = sw;
        fit.within_shift = shift;
        fit.icc = static_cast<double>(icc);
        fit.loglik = loglik;
        return fit;
    };
    if (g < 2 || n - g < 1) return finish(insufficient, nan, nan, 0, nan, nan);
    // Spread of the group means, unweighted and weighted by group size.
    ld mean_unweighted = 0, mean_weighted = 0;
    for (const auto& c : classes) {
        mean_unweighted += c.count * c.mean / g;
        mean_weighted += c.count * c.r * c.mean / n;
    }
    ld ss_unweighted = 0, ss_weighted = 0;
    for (const auto& c : classes) {
        const ld du = c.mean - mean_unweighted, dw = c.mean - mean_weighted;
        ss_unweighted += c.ss + c.count * du * du;
        ss_weighted += c.r * (c.ss + c.count * dw * dw);
    }
    const int shift = 2 * within_exponent;
    if (ssw_q <= 0) {
        if (ss_unweighted <= 0) return finish(zero_total, nan, nan, 0, nan, nan);
        return finish(within_boundary, ss_unweighted / (g - 1), 0, 0, 1, nan);
    }
    // log2 of the ANOVA ratio F = MSB / MSW, without forming MSW.
    const ld msb = ss_weighted / (g - 1), msw_q = ssw_q / (n - g);
    const ld log2_f = msb > 0 ? std::log2(msb) - std::log2(msw_q) - static_cast<ld>(shift)
                              : -std::numeric_limits<ld>::infinity();
    if (log2_f > asymptotic_log2_f) {
        // REML limit, exact to double precision: s2w = SSW/(N-g),
        // s2b = sample variance of the group means.
        const ld sb = ss_unweighted / (g - 1);
        const ld ratio = std::scalbn(msw_q / sb, shift);
        const ld log_sw = std::log(msw_q) + static_cast<ld>(shift) * std::log(2.0L);
        const ld loglik = -0.5L * ((n - 1) * log_two_pi + (n - g) * (log_sw + 1) + sum_log_r +
                                   (g - 1) * (std::log(sb) + 1) + std::log(g));
        fit.candidates = 1;
        return finish(interior, sb, msw_q, shift, 1 / (1 + ratio), loglik);
    }
    const ld ssw = std::scalbn(ssw_q, shift);
    const Profile profile{classes, ssw, n};
    auto loglik = [&](ld f) { return -0.5L * ((n - 1) * (log_two_pi + 1) + sum_log_r + f); };
    const double top =
        static_cast<double>(std::log10(1e3L) + std::max(0.0L, log2_f * std::log10(2.0L)));
    const auto cells = static_cast<size_t>(std::ceil((top - grid_low) / grid_step));
    std::vector<ld> gamma(cells + 2, 0.0L);
    std::vector<Profile::Value> at(gamma.size());
    for (size_t k = 1; k < gamma.size(); ++k)
        gamma[k] = std::pow(10.0L, grid_low + grid_step * static_cast<double>(k - 1));
    for (size_t k = 0; k < gamma.size(); ++k) at[k] = profile(gamma[k]);
    const size_t last = gamma.size() - 1;
    if (at[last].slope < 0) return finish(not_converged, nan, nan, 0, nan, nan);
    // Every local minimum of f on [0, gamma_max]: the boundary when
    // f'(0) >= 0, and each - to + sign change of f' refined by bisection.
    const ld tolerance = 4 * static_cast<ld>(std::numeric_limits<double>::epsilon());
    bool found = false;
    ld best_gamma = 0, best_f = 0;
    auto consider = [&](ld gm, ld fv) {
        ++fit.candidates;
        if (!found || fv < best_f) {
            best_gamma = gm;
            best_f = fv;
            found = true;
        }
    };
    if (at[0].slope >= 0) consider(0, at[0].f);
    for (size_t k = 0; k < last; ++k) {
        if (k > 0 && at[k].slope == 0) {
            consider(gamma[k], at[k].f);
            continue;
        }
        if (!(at[k].slope < 0 && at[k + 1].slope > 0)) continue;
        ld lo = gamma[k], hi = gamma[k + 1];
        int steps = 0;
        while (hi - lo > tolerance * hi && steps < max_bisections) {
            const ld mid = lo + 0.5L * (hi - lo);
            if (profile(mid).slope < 0) lo = mid; else hi = mid;
            ++steps;
        }
        fit.iterations += steps;
        if (hi - lo > tolerance * hi) return finish(not_converged, nan, nan, 0, nan, nan);
        const ld root = lo + 0.5L * (hi - lo);
        consider(root, profile(root).f);
    }
    if (!found) return finish(not_converged, nan, nan, 0, nan, nan);
    const auto best = profile(best_gamma);
    const ld s = (ssw + best.q) / (n - 1);
    if (best_gamma == 0) return finish(between_boundary, 0, s, 0, 0, loglik(best.f));
    return finish(interior, best_gamma * s, s, 0, best_gamma / (1 + best_gamma), loglik(best.f));
}

}  // namespace n4m::core::variance_components
