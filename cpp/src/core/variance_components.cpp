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
constexpr double grid_step = 0.25;
constexpr int max_bisections = 200;

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

Fit fit_reml(const std::vector<Group>& groups, const int64_t* multiplicity) {
    Fit fit;
    std::vector<Class> classes;
    ld ssw = 0, n = 0, g = 0, sum_log_r = 0;
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
        ssw += m * std::max(0.0L, grp.m2);
        n += m * r;
        g += m;
        sum_log_r += m * std::log(r);
    }
    fit.n_groups = static_cast<int64_t>(g);
    fit.n_obs = static_cast<int64_t>(n);
    const ld nan = std::numeric_limits<ld>::quiet_NaN();
    auto finish = [&fit](int32_t status, ld sb, ld sw, ld icc, ld loglik) {
        fit.status = status;
        fit.sigma2_between = static_cast<double>(sb);
        fit.sigma2_within = static_cast<double>(sw);
        fit.icc = static_cast<double>(icc);
        fit.loglik = static_cast<double>(loglik);
        return fit;
    };
    if (g < 2 || n - g < 1) return finish(insufficient, nan, nan, nan, nan);
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
    if (ssw <= 0) {
        if (ss_unweighted <= 0) return finish(zero_total, nan, nan, nan, nan);
        return finish(within_boundary, ss_unweighted / (g - 1), 0, 1, nan);
    }
    const Profile profile{classes, ssw, n};
    auto loglik = [&](ld f) { return -0.5L * ((n - 1) * (log_two_pi + 1) + sum_log_r + f); };
    const ld anova_f = (ss_weighted / (g - 1)) / (ssw / (n - g));
    const double top = static_cast<double>(std::log10(1e3L * std::max(1.0L, anova_f)));
    const auto cells = static_cast<int>(std::ceil((top - grid_low) / grid_step));
    std::vector<ld> gamma(static_cast<size_t>(cells) + 2, 0.0L), f(gamma.size());
    for (size_t k = 1; k < gamma.size(); ++k)
        gamma[k] = std::pow(10.0L, grid_low + grid_step * static_cast<double>(k - 1));
    size_t best = 0;
    for (size_t k = 0; k < gamma.size(); ++k) {
        f[k] = profile(gamma[k]).f;
        if (f[k] < f[best]) best = k;
    }
    const size_t last = gamma.size() - 1;
    for (size_t k = 0; k <= last; ++k)
        if ((k == 0 || f[k] < f[k - 1]) && (k == last || f[k] <= f[k + 1])) ++fit.local_minima;
    if (best == last) return finish(not_converged, nan, nan, nan, nan);
    ld lo = 0, hi = 0;
    const ld slope = profile(gamma[best]).slope;
    if (best == 0) {
        if (slope >= 0) {
            const ld s = (ssw + profile(0).q) / (n - 1);
            return finish(between_boundary, 0, s, 0, loglik(f[0]));
        }
        lo = 0;
        hi = gamma[1];
    } else if (slope > 0) {
        lo = gamma[best - 1];
        hi = gamma[best];
    } else {
        lo = gamma[best];
        hi = gamma[best + 1];
    }
    if (slope != 0 && !(profile(lo).slope < 0 && profile(hi).slope > 0))
        return finish(not_converged, nan, nan, nan, nan);
    if (slope == 0) lo = hi = gamma[best];
    const ld tolerance = 4 * static_cast<ld>(std::numeric_limits<double>::epsilon());
    while (hi - lo > tolerance * hi && fit.iterations < max_bisections) {
        const ld mid = lo + 0.5L * (hi - lo);
        if (profile(mid).slope < 0) lo = mid; else hi = mid;
        ++fit.iterations;
    }
    if (hi - lo > tolerance * hi) return finish(not_converged, nan, nan, nan, nan);
    const ld gamma_hat = lo + 0.5L * (hi - lo);
    const auto at = profile(gamma_hat);
    const ld s = (ssw + at.q) / (n - 1);
    return finish(interior, gamma_hat * s, s, gamma_hat / (1 + gamma_hat), loglik(at.f));
}

}  // namespace n4m::core::variance_components
