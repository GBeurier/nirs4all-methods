// SPDX-License-Identifier: CECILL-2.1
//
// TpeSampler (F4) — univariate Tree-structured Parzen Estimator. Per parameter,
// the completed history is split into a "good" set (top γ by score) and the
// rest; l(x)/g(x) are Parzen (KDE) densities over the good/bad numeric values in
// unit space (or Laplace-smoothed category frequencies). n_ei candidates are
// drawn from l and the one maximising l(x)/g(x) is returned. It plugs into the
// base sampler's per-parameter hooks (override_numeric / override_categorical),
// so the base handles constraints, conditions, forced values and sorted tuples.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace n4m::core::opt {

namespace {
double kde(double x, const std::vector<double>& centres, double h) {
    if (centres.empty()) return 1e-12;
    double s = 0.0;
    const double inv = 1.0 / h;
    for (const double c : centres) {
        const double d = (x - c) * inv;
        s += std::exp(-0.5 * d * d);
    }
    return s / (static_cast<double>(centres.size()) * h) + 1e-12;
}
}  // namespace

TpeSampler::TpeSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    (void)space;
    if (opts.n_startup_trials > 0) n_startup_ = opts.n_startup_trials;
}

bool TpeSampler::override_numeric(const ParamSpec& p, double* out) {
    // Completed (unit value, score) history for this axis.
    std::vector<std::pair<double, double>> hist;  // (unit, score)
    for (const auto& up : trials()) {
        if (up->status != N4M_TRIAL_COMPLETED || !up->has_score) continue;
        const TrialParam* tp = up->find(p.name);
        if (tp == nullptr || !tp->active) continue;
        hist.emplace_back(unit_from_numeric(p, tp->value), up->score);
    }
    const int n = static_cast<int>(hist.size());
    if (n < n_startup_) return false;  // random until enough history

    const bool maximize = direction() == N4M_OPT_MAXIMIZE;
    std::sort(hist.begin(), hist.end(), [maximize](const auto& a, const auto& b) {
        return maximize ? a.second > b.second : a.second < b.second;
    });
    int n_good = static_cast<int>(std::ceil(gamma_ * n));
    if (n_good < 1) n_good = 1;
    if (n_good > n - 1) n_good = n - 1;

    std::vector<double> good, bad;
    good.reserve(static_cast<std::size_t>(n_good));
    bad.reserve(static_cast<std::size_t>(n - n_good));
    for (int i = 0; i < n; ++i) {
        (i < n_good ? good : bad).push_back(hist[static_cast<std::size_t>(i)].first);
    }
    const double h = std::max(0.05, 1.0 / std::sqrt(static_cast<double>(n)));  // KDE bandwidth (unit space)

    // Draw n_ei candidates from l (a good centre + Gaussian noise) and keep the
    // one maximising l/g.
    double best_u = 0.5;
    double best_ratio = -1.0;
    for (std::int32_t e = 0; e < n_ei_; ++e) {
        const int gi = static_cast<int>(n4m_rng_next_double(&rng_) * n_good);
        double c = good[static_cast<std::size_t>(gi < n_good ? gi : n_good - 1)] +
                   n4m_rng_next_normal(&rng_) * h;
        if (c < 0.0) c = 0.0;
        if (c >= 1.0) c = std::nextafter(1.0, 0.0);
        const double ratio = kde(c, good, h) / kde(c, bad, h);
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_u = c;
        }
    }
    *out = numeric_from_unit(p, best_u);
    return true;
}

bool TpeSampler::override_categorical(const ParamSpec& p, std::int32_t* out_index) {
    const int K = static_cast<int>(std::max(p.labels.size(), p.num_values.size()));
    if (K <= 0) return false;

    std::vector<std::pair<int, double>> hist;  // (index, score)
    for (const auto& up : trials()) {
        if (up->status != N4M_TRIAL_COMPLETED || !up->has_score) continue;
        const TrialParam* tp = up->find(p.name);
        if (tp == nullptr || !tp->active || tp->cat_index < 0) continue;
        hist.emplace_back(tp->cat_index, up->score);
    }
    const int n = static_cast<int>(hist.size());
    if (n < n_startup_) return false;

    const bool maximize = direction() == N4M_OPT_MAXIMIZE;
    std::sort(hist.begin(), hist.end(), [maximize](const auto& a, const auto& b) {
        return maximize ? a.second > b.second : a.second < b.second;
    });
    int n_good = static_cast<int>(std::ceil(gamma_ * n));
    if (n_good < 1) n_good = 1;
    if (n_good > n - 1) n_good = n - 1;

    std::vector<double> good_ct(static_cast<std::size_t>(K), 1.0);  // Laplace prior
    std::vector<double> bad_ct(static_cast<std::size_t>(K), 1.0);
    for (int i = 0; i < n; ++i) {
        const int idx = hist[static_cast<std::size_t>(i)].first;
        if (idx < 0 || idx >= K) continue;
        (i < n_good ? good_ct : bad_ct)[static_cast<std::size_t>(idx)] += 1.0;
    }
    const double good_tot = static_cast<double>(n_good) + K;
    const double bad_tot = static_cast<double>(n - n_good) + K;
    int best = 0;
    double best_ratio = -1.0;
    for (int c = 0; c < K; ++c) {
        const double l = good_ct[static_cast<std::size_t>(c)] / good_tot;
        const double g = bad_ct[static_cast<std::size_t>(c)] / bad_tot;
        const double ratio = l / g;
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best = c;
        }
    }
    *out_index = best;
    return true;
}

}  // namespace n4m::core::opt
