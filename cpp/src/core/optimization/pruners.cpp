// SPDX-License-Identifier: CECILL-2.1
//
// Pruners (F2) — early-stopping policies over the tell_intermediate() score
// stream, composed with (and orthogonal to) the sampler. F2 ships the median
// stopping rule; ASHA / Hyperband / racing land behind the same factory.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace n4m::core::opt {

namespace {
// Running mean + count over a trial's recorded intermediate scores.
void mean_and_count(const ::n4m_trial_s& t, double& mean, int& n) {
    double sum = 0.0;
    n = 0;
    for (const auto& im : t.intermediates) {
        sum += im.second;
        ++n;
    }
    mean = n > 0 ? sum / n : 0.0;
}
}  // namespace

bool MedianPruner::should_prune(const ::n4m_trial_s& trial, std::int32_t step, double score,
                                const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                                n4m_opt_direction_t dir) const {
    if (step < warmup_steps_) return false;

    // Peer scores reported at the SAME step (excluding this trial).
    std::vector<double> peers;
    for (const auto& up : trials) {
        if (up.get() == &trial) continue;
        for (const auto& im : up->intermediates) {
            if (im.first == step) {
                peers.push_back(im.second);
                break;
            }
        }
    }
    if (static_cast<std::int32_t>(peers.size()) < min_peers_) return false;

    std::sort(peers.begin(), peers.end());
    const std::size_t n = peers.size();
    const double median =
        (n % 2 == 1) ? peers[n / 2] : 0.5 * (peers[n / 2 - 1] + peers[n / 2]);  // true 50th pct
    // Prune when this trial is strictly worse than the peer median at this step.
    return dir == N4M_OPT_MAXIMIZE ? (score < median) : (score > median);
}

bool AshaPruner::should_prune(const ::n4m_trial_s& trial, std::int32_t step, double score,
                              const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                              n4m_opt_direction_t dir) const {
    // Peer scores reported at the same rung (excluding this trial).
    std::vector<double> peers;
    for (const auto& up : trials) {
        if (up.get() == &trial) continue;
        for (const auto& im : up->intermediates) {
            if (im.first == step) {
                peers.push_back(im.second);
                break;
            }
        }
    }
    const std::int32_t n = static_cast<std::int32_t>(peers.size()) + 1;  // include self
    if (n < reduction_factor_) return false;  // too few to halve this rung yet
    const std::int32_t n_promote = std::max(1, n / reduction_factor_);
    std::int32_t better = 0;
    for (const double p : peers) {
        if (dir == N4M_OPT_MAXIMIZE ? (p > score) : (p < score)) ++better;
    }
    return better >= n_promote;  // ranked outside the surviving top → prune
}

bool RacingPruner::should_prune(const ::n4m_trial_s& trial, std::int32_t /*step*/, double /*score*/,
                                const std::vector<std::unique_ptr<::n4m_trial_s>>& trials,
                                n4m_opt_direction_t dir) const {
    double mu_self = 0.0;
    int n_self = 0;
    mean_and_count(trial, mu_self, n_self);
    if (n_self < 2) return false;  // need repeated observations for a bound

    // Observed score range across all trials (the Hoeffding loss range).
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const auto& up : trials) {
        for (const auto& im : up->intermediates) {
            lo = std::min(lo, im.second);
            hi = std::max(hi, im.second);
        }
    }
    const double range = (hi > lo) ? (hi - lo) : 1.0;

    // Best peer by mean (with ≥2 observations).
    bool have_best = false;
    double mu_best = 0.0;
    int n_best = 0;
    for (const auto& up : trials) {
        if (up.get() == &trial) continue;
        double m = 0.0;
        int n = 0;
        mean_and_count(*up, m, n);
        if (n < 2) continue;
        const bool better = dir == N4M_OPT_MAXIMIZE ? (m > mu_best) : (m < mu_best);
        if (!have_best || better) {
            mu_best = m;
            n_best = n;
            have_best = true;
        }
    }
    if (!have_best) return false;

    const double eps_self = range * std::sqrt(std::log(2.0 / delta_) / (2.0 * n_self));
    const double eps_best = range * std::sqrt(std::log(2.0 / delta_) / (2.0 * n_best));
    // Prune when this trial's optimistic bound is still worse than the best's pessimistic bound.
    return dir == N4M_OPT_MAXIMIZE ? ((mu_self + eps_self) < (mu_best - eps_best))
                                   : ((mu_self - eps_self) > (mu_best + eps_best));
}

std::unique_ptr<Pruner> make_pruner(const n4m_optimizer_options_t& opts, n4m_status_t* status) {
    if (status != nullptr) *status = N4M_OK;
    switch (opts.pruner) {
        case N4M_PRUNER_NONE:
            return nullptr;  // never prunes
        case N4M_PRUNER_MEDIAN: {
            const std::int32_t min_peers = opts.n_startup_trials > 0 ? opts.n_startup_trials : 1;
            return std::make_unique<MedianPruner>(min_peers, /*warmup_steps=*/0);
        }
        case N4M_PRUNER_ASHA:
            return std::make_unique<AshaPruner>(/*reduction_factor=*/3);
        case N4M_PRUNER_RACING:
            return std::make_unique<RacingPruner>(/*delta=*/0.05);
        default:  // Hyperband reserved (needs the bracket scheduler / total budget)
            if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;
            return nullptr;
    }
}

}  // namespace n4m::core::opt
