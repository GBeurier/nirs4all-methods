// SPDX-License-Identifier: CECILL-2.1
//
// Pruners (F2) — early-stopping policies over the tell_intermediate() score
// stream, composed with (and orthogonal to) the sampler. F2 ships the median
// stopping rule; ASHA / Hyperband / racing land behind the same factory.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace n4m::core::opt {

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
        default:  // Hyperband / racing reserved for later F2 blocks
            if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;
            return nullptr;
    }
}

}  // namespace n4m::core::opt
