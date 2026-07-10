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
    const double median = peers[peers.size() / 2];
    // Prune when this trial is worse than the median at this step.
    return dir == N4M_OPT_MAXIMIZE ? (score < median) : (score > median);
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
        default:  // ASHA / Hyperband / racing reserved for later F2 blocks
            if (status != nullptr) *status = N4M_ERR_NOT_IMPLEMENTED;
            return nullptr;
    }
}

}  // namespace n4m::core::opt
