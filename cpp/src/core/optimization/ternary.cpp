// SPDX-License-Identifier: CECILL-2.1
//
// TernarySampler (F1) — unimodal-integer search over a single axis, porting the
// nirs4all BinarySearchSampler. It probes low / high / midpoint first, then
// bisects the larger gap adjacent to the current best, converging in O(log n)
// evaluations for a unimodal objective (e.g. PLS n_components). Every other
// parameter falls through to the base uniform sampler. The proposal is a pure
// function of the completed-trial history, so it is idempotent within one ask().

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace n4m::core::opt {

TernarySampler::TernarySampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    for (const auto& p : space.params) {
        if (p.kind == N4M_PARAM_INT || p.kind == N4M_PARAM_LOG_INT) {
            target_ = p.name;
            low_ = static_cast<std::int64_t>(std::llround(p.low));
            high_ = static_cast<std::int64_t>(std::llround(p.high));
            break;  // ternary tunes the first integer axis; others stay random
        }
    }
}

bool TernarySampler::override_numeric(const ParamSpec& p, double* out) {
    if (target_.empty() || p.name != target_) return false;
    *out = static_cast<double>(next_ternary_value());
    return true;
}

std::int64_t TernarySampler::next_ternary_value() const {
    if (low_ >= high_) return low_;

    // Completed history for the target axis.
    std::vector<std::pair<std::int64_t, double>> hist;
    std::set<std::int64_t> tested;
    for (const auto& up : trials()) {
        if (up->status != N4M_TRIAL_COMPLETED || !up->has_score) continue;
        const TrialParam* tp = up->find(target_);
        if (tp == nullptr) continue;
        const auto v = static_cast<std::int64_t>(std::llround(tp->value));
        hist.emplace_back(v, up->score);
        tested.insert(v);
    }

    // Phase 1 — anchor the triplet.
    if (tested.find(low_) == tested.end()) return low_;
    if (tested.find(high_) == tested.end()) return high_;
    const std::int64_t mid = low_ + (high_ - low_) / 2;
    if (tested.find(mid) == tested.end()) return mid;

    // Current best value.
    std::int64_t best_v = low_;
    double best_s = 0.0;
    bool have = false;
    for (const auto& h : hist) {
        if (!have || better(h.second, best_s)) {
            best_s = h.second;
            best_v = h.first;
            have = true;
        }
    }

    // Nearest tested neighbours bracketing best_v.
    std::int64_t below = low_;
    std::int64_t above = high_;
    for (const std::int64_t v : tested) {
        if (v < best_v && v > below) below = v;
        if (v > best_v && v < above) above = v;
    }
    const std::int64_t left_gap = best_v - below;
    const std::int64_t right_gap = above - best_v;

    std::int64_t cand;
    if (left_gap >= right_gap && left_gap >= 1) {
        cand = below + (best_v - below) / 2;
    } else if (right_gap >= 1) {
        cand = best_v + (above - best_v) / 2;
    } else {
        cand = best_v;
    }

    // Ensure the candidate is fresh and in range; else spiral out from best_v.
    auto usable = [&](std::int64_t v) {
        return v >= low_ && v <= high_ && tested.find(v) == tested.end();
    };
    if (!usable(cand)) {
        for (std::int64_t d = 0; d <= (high_ - low_); ++d) {
            if (usable(best_v - d)) return best_v - d;
            if (usable(best_v + d)) return best_v + d;
        }
        return best_v;  // space exhausted
    }
    return cand;
}

}  // namespace n4m::core::opt
