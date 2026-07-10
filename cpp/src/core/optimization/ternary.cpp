// SPDX-License-Identifier: CECILL-2.1
//
// TernarySampler (F1) — unimodal-integer search over a single axis, porting the
// nirs4all BinarySearchSampler. It probes low / high / midpoint first, then
// bisects the larger gap adjacent to the current best, converging in O(log n)
// evaluations for a unimodal objective (e.g. PLS n_components). Every other
// parameter falls through to the base uniform sampler.
//
// The search runs in GRID-INDEX space (idx -> low + idx*step), so it honours the
// param `step`, keeps arithmetic bounded, and never proposes an off-grid value.
// RUNNING trials are reserved (so batched asks don't collide) while only
// COMPLETED, active, in-domain trials contribute scores — making the proposal a
// pure function of history and idempotent within one ask.

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
        if (p.kind != N4M_PARAM_INT && p.kind != N4M_PARAM_LOG_INT) continue;
        const std::int64_t lo = static_cast<std::int64_t>(std::llround(p.low));
        const std::int64_t hi = static_cast<std::int64_t>(std::llround(p.high));
        if (hi < lo) continue;
        const std::int64_t st = (p.step >= 1.0) ? static_cast<std::int64_t>(std::llround(p.step)) : 1;
        const std::uint64_t span = static_cast<std::uint64_t>(hi - lo);
        const std::uint64_t grid = span / static_cast<std::uint64_t>(st) + 1;
        if (grid > 100000000ULL) continue;  // too wide for ternary → leave this axis to random
        target_ = p.name;
        low_ = lo;
        high_ = hi;
        step_ = st;
        grid_ = static_cast<std::int64_t>(grid);
        break;  // ternary tunes the first eligible integer axis; others stay random
    }
}

bool TernarySampler::override_numeric(const ParamSpec& p, double* out) {
    if (target_.empty() || p.name != target_) return false;
    *out = static_cast<double>(next_ternary_value());
    return true;
}

std::int64_t TernarySampler::next_ternary_value() const {
    if (grid_ <= 1) return low_;
    const auto value_of = [&](std::int64_t idx) {
        std::int64_t v = low_ + idx * step_;
        return v > high_ ? high_ : v;
    };

    std::set<std::int64_t> reserved;                         // running + completed
    std::vector<std::pair<std::int64_t, double>> hist;       // completed, active, in-domain
    for (const auto& up : trials()) {
        const TrialParam* tp = up->find(target_);
        if (tp == nullptr || !tp->active) continue;
        const std::int64_t v = static_cast<std::int64_t>(std::llround(tp->value));
        if (v < low_ || v > high_) continue;                 // ignore off-domain history
        const std::int64_t idx = (v - low_) / step_;
        if (idx < 0 || idx >= grid_) continue;
        reserved.insert(idx);
        if (up->status == N4M_TRIAL_COMPLETED && up->has_score) hist.emplace_back(idx, up->score);
    }

    // Phase 1 — anchor the triplet (reserving running trials prevents collisions).
    if (reserved.find(0) == reserved.end()) return value_of(0);
    if (reserved.find(grid_ - 1) == reserved.end()) return value_of(grid_ - 1);
    if (reserved.find(grid_ / 2) == reserved.end()) return value_of(grid_ / 2);

    // Best-scoring index so far.
    std::int64_t best_i = 0;
    double best_s = 0.0;
    bool have = false;
    for (const auto& h : hist) {
        if (!have || better(h.second, best_s)) {
            best_s = h.second;
            best_i = h.first;
            have = true;
        }
    }
    const auto usable = [&](std::int64_t idx) {
        return idx >= 0 && idx < grid_ && reserved.find(idx) == reserved.end();
    };
    if (!have) {  // anchors all still running: hand out the next free index
        for (std::int64_t idx = 0; idx < grid_; ++idx) {
            if (usable(idx)) return value_of(idx);
        }
        return value_of(0);
    }

    // Bisect the larger reserved-neighbour gap toward the best.
    std::int64_t below = 0;
    std::int64_t above = grid_ - 1;
    for (const std::int64_t idx : reserved) {
        if (idx < best_i && idx > below) below = idx;
        if (idx > best_i && idx < above) above = idx;
    }
    const std::int64_t left_gap = best_i - below;
    const std::int64_t right_gap = above - best_i;
    std::int64_t cand;
    if (left_gap >= right_gap && left_gap >= 1) {
        cand = below + (best_i - below) / 2;
    } else if (right_gap >= 1) {
        cand = best_i + (above - best_i) / 2;
    } else {
        cand = best_i;
    }

    if (!usable(cand)) {
        for (std::int64_t d = 0; d <= grid_; ++d) {  // spiral out from the best
            if (usable(best_i - d)) return value_of(best_i - d);
            if (usable(best_i + d)) return value_of(best_i + d);
        }
        return value_of(best_i);  // grid exhausted
    }
    return value_of(cand);
}

}  // namespace n4m::core::opt
