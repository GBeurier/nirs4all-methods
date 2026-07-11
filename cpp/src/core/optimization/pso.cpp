// SPDX-License-Identifier: CECILL-2.1
//
// PsoSampler (F3) — binary/continuous Particle Swarm Optimization over the unit
// hypercube (Kennedy & Eberhart). A swarm of `swarm_size` particles is asked out
// per iteration; once the iteration's trials are scored, each particle's personal
// best and the global best update, velocities move toward them (Clerc & Kennedy
// 2002 constants w=0.729, c1=c2=1.494), and positions advance for the next
// iteration. Candidate decode is the shared Optimizer::decode_candidate.

#include "core/optimization/optimizer.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace n4m::core::opt {

namespace {
double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v >= 1.0) return std::nextafter(1.0, 0.0);
    return v;
}
}  // namespace

PsoSampler::PsoSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    (void)space;
    (void)opts;
}

void PsoSampler::ensure_iteration(std::int64_t member_base) {
    const bool first = iter_base_id_ < 0;
    const bool next = !first && member_base >= iter_base_id_ + swarm_size_;
    if (!first && !next) return;

    const std::size_t P = space_.params.size();
    const std::size_t S = static_cast<std::size_t>(swarm_size_);
    const double worst = (direction() == N4M_OPT_MAXIMIZE)
                             ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();

    if (first) {
        pos_.assign(S, std::vector<double>(P, 0.0));
        vel_.assign(S, std::vector<double>(P, 0.0));
        pbest_pos_.assign(S, std::vector<double>(P, 0.0));
        pbest_score_.assign(S, worst);
        for (std::size_t k = 0; k < S; ++k) {
            for (std::size_t j = 0; j < P; ++j) {
                pos_[k][j] = n4m_rng_next_double(&rng_);
                vel_[k][j] = (n4m_rng_next_double(&rng_) * 2.0 - 1.0) * 0.1;
            }
            pbest_pos_[k] = pos_[k];
        }
        gbest_pos_.assign(P, 0.5);
        have_gbest_ = false;
        iter_base_id_ = member_base;
        return;
    }

    // Update personal bests from the just-finished iteration's scores.
    for (std::size_t k = 0; k < S; ++k) {
        const std::int64_t tid = iter_base_id_ + static_cast<std::int64_t>(k);
        double score = worst;
        for (const auto& up : trials()) {
            if (up->id == tid) {
                if (up->status == N4M_TRIAL_COMPLETED && up->has_score) score = up->score;
                break;
            }
        }
        if (better(score, pbest_score_[k])) {
            pbest_pos_[k] = pos_[k];
            pbest_score_[k] = score;
        }
    }
    // Global best = best personal best.
    for (std::size_t k = 0; k < S; ++k) {
        if (!have_gbest_ || better(pbest_score_[k], gbest_score_)) {
            gbest_pos_ = pbest_pos_[k];
            gbest_score_ = pbest_score_[k];
            have_gbest_ = true;
        }
    }
    // Velocity + position update.
    for (std::size_t k = 0; k < S; ++k) {
        for (std::size_t j = 0; j < P; ++j) {
            const double r1 = n4m_rng_next_double(&rng_);
            const double r2 = n4m_rng_next_double(&rng_);
            double v = w_ * vel_[k][j] + c1_ * r1 * (pbest_pos_[k][j] - pos_[k][j]) +
                       c2_ * r2 * (gbest_pos_[j] - pos_[k][j]);
            constexpr double kVmax = 0.5;  // cap velocity to half the unit range
            if (v > kVmax) v = kVmax;
            else if (v < -kVmax) v = -kVmax;
            vel_[k][j] = v;
            pos_[k][j] = clamp01(pos_[k][j] + v);
        }
    }
    iter_base_id_ = member_base;
}

bool PsoSampler::sample(::n4m_trial_s& t,
                        const std::vector<std::pair<std::string, double>>* forced) {
    // Synchronous update (LIAR_NONE): refuse to cross a swarm-iteration boundary
    // until the current iteration is fully resolved (partial ask_batch instead of
    // updating on unscored particles).
    if (iter_base_id_ >= 0 && t.id >= iter_base_id_ + swarm_size_ &&
        resolved_in_range(iter_base_id_, swarm_size_) < swarm_size_) {
        return false;
    }
    ensure_iteration(t.id);
    std::size_t k = static_cast<std::size_t>(t.id - iter_base_id_);
    if (k >= static_cast<std::size_t>(swarm_size_)) k = static_cast<std::size_t>(swarm_size_) - 1;
    decode_candidate(pos_[k], t, forced);
    return true;
}

}  // namespace n4m::core::opt
