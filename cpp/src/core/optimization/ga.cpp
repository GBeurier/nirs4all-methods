// SPDX-License-Identifier: CECILL-2.1
//
// GaSampler (F3) — real-coded genetic algorithm over the unit hypercube. A
// candidate is a unit vector u∈[0,1)^P decoded per parameter; a generation of
// pop_size candidates is asked out, then tournament selection + uniform
// crossover + Gaussian mutation (with elitism) produce the next generation once
// the current one's trials have scores.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace n4m::core::opt {

GaSampler::GaSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    (void)space;
    (void)opts;
}

namespace {
double clamp_unit(double u) {
    if (u < 0.0) return 0.0;
    if (u >= 1.0) return std::nextafter(1.0, 0.0);
    return u;
}
}  // namespace

void GaSampler::ensure_generation(std::int64_t member_base) {
    const bool need_first = gen_base_id_ < 0;
    const bool need_next = !need_first && member_base >= gen_base_id_ + pop_size_;
    if (!need_first && !need_next) return;

    const std::size_t P = genome_length();
    std::vector<std::vector<double>> next(static_cast<std::size_t>(pop_size_),
                                          std::vector<double>(P, 0.0));

    if (need_first) {
        for (auto& g : next) {
            for (auto& v : g) v = n4m_rng_next_double(&rng_);
        }
        gen_ = std::move(next);
        gen_base_id_ = member_base;
        return;
    }

    // Evolve: pair each current-gen unit vector with its trial score.
    const double worst = (direction() == N4M_OPT_MAXIMIZE)
                             ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
    std::vector<double> fitness(static_cast<std::size_t>(pop_size_), worst);
    for (std::int32_t k = 0; k < pop_size_; ++k) {
        const std::int64_t tid = gen_base_id_ + k;
        for (const auto& up : trials()) {
            if (up->id == tid) {
                if (up->status == N4M_TRIAL_COMPLETED && up->has_score) {
                    fitness[static_cast<std::size_t>(k)] = up->score;
                }
                break;
            }
        }
    }
    auto tournament = [&]() -> const std::vector<double>& {
        const int a = static_cast<int>(n4m_rng_next_double(&rng_) * pop_size_);
        const int b = static_cast<int>(n4m_rng_next_double(&rng_) * pop_size_);
        const int ai = a < pop_size_ ? a : pop_size_ - 1;
        const int bi = b < pop_size_ ? b : pop_size_ - 1;
        return better(fitness[static_cast<std::size_t>(ai)], fitness[static_cast<std::size_t>(bi)])
                   ? gen_[static_cast<std::size_t>(ai)]
                   : gen_[static_cast<std::size_t>(bi)];
    };
    for (std::int32_t m = 0; m < pop_size_; ++m) {
        const std::vector<double>& p1 = tournament();
        const std::vector<double>& p2 = tournament();
        std::vector<double>& child = next[static_cast<std::size_t>(m)];
        for (std::size_t j = 0; j < P; ++j) {
            double v = (n4m_rng_next_double(&rng_) < 0.5) ? p1[j] : p2[j];  // uniform crossover
            if (n4m_rng_next_double(&rng_) < mutation_rate_) {
                v = clamp_unit(v + n4m_rng_next_normal(&rng_) * mutation_sigma_);
            }
            child[j] = v;
        }
    }
    // Elitism: carry the best current-gen unit vector unchanged.
    int best_k = 0;
    for (std::int32_t k = 1; k < pop_size_; ++k) {
        if (better(fitness[static_cast<std::size_t>(k)], fitness[static_cast<std::size_t>(best_k)])) {
            best_k = k;
        }
    }
    next[0] = gen_[static_cast<std::size_t>(best_k)];

    gen_ = std::move(next);
    gen_base_id_ = member_base;
}

bool GaSampler::sample(::n4m_trial_s& t, const std::vector<std::pair<std::string, double>>* forced) {
    ensure_generation(t.id);
    std::size_t k = static_cast<std::size_t>(t.id - gen_base_id_);
    if (k >= static_cast<std::size_t>(pop_size_)) k = static_cast<std::size_t>(pop_size_) - 1;
    decode_candidate(gen_[k], t, forced);
    return true;  // GA proposes a fixed candidate; hard-constraint handling is via fitness
}

}  // namespace n4m::core::opt
