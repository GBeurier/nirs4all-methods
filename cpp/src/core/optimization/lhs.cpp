// SPDX-License-Identifier: CECILL-2.1
//
// LhsSampler (F1) — Latin Hypercube sampling over the numeric axes. For the
// first `n_startup_trials` asks it hands out a stratified + jittered design
// (one independent permutation per dimension, so each 1-D projection is evenly
// covered); beyond the startup batch, and for categorical/ordinal/sorted-tuple
// axes, it falls back to the base uniform sampler. The design is precomputed
// from a seed, so it is deterministic and idempotent within one ask.

#include "core/optimization/optimizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "core/common/rng_engine.h"

namespace n4m::core::opt {

namespace {
bool is_numeric_kind(n4m_param_kind_t k) {
    return k == N4M_PARAM_INT || k == N4M_PARAM_FLOAT || k == N4M_PARAM_LOG_INT ||
           k == N4M_PARAM_LOG_FLOAT;
}
}  // namespace

LhsSampler::LhsSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    n_startup_ = opts.n_startup_trials > 0 ? opts.n_startup_trials : 0;
    for (const auto& p : space.params) {
        if (is_numeric_kind(p.kind)) numeric_names_.push_back(p.name);
    }
    if (n_startup_ <= 0 || numeric_names_.empty()) return;

    n4m_rng lrng;
    n4m_rng_seed(&lrng, N4M_RNGK_SPLITMIX64, opts.seed);
    const int N = n_startup_;
    unit_.assign(numeric_names_.size(), std::vector<double>(static_cast<std::size_t>(N), 0.0));
    for (auto& axis : unit_) {
        std::vector<int> perm(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) perm[static_cast<std::size_t>(i)] = i;
        for (int i = N - 1; i >= 1; --i) {  // Fisher-Yates
            const int j = static_cast<int>(n4m_rng_next_double(&lrng) * (i + 1));
            std::swap(perm[static_cast<std::size_t>(i)],
                      perm[static_cast<std::size_t>(j < i ? j : i)]);
        }
        for (int i = 0; i < N; ++i) {
            const double jitter = n4m_rng_next_double(&lrng);
            axis[static_cast<std::size_t>(i)] =
                (static_cast<double>(perm[static_cast<std::size_t>(i)]) + jitter) / N;
        }
    }
}

bool LhsSampler::override_numeric(const ParamSpec& p, double* out) {
    if (unit_.empty()) return false;
    const std::size_t i = trials().size();  // this ask's index (current trial not yet stored)
    if (static_cast<std::int32_t>(i) >= n_startup_) return false;
    std::size_t axis = 0;
    bool found = false;
    for (std::size_t a = 0; a < numeric_names_.size(); ++a) {
        if (numeric_names_[a] == p.name) { axis = a; found = true; break; }
    }
    if (!found) return false;
    *out = numeric_from_unit(p, unit_[axis][i]);
    return true;
}

}  // namespace n4m::core::opt
