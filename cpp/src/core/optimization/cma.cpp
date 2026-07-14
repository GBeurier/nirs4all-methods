// SPDX-License-Identifier: CECILL-2.1
//
// CmaEsSampler (F4) — separable (diagonal) CMA-ES over the CONTINUOUS axes of the
// search space (Ros & Hansen 2008). The distribution (mean, diagonal covariance,
// step-size, evolution paths) lives only over the numeric axes; categorical /
// ordinal / sorted-tuple / conditionally-inactive axes are sampled independently
// by the shared decode (Optuna's independent-fallback for mixed spaces). Updates
// use only completed, scored trials, so pruned/failed members never enter the
// mean/covariance. No eigendecomposition — cheap and robust for modest
// continuous dimensionality.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace n4m::core::opt {

namespace {
double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v >= 1.0) return std::nextafter(1.0, 0.0);
    return v;
}
bool is_continuous(n4m_param_kind_t k) {
    return k == N4M_PARAM_INT || k == N4M_PARAM_FLOAT || k == N4M_PARAM_LOG_INT ||
           k == N4M_PARAM_LOG_FLOAT;
}
}  // namespace

CmaEsSampler::CmaEsSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    (void)opts;
    for (std::size_t j = 0; j < space.params.size(); ++j) {
        if (is_continuous(space.params[j].kind)) cont_axes_.push_back(j);
    }
    P_ = cont_axes_.size();
    const double P = static_cast<double>(P_ > 0 ? P_ : 1);

    lambda_ = 4 + static_cast<std::int32_t>(std::floor(3.0 * std::log(P)));
    if (lambda_ < 4) lambda_ = 4;
    mu_ = lambda_ / 2;
    if (mu_ < 1) mu_ = 1;

    weights_.assign(static_cast<std::size_t>(mu_), 0.0);
    double sum = 0.0;
    for (std::int32_t i = 0; i < mu_; ++i) {
        weights_[static_cast<std::size_t>(i)] = std::log(mu_ + 0.5) - std::log(i + 1.0);
        sum += weights_[static_cast<std::size_t>(i)];
    }
    double sumsq = 0.0;
    for (auto& w : weights_) {
        w /= sum;
        sumsq += w * w;
    }
    mu_eff_ = 1.0 / sumsq;

    c_sigma_ = (mu_eff_ + 2.0) / (P + mu_eff_ + 5.0);
    d_sigma_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff_ - 1.0) / (P + 1.0)) - 1.0) + c_sigma_;
    c_c_ = (4.0 + mu_eff_ / P) / (P + 4.0 + 2.0 * mu_eff_ / P);
    c_1_ = 2.0 / ((P + 1.3) * (P + 1.3) + mu_eff_);
    c_mu_ = std::min(1.0 - c_1_,
                     2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) / ((P + 2.0) * (P + 2.0) + mu_eff_));
    const double sep = (P + 2.0) / 3.0;  // sep-CMA-ES learning-rate rescaling
    c_1_ = std::min(1.0, c_1_ * sep);
    c_mu_ = std::min(1.0 - c_1_, c_mu_ * sep);
    chi_n_ = std::sqrt(P) * (1.0 - 1.0 / (4.0 * P) + 1.0 / (21.0 * P * P));

    m_.assign(P_, 0.5);
    C_.assign(P_, 1.0);
    ps_.assign(P_, 0.0);
    pc_.assign(P_, 0.0);
    sigma_ = 0.2;
}

void CmaEsSampler::sample_population() {
    pop_x_.assign(static_cast<std::size_t>(lambda_), std::vector<double>(P_, 0.0));
    for (std::int32_t k = 0; k < lambda_; ++k) {
        for (std::size_t j = 0; j < P_; ++j) {
            const double z = n4m_rng_next_normal(&rng_);
            pop_x_[static_cast<std::size_t>(k)][j] = clamp01(m_[j] + sigma_ * std::sqrt(C_[j]) * z);
        }
    }
}

void CmaEsSampler::ensure_generation(std::int64_t member_base) {
    const bool first = gen_base_id_ < 0;
    const bool next = !first && member_base >= gen_base_id_ + lambda_;
    if (!first && !next) return;

    if (first) {
        sample_population();
        gen_base_id_ = member_base;
        return;
    }

    // Rank only the COMPLETED, scored members of the finished generation.
    std::vector<std::pair<double, int>> scored;
    scored.reserve(static_cast<std::size_t>(lambda_));
    for (std::int32_t k = 0; k < lambda_; ++k) {
        const std::int64_t tid = gen_base_id_ + k;
        for (const auto& up : trials()) {
            if (up->id == tid) {
                if (up->status == N4M_TRIAL_COMPLETED && up->has_score) {
                    scored.emplace_back(up->score, k);
                }
                break;
            }
        }
    }
    if (P_ == 0 || scored.empty()) {  // nothing to learn from; just resample
        sample_population();
        ++gen_count_;
        gen_base_id_ = member_base;
        return;
    }
    const bool maximize = direction() == N4M_OPT_MAXIMIZE;
    std::sort(scored.begin(), scored.end(), [maximize](const auto& a, const auto& b) {
        return maximize ? a.first > b.first : a.first < b.first;
    });
    const int n_use = std::min<int>(mu_, static_cast<int>(scored.size()));
    double wsum = 0.0;
    for (int i = 0; i < n_use; ++i) wsum += weights_[static_cast<std::size_t>(i)];

    // Weighted mean of the best n_use (weights renormalised over n_use).
    std::vector<double> m_new(P_, 0.0);
    for (int i = 0; i < n_use; ++i) {
        const int k = scored[static_cast<std::size_t>(i)].second;
        const double w = weights_[static_cast<std::size_t>(i)] / wsum;
        for (std::size_t j = 0; j < P_; ++j) m_new[j] += w * pop_x_[static_cast<std::size_t>(k)][j];
    }
    std::vector<double> yw(P_, 0.0);
    for (std::size_t j = 0; j < P_; ++j) yw[j] = (m_new[j] - m_[j]) / sigma_;

    // Step-size path from the REPAIRED (post-clip) step: C^{-1/2}·yw = yw/sqrt(C).
    const double cs_disc = std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_);
    double ps_norm_sq = 0.0;
    for (std::size_t j = 0; j < P_; ++j) {
        ps_[j] = (1.0 - c_sigma_) * ps_[j] + cs_disc * (yw[j] / std::sqrt(C_[j]));
        ps_norm_sq += ps_[j] * ps_[j];
    }
    const double ps_norm = std::sqrt(ps_norm_sq);

    const double gen = static_cast<double>(gen_count_ + 1);
    const double hsig_denom = std::sqrt(1.0 - std::pow(1.0 - c_sigma_, 2.0 * gen));
    const double hsig =
        (ps_norm / hsig_denom < (1.4 + 2.0 / (static_cast<double>(P_) + 1.0)) * chi_n_) ? 1.0 : 0.0;

    const double cc_disc = std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_);
    for (std::size_t j = 0; j < P_; ++j) pc_[j] = (1.0 - c_c_) * pc_[j] + hsig * cc_disc * yw[j];

    const double c_old = 1.0 - c_1_ - c_mu_ + (1.0 - hsig) * c_1_ * c_c_ * (2.0 - c_c_);
    for (std::size_t j = 0; j < P_; ++j) {
        double rank_mu = 0.0;
        for (int i = 0; i < n_use; ++i) {
            const int k = scored[static_cast<std::size_t>(i)].second;
            const double w = weights_[static_cast<std::size_t>(i)] / wsum;
            const double yij = (pop_x_[static_cast<std::size_t>(k)][j] - m_[j]) / sigma_;
            rank_mu += w * yij * yij;
        }
        C_[j] = c_old * C_[j] + c_1_ * pc_[j] * pc_[j] + c_mu_ * rank_mu;
        if (C_[j] < 1e-20) C_[j] = 1e-20;
    }

    sigma_ *= std::exp((c_sigma_ / d_sigma_) * (ps_norm / chi_n_ - 1.0));
    if (sigma_ > 10.0) sigma_ = 10.0;
    if (sigma_ < 1e-10) sigma_ = 1e-10;

    m_ = std::move(m_new);
    ++gen_count_;
    sample_population();
    gen_base_id_ = member_base;
}

bool CmaEsSampler::sample(::n4m_trial_s& t,
                          const std::vector<std::pair<std::string, double>>* forced) {
    if (gen_base_id_ >= 0 && t.id >= gen_base_id_ + lambda_ &&
        resolved_in_range(gen_base_id_, lambda_) < lambda_) {
        return false;  // synchronous: complete the current generation first
    }
    ensure_generation(t.id);
    std::size_t k = static_cast<std::size_t>(t.id - gen_base_id_);
    if (k >= static_cast<std::size_t>(lambda_)) k = static_cast<std::size_t>(lambda_) - 1;

    // Build the full unit vector: continuous axes from the CMA position, all other
    // axes drawn independently (bucketed/decoded by decode_candidate).
    std::vector<double> full_u(space_.params.size(), 0.0);
    for (std::size_t j = 0; j < space_.params.size(); ++j) full_u[j] = n4m_rng_next_double(&rng_);
    for (std::size_t a = 0; a < cont_axes_.size(); ++a) full_u[cont_axes_[a]] = pop_x_[k][a];
    decode_candidate(full_u, t, forced);
    return true;
}

bool CmaEsSampler::at_generation_boundary(std::int64_t prospective_id) const {
    // Mirror the sample() guard: the current CMA generation is fully dispensed
    // but not yet fully scored, so the next ask cannot advance.
    return gen_base_id_ >= 0 && prospective_id >= gen_base_id_ + lambda_ &&
           resolved_in_range(gen_base_id_, lambda_) < lambda_;
}

}  // namespace n4m::core::opt
