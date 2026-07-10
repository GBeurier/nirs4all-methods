// SPDX-License-Identifier: CECILL-2.1
//
// GpEiSampler (F4) — Bayesian optimization with a Gaussian-process surrogate and
// Expected Improvement acquisition. Each ask (after the random startup) fits an
// RBF GP on the completed, scored history over the continuous axes in unit space,
// then returns the candidate that maximises EI over a random acquisition batch.
// The lengthscale uses a median-pairwise-distance heuristic (no marginal-
// likelihood inner loop); the small dense Cholesky is cheap at NIRS trial counts.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace n4m::core::opt {

namespace {
bool is_continuous(n4m_param_kind_t k) {
    return k == N4M_PARAM_INT || k == N4M_PARAM_FLOAT || k == N4M_PARAM_LOG_INT ||
           k == N4M_PARAM_LOG_FLOAT;
}
double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v >= 1.0) return std::nextafter(1.0, 0.0);
    return v;
}
double sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}
// Standard normal pdf / cdf.
double norm_pdf(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI); }
double norm_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

// Cholesky K = L Lᵀ (K symmetric PD, lower L). Returns false on non-PD.
bool cholesky(const std::vector<std::vector<double>>& K, std::vector<std::vector<double>>& L) {
    const std::size_t n = K.size();
    L.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = K[i][j];
            for (std::size_t k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) {
                if (sum <= 0.0) return false;
                L[i][i] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return true;
}
// Solve K x = b given K = L Lᵀ (forward then back substitution).
std::vector<double> chol_solve(const std::vector<std::vector<double>>& L,
                               const std::vector<double>& b) {
    const std::size_t n = L.size();
    std::vector<double> z(n, 0.0), x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= L[i][k] * z[k];
        z[i] = s / L[i][i];
    }
    for (std::size_t ii = n; ii-- > 0;) {
        double s = z[ii];
        for (std::size_t k = ii + 1; k < n; ++k) s -= L[k][ii] * x[k];
        x[ii] = s / L[ii][ii];
    }
    return x;
}
}  // namespace

GpEiSampler::GpEiSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    for (std::size_t j = 0; j < space.params.size(); ++j) {
        if (is_continuous(space.params[j].kind)) cont_axes_.push_back(j);
    }
    P_ = cont_axes_.size();
    if (opts.n_startup_trials > 0) n_startup_ = opts.n_startup_trials;
}

bool GpEiSampler::gather(std::vector<std::vector<double>>& X, std::vector<double>& y) const {
    X.clear();
    y.clear();
    for (const auto& up : trials()) {
        if (up->status != N4M_TRIAL_COMPLETED || !up->has_score) continue;
        auto it = proposals_.find(up->id);
        if (it == proposals_.end()) continue;
        X.push_back(it->second);
        y.push_back(up->score);
    }
    return static_cast<std::int32_t>(X.size()) >= n_startup_ && X.size() >= 2;
}

double GpEiSampler::expected_improvement(double mu, double sigma, double y_best) const {
    if (sigma <= 1e-12) return 0.0;
    const double xi = 0.01;  // small exploration margin
    // Cast to a minimization improvement: for MAXIMIZE, negate so "smaller is better".
    const bool maximize = direction() == N4M_OPT_MAXIMIZE;
    const double m = maximize ? -mu : mu;
    const double best = maximize ? -y_best : y_best;
    const double imp = best - m - xi;
    const double z = imp / sigma;
    return imp * norm_cdf(z) + sigma * norm_pdf(z);
}

bool GpEiSampler::sample(::n4m_trial_s& t,
                         const std::vector<std::pair<std::string, double>>* forced) {
    std::vector<double> full_u(space_.params.size(), 0.0);
    for (std::size_t j = 0; j < space_.params.size(); ++j) full_u[j] = n4m_rng_next_double(&rng_);

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    if (P_ > 0 && gather(X, y)) {
        const std::size_t n = X.size();
        // Standardize y for a well-conditioned kernel.
        double ymean = 0.0;
        for (double v : y) ymean += v;
        ymean /= static_cast<double>(n);
        double yvar = 0.0;
        for (double v : y) yvar += (v - ymean) * (v - ymean);
        const double ystd = std::sqrt(yvar / static_cast<double>(n)) + 1e-12;
        std::vector<double> ys(n);
        for (std::size_t i = 0; i < n; ++i) ys[i] = (y[i] - ymean) / ystd;

        // Median pairwise squared distance → RBF lengthscale (median heuristic).
        std::vector<double> d2;
        d2.reserve(n * (n - 1) / 2);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) d2.push_back(sq_dist(X[i], X[j]));
        double med = 0.25;  // fallback when all points coincide
        if (!d2.empty()) {
            std::sort(d2.begin(), d2.end());
            med = d2[d2.size() / 2];
            if (med < 1e-9) med = 0.25;
        }
        const double ell2 = 0.5 * med;  // 2*ell^2 = med  =>  ell^2 = med/2
        auto rbf = [&](const std::vector<double>& a, const std::vector<double>& b) {
            return std::exp(-sq_dist(a, b) / (2.0 * ell2));
        };

        // K + jitter, Cholesky, alpha = K^{-1} ys.
        const double noise = 1e-6;
        std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                K[i][j] = rbf(X[i], X[j]) + (i == j ? noise : 0.0);
        std::vector<std::vector<double>> L;
        if (cholesky(K, L)) {
            const std::vector<double> alpha = chol_solve(L, ys);
            double ybest = y[0];
            const bool maximize = direction() == N4M_OPT_MAXIMIZE;
            for (double v : y) ybest = maximize ? std::max(ybest, v) : std::min(ybest, v);

            // Random acquisition batch over the continuous axes; keep the max-EI point.
            std::vector<double> best_cont;
            double best_ei = -1.0;
            for (std::int32_t c = 0; c < n_candidates_; ++c) {
                std::vector<double> cand(P_);
                for (std::size_t a = 0; a < P_; ++a) cand[a] = n4m_rng_next_double(&rng_);
                std::vector<double> kstar(n);
                for (std::size_t i = 0; i < n; ++i) kstar[i] = rbf(cand, X[i]);
                double mu_s = 0.0;
                for (std::size_t i = 0; i < n; ++i) mu_s += kstar[i] * alpha[i];
                // posterior variance: k** - kstarᵀ K^{-1} kstar
                const std::vector<double> v = chol_solve(L, kstar);
                double var_s = 1.0;  // rbf(x,x) = 1
                for (std::size_t i = 0; i < n; ++i) var_s -= kstar[i] * v[i];
                if (var_s < 0.0) var_s = 0.0;
                const double mu = ymean + ystd * mu_s;
                const double sigma = ystd * std::sqrt(var_s);
                const double ei = expected_improvement(mu, sigma, ybest);
                if (ei > best_ei) {
                    best_ei = ei;
                    best_cont = std::move(cand);
                }
            }
            if (!best_cont.empty())
                for (std::size_t a = 0; a < P_; ++a) full_u[cont_axes_[a]] = clamp01(best_cont[a]);
        }
    }

    decode_candidate(full_u, t, forced);
    // Record the continuous-axis coords so this trial can seed future GP fits.
    std::vector<double> cont(P_);
    for (std::size_t a = 0; a < P_; ++a) cont[a] = full_u[cont_axes_[a]];
    proposals_[t.id] = std::move(cont);
    return true;
}

}  // namespace n4m::core::opt
