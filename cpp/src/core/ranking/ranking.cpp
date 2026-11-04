// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "ranking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/common/distributions.hpp"

namespace n4m::ranking {
namespace {
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr size_t width = 10;
double sigmoid(double x) {
    if (x >= 0)
        return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}
double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::acos(-1.0));
}
double cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}
size_t root(std::vector<size_t>& parent, size_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}
// Stable lower-tail inverse Mills ratio. Continued fraction avoids 0/0.
double mills(double x) {
    if (x >= -8)
        return phi(x) / cdf(x);
    const double a = -x;
    double r = 0;
    for (int k = 128; k >= 1; --k)
        r = static_cast<double>(k) / (a + r);
    return a + r;
}
// Exact two-player moment update; no multi-team factor graph is claimed.
bool bayesian_update(double& mi,
                     double& mj,
                     double& vi,
                     double& vj,
                     double outcome,
                     const n4m_ranking_options_v1_t& o) {
    vi += o.dynamics * o.dynamics;
    vj += o.dynamics * o.dynamics;
    const double c2 = vi + vj + 2 * o.beta * o.beta;
    const double c = std::sqrt(c2);
    const double delta = (mi - mj) / c;
    double v = 0, w = 0;
    if (o.method == N4M_RANKING_WENG_LIN_BRADLEY_TERRY) {
        const double p = sigmoid(delta);
        // Weng-Lin Algorithm 1, 1v1, gamma_i=sigma_i/c, kappa=1e-12.
        const double curvature = p * (1 - p);
        mi += vi / c * (outcome - p);
        mj -= vj / c * (outcome - p);
        vi *= std::max(1e-12, 1 - std::sqrt(vi) / c * vi / c2 * curvature);
        vj *= std::max(1e-12, 1 - std::sqrt(vj) / c * vj / c2 * curvature);
    } else {
        if (outcome == 0.5) {
            const double a = (-o.draw_margin / c) - delta;
            const double b = (o.draw_margin / c) - delta;
            // Evaluate tail probabilities on the same side without subtracting
            // nearly equal values near one.
            const double z = a > 0 ? cdf(-a) - cdf(-b) : cdf(b) - cdf(a);
            if (!(z > 1e-280))
                return false;
            v = (phi(a) - phi(b)) / z;
            w = v * v - (a * phi(a) - b * phi(b)) / z;
        } else {
            const double sign = outcome == 1 ? 1.0 : -1.0;
            const double t = sign * delta - o.draw_margin / c;
            const double m = mills(t);
            v = sign * m;
            w = m * (m + t);
        }
        if (!std::isfinite(v) || !std::isfinite(w) || w < -1e-8 || w > 1 + 1e-8)
            return false;
        w = std::clamp(w, 0.0, 1.0);
        mi += vi / c * v;
        mj -= vj / c * v;
        vi *= std::max(1e-12, 1 - vi / c2 * w);
        vj *= std::max(1e-12, 1 - vj / c2 * w);
    }
    return std::isfinite(mi) && std::isfinite(mj) && std::isfinite(vi) && std::isfinite(vj);
}
}  // namespace

n4m_status_t compute(const n4m_matrix_view_t& view,
                     const uint8_t* mask,
                     const n4m_ranking_options_v1_t& o,
                     Result& out) {
    if (o.struct_size != sizeof(o))
        return N4M_ERR_ABI_MISMATCH;
    if (o.method < 0 || o.method > 4)
        return N4M_ERR_UNSUPPORTED;
    if (o.higher_is_better != 0 && o.higher_is_better != 1)
        return N4M_ERR_INVALID_ARGUMENT;
    const double opts[] = {o.tie_tolerance,
                           o.initial_mean,
                           o.k_factor,
                           o.elo_scale,
                           o.prior_sigma,
                           o.beta,
                           o.dynamics,
                           o.draw_margin,
                           o.regularization,
                           o.convergence_tolerance};
    for (double x : opts)
        if (!std::isfinite(x))
            return N4M_ERR_INVALID_ARGUMENT;
    if (o.tie_tolerance < 0 || o.k_factor <= 0 || o.elo_scale <= 0 || o.prior_sigma <= 0
        || o.beta <= 0 || o.dynamics < 0 || o.draw_margin <= 0 || o.regularization <= 0
        || o.convergence_tolerance <= 0 || o.max_iterations <= 0 || o.max_iterations > 1000000)
        return N4M_ERR_INVALID_ARGUMENT;
    // Bound parameter arithmetic; metric observations remain any finite F64.
    for (size_t i = 1; i < sizeof(opts) / sizeof(opts[0]); ++i)
        if (std::abs(opts[i]) > 1e100 || (opts[i] != 0 && std::abs(opts[i]) < 1e-100))
            return N4M_ERR_UNSUPPORTED;
    const size_t n = static_cast<size_t>(view.cols), blocks = static_cast<size_t>(view.rows);
    const auto* x = static_cast<const double*>(view.data);
    auto value = [&](size_t b, size_t j) {
        return x[static_cast<int64_t>(b) * view.row_stride
                 + static_cast<int64_t>(j) * view.col_stride];
    };
    uint64_t estimated_pairs = 0;
    for (size_t b = 0; b < blocks; ++b) {
        uint64_t active = 0;
        for (size_t j = 0; j < n; ++j)
            if (mask[b * n + j] == 1)
                ++active;
        estimated_pairs += active * (active - (active > 0 ? 1 : 0)) / 2;
        if (estimated_pairs > 100000000)
            return N4M_ERR_UNSUPPORTED;
    }
    // Complete validation before any update. Native ABI never infers missingness.
    for (size_t b = 0; b < blocks; ++b)
        for (size_t j = 0; j < n; ++j) {
            if (mask[b * n + j] > 1)
                return N4M_ERR_INVALID_ARGUMENT;
            if (mask[b * n + j] && !std::isfinite(value(b, j)))
                return N4M_ERR_INVALID_ARGUMENT;
        }
    out.candidates.assign(n * width, 0);
    auto cell = [&](size_t j, size_t k) -> double& {
        return out.candidates[j * width + k];
    };
    std::vector<double> mu(n, o.initial_mean), variance(n, o.prior_sigma * o.prior_sigma);
    std::vector<double> complete_rank_sum(n, 0);
    std::vector<size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    // Aggregate only batch BT; online/average methods avoid quadratic storage.
    std::vector<double> wins, counts;
    if (o.method == N4M_RANKING_BRADLEY_TERRY) {
        wins.assign(n * n, 0);
        counts.assign(n * n, 0);
    }
    double complete = 0, pair_count = 0, tie_sum = 0;
    bool complete_has_rank_difference = false;
    for (size_t b = 0; b < blocks; ++b) {
        std::vector<size_t> active;
        for (size_t j = 0; j < n; ++j)
            if (mask[b * n + j]) {
                active.push_back(j);
                cell(j, 3)++;
            }
        std::sort(active.begin(), active.end(), [&](size_t i, size_t j) {
            return o.higher_is_better ? value(b, i) > value(b, j) : value(b, i) < value(b, j);
        });
        const bool full = active.size() == n;
        if (full)
            complete++;
        // Tolerance ties use bounded-range groups anchored at the best score;
        // this is deterministic even when tolerance closeness is not transitive.
        for (size_t first = 0; first < active.size();) {
            size_t last = first + 1;
            while (last < active.size()
                   && std::abs(value(b, active[last]) - value(b, active[first])) <= o.tie_tolerance)
                ++last;
            const double rank = 0.5 * static_cast<double>(first + 1 + last);
            for (size_t k = first; k < last; ++k) {
                cell(active[k], 2) += rank;
                if (full)
                    complete_rank_sum[active[k]] += rank;
            }
            if (full) {
                if (last - first < n)
                    complete_has_rank_difference = true;
                const double t = static_cast<double>(last - first);
                tie_sum += t * t * t - t;
            }
            first = last;
        }
        // Native online update order is row-major candidate indices, not score order.
        for (size_t i = 0; i < n; ++i)
            if (mask[b * n + i])
                for (size_t j = i + 1; j < n; ++j)
                    if (mask[b * n + j]) {
                        const double a = value(b, i), z = value(b, j);
                        const double outcome =
                            std::abs(a - z) <= o.tie_tolerance
                                ? 0.5
                                : ((o.higher_is_better ? a > z : a < z) ? 1.0 : 0.0);
                        pair_count++;
                        cell(i, 4)++;
                        cell(j, 4)++;
                        if (outcome == 0.5) {
                            cell(i, 7)++;
                            cell(j, 7)++;
                        } else if (outcome == 1) {
                            cell(i, 6)++;
                            cell(j, 8)++;
                        } else {
                            cell(i, 8)++;
                            cell(j, 6)++;
                        }
                        parent[root(parent, j)] = root(parent, i);
                        if (o.method == N4M_RANKING_ELO) {
                            const double p =
                                sigmoid((mu[i] - mu[j]) / o.elo_scale * std::log(10.0));
                            const double change = o.k_factor * (outcome - p);
                            mu[i] += change;
                            mu[j] -= change;
                        } else if (o.method == N4M_RANKING_BRADLEY_TERRY) {
                            wins[i * n + j] += outcome;
                            counts[i * n + j]++;
                        } else if (o.method >= N4M_RANKING_TRUESKILL_1V1) {
                            if (!bayesian_update(
                                    mu[i], mu[j], variance[i], variance[j], outcome, o))
                                return N4M_ERR_NUMERICAL_FAILURE;
                        }
                    }
    }
    bool converged = true;
    int64_t iterations = 1;
    if (o.method == N4M_RANKING_BRADLEY_TERRY) {
        std::fill(mu.begin(), mu.end(), 0);
        std::vector<double> gradient(n);
        // Global Lipschitz bound on negative-log-likelihood Hessian row sums.
        double max_matches = 0;
        for (size_t i = 0; i < n; ++i)
            max_matches = std::max(max_matches, cell(i, 4));
        const double step = 1 / (o.regularization + 0.5 * max_matches);
        converged = false;
        for (iterations = 0; iterations < o.max_iterations; ++iterations) {
            for (size_t i = 0; i < n; ++i)
                gradient[i] = -o.regularization * mu[i];
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j) {
                    const double residual =
                        wins[i * n + j] - counts[i * n + j] * sigmoid(mu[i] - mu[j]);
                    gradient[i] += residual;
                    gradient[j] -= residual;
                }
            double norm = 0;
            for (double g : gradient)
                norm = std::max(norm, std::abs(g));
            if (norm <= o.convergence_tolerance) {
                converged = true;
                break;
            }
            for (size_t i = 0; i < n; ++i)
                mu[i] += step * gradient[i];
        }
    }
    std::vector<size_t> components(n, n);
    size_t n_components = 0;
    for (size_t j = 0; j < n; ++j) {
        const size_t r = root(parent, j);
        if (components[r] == n)
            components[r] = n_components++;
        cell(j, 5) = static_cast<double>(components[r]);
        cell(j, 1) = NaN;
        cell(j, 9) = NaN;
        if (cell(j, 3) == 0) {
            cell(j, 0) = NaN;
            cell(j, 2) = NaN;
            continue;
        }
        cell(j, 2) /= cell(j, 3);
        cell(j, 0) = o.method == N4M_RANKING_AVERAGE_RANK ? cell(j, 2) : mu[j];
        if (!std::isfinite(cell(j, 0)))
            return N4M_ERR_NUMERICAL_FAILURE;
        if (o.method >= N4M_RANKING_TRUESKILL_1V1)
            cell(j, 1) = std::sqrt(variance[j]);
    }
    // Across disconnected graphs, priors cannot identify a global ordering.
    if (n_components == 1 && converged) {
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
            return o.method == N4M_RANKING_AVERAGE_RANK ? cell(i, 0) < cell(j, 0)
                                                        : cell(i, 0) > cell(j, 0);
        });
        for (size_t first = 0; first < n;) {
            size_t last = first + 1;
            while (last < n && cell(order[first], 0) == cell(order[last], 0))
                ++last;
            for (size_t k = first; k < last; ++k)
                if (cell(order[k], 4) > 0)
                    cell(order[k], 9) = 0.5 * static_cast<double>(first + 1 + last);
            first = last;
        }
    }
    double friedman = NaN, df = NaN;
    if (n >= 2 && complete >= 2) {
        const double k = static_cast<double>(n);
        const double correction = 1 - tie_sum / (complete * (k * k * k - k));
        if (correction > 0 && complete_has_rank_difference) {
            // Centered rank sums avoid cancellation for almost tied data.
            double ss = 0;
            for (double sum : complete_rank_sum) {
                const double d = sum - complete * (k + 1) / 2;
                ss += d * d;
            }
            friedman = 12 * ss / (complete * k * (k + 1) * correction);
            df = k - 1;
        }
    }
    out.summary = {static_cast<double>(blocks),
                   complete,
                   pair_count,
                   static_cast<double>(n_components),
                   friedman,
                   df,
                   converged ? 1.0 : 0.0,
                   static_cast<double>(iterations),
                   std::isfinite(friedman)
                       ? n4m::core::distributions::regularized_gamma_q(df / 2, friedman / 2)
                       : NaN};
    return N4M_OK;
}

n4m_status_t relative_gain(const double* baseline,
                           const double* candidate,
                           const uint8_t* present,
                           const double* variances,
                           int64_t n,
                           std::array<double, 8>& out) {
    std::vector<double> effects, vars;
    for (int64_t i = 0; i < n; ++i) {
        if (present[i] > 1)
            return N4M_ERR_INVALID_ARGUMENT;
        if (!present[i])
            continue;
        if (!std::isfinite(baseline[i]) || !std::isfinite(candidate[i]) || baseline[i] <= 0
            || candidate[i] <= 0)
            return N4M_ERR_INVALID_ARGUMENT;
        const double effect = (baseline[i] - candidate[i]) / baseline[i];
        if (!std::isfinite(effect))
            return N4M_ERR_UNSUPPORTED;
        effects.push_back(effect);
        if (variances) {
            if (!std::isfinite(variances[i]) || variances[i] <= 0)
                return N4M_ERR_INVALID_ARGUMENT;
            vars.push_back(variances[i]);
        }
    }
    if (effects.empty())
        return N4M_ERR_INVALID_ARGUMENT;
    const double k = static_cast<double>(effects.size());
    double mean = 0, m2 = 0, count = 0;
    for (double e : effects) {
        count++;
        const double d = e - mean;
        mean += d / count;
        m2 += d * (e - mean);
    }
    out = {k, mean, k > 1 ? std::sqrt(std::max(0.0, m2 / (k - 1))) : NaN, NaN, NaN, NaN, NaN, NaN};
    if (variances && k >= 2) {
        // Scale inverse-variance weights to keep finite positive variances from
        // overflowing merely because a caller uses small effect units.
        const double vmin = *std::min_element(vars.begin(), vars.end());
        double sw = 0, sw2 = 0, fixed = 0;
        for (size_t i = 0; i < effects.size(); ++i) {
            const double w = vmin / vars[i];
            sw += w;
            sw2 += w * w;
            fixed += w * effects[i];
        }
        fixed /= sw;
        double qscaled = 0;
        for (size_t i = 0; i < effects.size(); ++i) {
            const double d = effects[i] - fixed;
            qscaled += vmin / vars[i] * d * d;
        }
        const double denom = sw - sw2 / sw;
        if (!(denom > 0))
            return N4M_ERR_NUMERICAL_FAILURE;
        const double tau2 = std::max(0.0, (qscaled - (k - 1) * vmin) / denom);
        const double scale = vmin + tau2;
        double rw = 0, pooled = 0;
        for (size_t i = 0; i < effects.size(); ++i) {
            const double w = scale / (vars[i] + tau2);
            rw += w;
            pooled += w * effects[i];
        }
        const double q = qscaled / vmin;
        out[3] = pooled / rw;
        out[4] = std::sqrt(scale / rw);
        out[5] = tau2;
        out[6] = q;
        out[7] = q > 0 ? std::max(0.0, (q - (k - 1)) / q) : 0;
    }
    for (double z : out)
        if (std::isinf(z))
            return N4M_ERR_UNSUPPORTED;
    if (!std::isfinite(out[1]) || (k > 1 && !std::isfinite(out[2])))
        return N4M_ERR_UNSUPPORTED;
    if (variances && k >= 2)
        for (size_t i = 3; i < out.size(); ++i)
            if (!std::isfinite(out[i]))
                return N4M_ERR_NUMERICAL_FAILURE;
    return N4M_OK;
}
}  // namespace n4m::ranking
