// SPDX-License-Identifier: CECILL-2.1
// One-way random-effects variance components x_ij = mu + u_i + e_ij with
// u_i ~ N(0, s2b) and e_ij ~ N(0, s2w), estimated by REML for balanced or
// unbalanced groups.
//
// With gamma = s2b/s2w, a_i = gamma + 1/r_i and w_i = 1/a_i, s2w profiles
// out of the restricted likelihood as (SSW + Q(gamma))/(N-1), where SSW is
// the pooled within-group sum of squares and
// Q(gamma) = sum_i w_i (ybar_i - mu_hat)^2 with mu_hat the w-weighted mean.
// The profiled criterion, -2 restricted log-likelihood up to constants,
//   f(gamma) = (N-1) log((SSW+Q)/(N-1)) + sum_i log a_i + log sum_i w_i,
// has derivative
//   f'(gamma) = -(N-1) sum_i w_i^2 e_i^2/(SSW+Q) + sum_i w_i - sum_i w_i^2/sum_i w_i.
// f is minimised over gamma >= 0 by a grid on gamma = 0 and log10(gamma) in
// [-8, log10(1e3 max(1, F))] with step 1/4, F the one-way ANOVA ratio, then
// by bisection on f' inside the cell next to the best grid point. The
// boundary gamma = 0 is retained when it is the best grid point and
// f'(0) >= 0. Groups enter only through (r_i, ybar_i, within SS_i), so a
// cluster-bootstrap replicate is a multiplicity vector over the groups.
#ifndef N4M_CORE_VARIANCE_COMPONENTS_HPP
#define N4M_CORE_VARIANCE_COMPONENTS_HPP

#include <cstdint>
#include <vector>

namespace n4m::core::variance_components {

// Band status codes shared with the public C ABI.
enum Status : int32_t {
    interior = 0,
    insufficient = 1,      // fewer than two groups, or no within-group df
    between_boundary = 2,  // REML optimum at s2b = 0 (ICC = 0)
    zero_total = 3,        // no variation at all: ICC undefined
    within_boundary = 4,   // SSW = 0 with distinct means: s2w = 0, ICC = 1
    not_converged = 5,     // optimum at the search bound or no sign change
};

// One group observed in one feature, on a scaled value axis.
struct Group {
    int64_t r = 0;
    long double mean = 0;
    long double m2 = 0;  // within-group sum of squared deviations
};

struct Fit {
    int32_t status = insufficient;
    double sigma2_between = 0;  // scaled units
    double sigma2_within = 0;   // scaled units
    double icc = 0;
    double loglik = 0;  // restricted log-likelihood, scaled units, NaN at s2w = 0
    int64_t n_groups = 0;
    int64_t n_obs = 0;
    int32_t local_minima = 0;
    int32_t iterations = 0;
};

// Groups must be sorted by r. multiplicity is null (each group once) or a
// count per group, as drawn by a cluster bootstrap.
Fit fit_reml(const std::vector<Group>& groups, const int64_t* multiplicity);

}  // namespace n4m::core::variance_components

#endif  // N4M_CORE_VARIANCE_COMPONENTS_HPP
