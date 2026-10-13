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
// f may have several local minima for unbalanced groups. f and f' are
// evaluated on gamma = 0 and on log10(gamma) in [-8, log10(1000 max(1, F))]
// with step 1/8 (F the one-way ANOVA ratio). Every grid cell where f'
// changes sign from - to + is refined by bisection on f', the boundary
// gamma = 0 is a candidate when f'(0) >= 0, and the candidate with the
// smallest f is retained. A minimum narrower than one grid cell can still
// be missed; the count of candidates is reported.
//
// Inputs live on an exact power-of-two scale chosen by the caller from the
// centred variation of the feature (|centred value| < 1). Group means are on
// that scale; within-group sums of squares are carried as m2 * 2^(2 ew),
// so a within variation far below the between variation never underflows.
// When F > 2^64 (gamma beyond any double-resolvable ICC) the REML limit is
// used: s2w = SSW/(N-g), s2b = sum(ybar_i - ybar)^2/(g-1); its relative
// error is O(1/F), below double resolution.
#ifndef N4M_CORE_VARIANCE_COMPONENTS_HPP
#define N4M_CORE_VARIANCE_COMPONENTS_HPP

#include <cstdint>
#include <vector>

namespace n4m::core::variance_components {

// Band status codes shared with the public C ABI.
enum Status : int32_t {
    interior = 0,
    insufficient = 1,       // fewer than two groups, or no within-group df
    between_boundary = 2,   // REML optimum at s2b = 0 (ICC = 0)
    zero_total = 3,         // no variation at all: ICC undefined
    within_boundary = 4,    // SSW = 0 with distinct means: s2w = 0, ICC = 1
    not_converged = 5,      // minimum beyond the search bound or failed bisection
    not_representable = 6,  // a published value overflows or underflows a double
};

// One group observed in one feature: mean on the scaled axis, within SS in
// units of 2^(2 ew).
struct Group {
    int64_t r = 0;
    long double mean = 0;
    long double m2 = 0;
};

struct Fit {
    int32_t status = insufficient;
    long double sigma2_between = 0;  // scaled units
    long double sigma2_within = 0;   // scaled units times 2^within_shift
    int within_shift = 0;
    double icc = 0;
    long double loglik = 0;  // restricted log-likelihood, scaled units; NaN when undefined
    int64_t n_groups = 0;
    int64_t n_obs = 0;
    int32_t candidates = 0;
    int32_t iterations = 0;
};

// Groups must be sorted by r. multiplicity is null (each group once) or a
// count per group, as drawn by a cluster bootstrap. within_exponent is ew.
Fit fit_reml(const std::vector<Group>& groups, const int64_t* multiplicity,
             int within_exponent);

}  // namespace n4m::core::variance_components

#endif  // N4M_CORE_VARIANCE_COMPONENTS_HPP
