// SPDX-License-Identifier: CECILL-2.1
// Chi-square and F distribution functions for exact confidence intervals.
//
// The regularized incomplete gamma uses its power series below a+1 and its
// Legendre continued fraction above (DLMF 8.7.1, 8.9.2); the regularized
// incomplete beta uses its continued fraction (DLMF 8.17.22) on the side
// where it converges. Both continued fractions use the modified Lentz method.
// The prefactors x^a e^-x / Gamma(a) and x^a (1-x)^b / B(a,b) are formed
// without cancellation for large parameters: with phi(t) = t - 1 - log t
// (series near t = 1) and the Stirling remainder of log Gamma (a >= 15),
//   log[x^a e^-x / Gamma(a)] = -a phi(x/a) + log(a/(2 pi))/2 - delta(a),
// and the analogous form around x0 = a/(a+b) for the beta (DLMF 5.11.1;
// the bd0 device of Loader 2000). Lower and upper tails are both evaluated
// directly, never as 1 minus a value close to 1.
//
// Quantiles bisect a monotone tail on log x (chi-square) or on the logit of
// the beta variable (F = (d2/d1) e^t), choosing the lower tail for p <= 1/2
// and the upper tail otherwise, to a relative resolution of a few ulps.
//
// Domain: 0 < p < 1; degrees of freedom in [1/2, 2^31]; a chi-square
// quantile in [1e-300, 1e300] and an F quantile in [e^-700, e^700] times
// d2/d1. Outside it the quantile is NaN; callers turn NaN into a status.
#ifndef N4M_CORE_COMMON_DISTRIBUTIONS_HPP
#define N4M_CORE_COMMON_DISTRIBUTIONS_HPP

namespace n4m::core::distributions {

// Lower and upper regularized incomplete gamma P(a, x), Q(a, x).
double regularized_gamma_p(double a, double x) noexcept;
double regularized_gamma_q(double a, double x) noexcept;
// Regularized incomplete beta I_x(a, b), given x and y = 1 - x computed
// independently by the caller (so either may be tiny).
double regularized_beta(double a, double b, double x, double y) noexcept;

// Chi-square and F quantiles, 0 < p < 1 (NaN outside the domain above).
double chi2_quantile(double p, double df) noexcept;
double f_quantile(double p, double d1, double d2) noexcept;

}  // namespace n4m::core::distributions

#endif  // N4M_CORE_COMMON_DISTRIBUTIONS_HPP
