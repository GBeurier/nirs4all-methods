// SPDX-License-Identifier: CECILL-2.1
// Chi-square and F distribution functions for exact confidence intervals.
//
// The regularized incomplete gamma uses its power series below a+1 and its
// Legendre continued fraction above (DLMF 8.7.1 and 8.9.2); the regularized
// incomplete beta uses its continued fraction (DLMF 8.17.22) on the side
// where it converges, with I_x(a,b) = 1 - I_{1-x}(b,a). Both continued
// fractions are evaluated with the modified Lentz method. Quantiles invert
// the monotone CDF by bisection to the resolution of a double.
#ifndef N4M_CORE_COMMON_DISTRIBUTIONS_HPP
#define N4M_CORE_COMMON_DISTRIBUTIONS_HPP

namespace n4m::core::distributions {

// Lower regularized incomplete gamma P(a, x), a > 0, x >= 0.
double regularized_gamma_p(double a, double x) noexcept;
// Regularized incomplete beta I_x(a, b), a > 0, b > 0, 0 <= x <= 1.
double regularized_beta(double a, double b, double x) noexcept;

// Chi-square CDF and quantile, df > 0. The quantile takes 0 < p < 1.
double chi2_cdf(double x, double df) noexcept;
double chi2_quantile(double p, double df) noexcept;

// Fisher-Snedecor F CDF and quantile, d1 > 0, d2 > 0, 0 < p < 1.
double f_cdf(double x, double d1, double d2) noexcept;
double f_quantile(double p, double d1, double d2) noexcept;

}  // namespace n4m::core::distributions

#endif  // N4M_CORE_COMMON_DISTRIBUTIONS_HPP
