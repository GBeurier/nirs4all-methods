// SPDX-License-Identifier: CECILL-2.1
#include "core/common/distributions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace n4m::core::distributions {
namespace {
constexpr double tiny = 1e-300;
constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr int max_terms = 1000000;

// Power series: P(a,x) = x^a e^-x / Gamma(a+1) * sum_k x^k / ((a+1)...(a+k)).
double gamma_series(double a, double x) noexcept {
    double term = 1.0 / a, sum = term;
    for (int k = 1; k < max_terms; ++k) {
        term *= x / (a + k);
        sum += term;
        if (std::abs(term) < std::abs(sum) * eps) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Continued fraction for Q(a,x) = 1 - P(a,x), x >= a+1.
double gamma_continued_fraction(double a, double x) noexcept {
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int k = 1; k < max_terms; ++k) {
        const double an = -k * (k - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < eps) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

// Continued fraction of I_x(a,b) without its prefactor, x < (a+1)/(a+b+2).
double beta_continued_fraction(double a, double b, double x) noexcept {
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::abs(d) < tiny) d = tiny;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m < max_terms; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < eps) break;
    }
    return h;
}

// Smallest x in [lo, hi] with cdf(x) >= p, for a nondecreasing cdf.
template <typename Cdf>
double invert(Cdf cdf, double p, double lo, double hi) noexcept {
    for (int i = 0; i < 2000 && hi - lo > 2.0 * eps * hi; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        if (mid <= lo || mid >= hi) break;
        if (cdf(mid) < p) lo = mid; else hi = mid;
    }
    return lo + 0.5 * (hi - lo);
}
}  // namespace

double regularized_gamma_p(double a, double x) noexcept {
    if (!(a > 0.0) || !(x >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return 0.0;
    if (x < a + 1.0) return gamma_series(a, x);
    return 1.0 - gamma_continued_fraction(a, x);
}

double regularized_beta(double a, double b, double x) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || !(x >= 0.0) || !(x <= 1.0))
        return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    const double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                                  a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) return front * beta_continued_fraction(a, b, x) / a;
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b;
}

double chi2_cdf(double x, double df) noexcept {
    return x <= 0.0 ? 0.0 : regularized_gamma_p(0.5 * df, 0.5 * x);
}

double chi2_quantile(double p, double df) noexcept {
    if (!(p > 0.0 && p < 1.0) || !(df > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    double hi = std::max(1.0, df);
    while (chi2_cdf(hi, df) < p && hi < 1e300) hi *= 2.0;
    return invert([df](double x) { return chi2_cdf(x, df); }, p, 0.0, hi);
}

double f_cdf(double x, double d1, double d2) noexcept {
    if (x <= 0.0) return 0.0;
    return regularized_beta(0.5 * d1, 0.5 * d2, d1 * x / (d1 * x + d2));
}

double f_quantile(double p, double d1, double d2) noexcept {
    if (!(p > 0.0 && p < 1.0) || !(d1 > 0.0) || !(d2 > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    // Invert I_y(d1/2, d2/2) = p for y in (0,1), then F = d2 y / (d1 (1-y)).
    const double y = invert(
        [d1, d2](double v) { return regularized_beta(0.5 * d1, 0.5 * d2, v); }, p, 0.0, 1.0);
    return d2 * y / (d1 * (1.0 - y));
}

}  // namespace n4m::core::distributions
