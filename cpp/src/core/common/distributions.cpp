// SPDX-License-Identifier: CECILL-2.1
#include "core/common/distributions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace n4m::core::distributions {
namespace {
constexpr double tiny = 1e-300;
constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr double half_log_two_pi = 0.918938533204672741780329736406;
constexpr double stirling_threshold = 15.0;
constexpr int max_terms = 10000000;

// phi(t) = t - 1 - log t >= 0, accurate near t = 1 (series in u = t - 1).
double phi(double t) noexcept {
    const double u = t - 1.0;
    if (std::abs(u) >= 0.1) return u - std::log(t);
    // u - log1p(u) = sum_{k>=2} (-1)^k u^k / k.
    double term = u * u, sum = 0.0;
    for (int k = 2; k < 60; ++k) {
        const double add = (k % 2 == 0 ? term : -term) / k;
        sum += add;
        if (std::abs(add) <= eps * sum) break;
        term *= u;
    }
    return sum;
}

// Stirling remainder delta(a) = lgamma(a) - [(a-1/2) log a - a + log(2 pi)/2],
// a >= 15: the omitted term is below 2^-52 delta(a).
double stirling_delta(double a) noexcept {
    const double inv = 1.0 / a, inv2 = inv * inv;
    return inv * (1.0 / 12.0 -
                  inv2 * (1.0 / 360.0 -
                          inv2 * (1.0 / 1260.0 -
                                  inv2 * (1.0 / 1680.0 -
                                          inv2 * (1.0 / 1188.0 - inv2 * 691.0 / 360360.0)))));
}

// log of x^a e^-x / Gamma(a).
double log_gamma_prefactor(double a, double x) noexcept {
    if (a < stirling_threshold) return a * std::log(x) - x - std::lgamma(a);
    return -a * phi(x / a) + 0.5 * std::log(a) - half_log_two_pi - stirling_delta(a);
}

// lgamma(b + a) - lgamma(b) for b >= 15 without cancellation:
// (b - 1/2) log1p(a/b) + a log(b + a) - a + delta(b + a) - delta(b).
double log_gamma_ratio(double b, double a) noexcept {
    return (b - 0.5) * std::log1p(a / b) + a * std::log(b + a) - a + stirling_delta(b + a) -
           stirling_delta(b);
}

// log of x^a y^b / B(a, b), y = 1 - x, given lx = log x and ly = log y
// computed by the caller without forming log of a value close to 1.
double log_beta_prefactor(double a, double b, double x, double y, double lx,
                          double ly) noexcept {
    const double s = a + b;
    if (std::min(a, b) >= stirling_threshold)
        return -a * phi(x * s / a) - b * phi(y * s / b) + 0.5 * std::log(a / s * b) -
               half_log_two_pi - stirling_delta(a) - stirling_delta(b) + stirling_delta(s);
    const double head = a * lx + b * ly;
    if (b >= stirling_threshold) return head + log_gamma_ratio(b, a) - std::lgamma(a);
    if (a >= stirling_threshold) return head + log_gamma_ratio(a, b) - std::lgamma(b);
    return head + std::lgamma(s) - std::lgamma(a) - std::lgamma(b);
}

// Power series of P(a, x) / prefactor, x < a + 1.
double gamma_series(double a, double x) noexcept {
    double term = 1.0 / a, sum = term;
    for (int k = 1; k < max_terms; ++k) {
        term *= x / (a + k);
        sum += term;
        if (std::abs(term) < std::abs(sum) * eps) break;
    }
    return sum;
}

// Continued fraction of Q(a, x) / prefactor, x >= a + 1.
double gamma_fraction(double a, double x) noexcept {
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
    return h;
}

// Continued fraction of I_x(a, b) without its prefactor, x < (a+1)/(a+b+2).
// Its first denominator 1 - (a+b) x/(a+1) equals (1 - b + (a+b) y)/(a+1)
// with y = 1 - x; the form with the smaller rounding bound is used, which
// keeps x close to 1 (large a, small b) accurate.
double beta_fraction(double a, double b, double x, double y) noexcept {
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    const double via_x = 1.0 - qab * x / qap;
    const double via_y = (1.0 - b + qab * y) / qap;
    double c = 1.0, d = qab * x <= std::abs(1.0 - b) + qab * y ? via_x : via_y;
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

// Both tails of the gamma: lower = P when lower is true, else Q.
double gamma_tail(double a, double x, bool lower) noexcept {
    if (!(a > 0.0) || !(x >= 0.0)) return nan_value;
    if (x == 0.0) return lower ? 0.0 : 1.0;
    if (std::isinf(x)) return lower ? 1.0 : 0.0;
    const double front = std::exp(log_gamma_prefactor(a, x));
    if (x < a + 1.0) {
        const double p = front * gamma_series(a, x);
        return lower ? p : 1.0 - p;
    }
    const double q = front * gamma_fraction(a, x);
    return lower ? 1.0 - q : q;
}

bool valid_df(double df) noexcept { return df >= 0.5 && df <= 0x1p31; }
bool valid_tail(double tail) noexcept { return tail >= 0x1p-64 && tail <= 0.5; }

// Bisection of a monotone tail on t in [lo, hi] until the bracket is a few
// ulps wide: `below(t)` is true when the root lies above t.
template <typename Below>
double bisect(Below below, double lo, double hi) noexcept {
    for (int i = 0; i < 4096; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        if (mid <= lo || mid >= hi) break;
        if (below(mid)) lo = mid; else hi = mid;
    }
    return lo + 0.5 * (hi - lo);
}
}  // namespace

double regularized_gamma_p(double a, double x) noexcept { return gamma_tail(a, x, true); }
double regularized_gamma_q(double a, double x) noexcept { return gamma_tail(a, x, false); }

// I_x(a, b) from x, y = 1 - x and their logarithms.
static double beta_tail(double a, double b, double x, double y, double lx, double ly) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || !(x >= 0.0) || !(y >= 0.0)) return nan_value;
    if (x == 0.0) return 0.0;
    if (y == 0.0) return 1.0;
    const double s = a + b;
    if (x < (a + 1.0) / (s + 2.0))
        return std::exp(log_beta_prefactor(a, b, x, y, lx, ly)) * beta_fraction(a, b, x, y) / a;
    return 1.0 - std::exp(log_beta_prefactor(b, a, y, x, ly, lx)) * beta_fraction(b, a, y, x) / b;
}

double chi2_quantile(double tail, double df, bool upper) noexcept {
    if (!valid_tail(tail) || !valid_df(df)) return nan_value;
    const double a = 0.5 * df;
    // t = log(x): the lower tail P(a, x/2) increases with t, Q decreases.
    const double lo = std::log(1e-300), hi = std::log(1e300);
    auto below = [&](double t) {
        const double x = 0.5 * std::exp(t);
        return upper ? regularized_gamma_q(a, x) > tail : regularized_gamma_p(a, x) < tail;
    };
    if (!below(lo) || below(hi)) return nan_value;
    return std::exp(bisect(below, lo, hi));
}

double f_quantile(double tail, double d1, double d2, bool upper) noexcept {
    if (!valid_tail(tail) || !valid_df(d1) || !valid_df(d2)) return nan_value;
    const double a = 0.5 * d1, b = 0.5 * d2;
    // t = logit(Y) with Y ~ Beta(a, b) and F = (d2/d1) e^t: Y = 1/(1+e^-t),
    // 1 - Y = 1/(1+e^t), and their logarithms -log1p(e^-t), -log1p(e^t),
    // all formed directly from t (b log(1 - Y) never amplifies a rounding).
    auto below = [&](double t) {
        const double y = 1.0 / (1.0 + std::exp(-t)), z = 1.0 / (1.0 + std::exp(t));
        const double ly = -std::log1p(std::exp(-t)), lz = -std::log1p(std::exp(t));
        return upper ? beta_tail(b, a, z, y, lz, ly) > tail : beta_tail(a, b, y, z, ly, lz) < tail;
    };
    const double lo = -700.0, hi = 700.0;
    if (!below(lo) || below(hi)) return nan_value;
    return d2 / d1 * std::exp(bisect(below, lo, hi));
}

}  // namespace n4m::core::distributions
