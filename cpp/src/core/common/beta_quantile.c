/* SPDX-License-Identifier: CECILL-2.1 */
/* Small-tail Beta quantile; see beta_quantile.h for the contract. */

#include "beta_quantile.h"

#include <float.h>
#include <math.h>

/* Lanczos approximation (g = 7, nine terms; relative error about 1e-15 on
 * Gamma for x >= 0.5): Gamma(x) = sqrt(2 pi) t^(x - 0.5) e^-t S(x), with
 * t = x + 6.5. Kept in this split form so that ln B(a, b) cancels its large
 * terms analytically. */
static double lanczos_sum(double x) {
    static const double c[9] = {
        0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
        771.32342877765313,   -176.61502916214059,   12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    };
    double s = c[0];
    for (int i = 1; i < 9; ++i) s += c[i] / (x - 1.0 + (double)i);
    return s;
}

static double log_gamma(double x) {
    const double t = x + 6.5;
    return 0.91893853320467274178 + (x - 0.5) * log(t) - t + log(lanczos_sum(x));
}

/* ln B(a, b), a, b >= 0.5. With b >= a, ln Gamma(b) - ln Gamma(a + b) is
 * -(b - 0.5) log1p(a / (b + 6.5)) - a ln(a + b + 6.5) + a + ln(S(b) / S(a + b)):
 * no difference of two terms of order b ln b, so a large b keeps the
 * relative accuracy of the tails. */
static double log_beta(double a, double b) {
    if (a > b) {
        const double t = a;
        a = b;
        b = t;
    }
    const double tb = b + 6.5;
    return log_gamma(a) - (b - 0.5) * log1p(a / tb) - a * log(a + tb) + a +
           log(lanczos_sum(b) / lanczos_sum(a + b));
}

/* Continued fraction of I_x(a, b) (Numerical Recipes 6.4, modified Lentz).
 * It needs O(sqrt(max(a, b))) terms; NaN when it has not converged within
 * the cap, so a truncated value is never used. */
static double beta_continued_fraction(double a, double b, double x) {
    const int    max_it = 20000;
    const double fpmin  = 1e-300;
    const double eps    = 3.0e-16;
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < fpmin) d = fpmin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= max_it; ++m) {
        const double dm = (double)m;
        const double m2 = 2.0 * dm;
        double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < eps) return h;
    }
    return (double)NAN;
}

/* Logarithms of both tails of Beta(a, b) at y = s / (1 + s), s > 0, given s
 * and ln s: ln y = ln s - log1p(s) and ln(1 - y) = -log1p(s) without forming
 * y or 1 - y. The tail on the side of y is computed directly, the other as
 * log1p(-exp(.)) of it (that side is at least about 0.3). NaN in *log_lower
 * when the continued fraction has not converged. */
static void beta_log_tails(double a, double b, double s, double log_s,
                           double* log_lower, double* log_upper) {
    const double log_complement = -log1p(s);
    const double log_y = log_s + log_complement;
    const double log_front = a * log_y + b * log_complement - log_beta(a, b);
    const double y = s / (1.0 + s);
    if (y < (a + 1.0) / (a + b + 2.0)) {
        *log_lower = log_front + log(beta_continued_fraction(a, b, y) / a);
        *log_upper = log1p(-fmin(exp(*log_lower), 1.0));
    } else {
        *log_upper = log_front + log(beta_continued_fraction(b, a, 1.0 / (1.0 + s)) / b);
        *log_lower = log1p(-fmin(exp(*log_upper), 1.0));
    }
    if (isnan(*log_upper)) *log_lower = (double)NAN;
}

typedef struct {
    double a, b, scale, log_scale, log_target;
    int upper;
} beta_target;

/* +1 when v is below the root, 0 at or above it, -1 when a tail could not
 * be evaluated. */
static int below_root(const beta_target* t, double v) {
    double log_lower, log_upper;
    beta_log_tails(t->a, t->b, v / t->scale, log(v) - t->log_scale, &log_lower, &log_upper);
    if (isnan(log_lower)) return -1;
    return t->upper ? (log_upper > t->log_target) : (log_lower < t->log_target);
}

n4m_beta_root_status_t n4m_util_beta_small_tail_root(double alpha, double a, double b,
                                                     double scale, double* v) {
    beta_target t;
    t.a = a;
    t.b = b;
    t.scale = scale;
    t.log_scale = log(scale);
    t.upper = alpha <= 0.5;
    t.log_target = t.upper ? log(alpha) : log1p(-alpha);
    double lo = 1.0, hi = 1.0;
    int side = below_root(&t, 1.0);
    if (side < 0) return N4M_BETA_ROOT_TAIL_FAILURE;
    if (side == 1) {
        while (side == 1) {
            if (hi == DBL_MAX) {
                *v = DBL_MAX;
                return N4M_BETA_ROOT_ABOVE_RANGE;
            }
            lo = hi;
            hi = hi > DBL_MAX / 2.0 ? DBL_MAX : 2.0 * hi;
            side = below_root(&t, hi);
            if (side < 0) return N4M_BETA_ROOT_TAIL_FAILURE;
        }
    } else {
        while (side == 0) {
            if (lo == DBL_MIN) {
                *v = DBL_MIN;
                return N4M_BETA_ROOT_BELOW_RANGE;
            }
            hi = lo;
            lo = lo / 2.0 < DBL_MIN ? DBL_MIN : lo / 2.0;
            side = below_root(&t, lo);
            if (side < 0) return N4M_BETA_ROOT_TAIL_FAILURE;
        }
    }
    for (int it = 0; it < 200 && hi - lo > 2.0 * DBL_EPSILON * hi; ++it) {
        const double mid = lo + (hi - lo) / 2.0;
        side = below_root(&t, mid);
        if (side < 0) return N4M_BETA_ROOT_TAIL_FAILURE;
        if (side == 1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    *v = lo + (hi - lo) / 2.0;
    return N4M_BETA_ROOT_OK;
}
