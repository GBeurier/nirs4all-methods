// SPDX-License-Identifier: CECILL-2.1
//
// Internal-core test of the Beta upper quantile used by the T2-PLS variable
// selection limit (core/t2_selection.cpp, hidden in libn4m, hence linked into
// n4m_internal_tests). Closed forms at both ends of alpha:
//   Beta(1, b): P(X > x) = (1 - x)^b, x = -expm1(ln(alpha) / b);
//   Beta(a, 1): P(X <= x) = x^a,      x = exp(log1p(-alpha) / a).
// The former bisection of the CDF against 1 - alpha returned ~1 whenever
// 1 - alpha rounded to 1 (alpha = 1e-20, Beta(1, 50): 1 - 10^-0.4 = 0.6019
// expected), and lost the small lower tail for alpha close to 1.

#include <cmath>
#include <cstdio>

#include "core/t2_selection.hpp"

int run_internal_beta_quantile_tests() {
    const double alphas[] = {1e-300, 1e-20, 1e-8, 0.05, 0.5, 0.75,
                             1.0 - 1e-8, 1.0 - std::ldexp(1.0, -40)};
    const double shapes[] = {0.5, 1.0, 1.5, 2.0, 7.5, 50.0, 5000.0};
    int failures = 0;
    for (const double alpha : alphas) {
        for (const double shape : shapes) {
            for (const bool first : {true, false}) {
                const double a = first ? 1.0 : shape;
                const double b = first ? shape : 1.0;
                const double want = first ? -std::expm1(std::log(alpha) / shape)
                                          : std::exp(std::log1p(-alpha) / shape);
                double got = 0.0;
                const bool ok = ::n4m::core::t2_beta_upper_quantile(alpha, a, b, got);
                if (!ok || !(std::fabs(got - want) <= 1e-12 * want)) {
                    std::printf("  FAIL Beta(%g, %g) upper quantile at alpha %.17g: %.17g, "
                                "expected %.17g\n", a, b, alpha, got, want);
                    ++failures;
                }
            }
        }
    }
    std::printf("[internal] T2 beta upper quantile closed forms ... %s\n",
                failures == 0 ? "ok" : "FAIL");
    return failures;
}
