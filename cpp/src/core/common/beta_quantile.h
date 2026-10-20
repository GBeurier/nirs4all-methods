/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Small-tail Beta quantile shared by the Hotelling T² UCL
 * (core/utilities/hotelling_t2.c) and the T2-PLS variable selection limit
 * (core/t2_selection.cpp).
 *
 * For Y ~ Beta(a, b), a, b >= 1/2, and alpha in (0, 1), the root v > 0 of
 * P(Y > y) = alpha with y = s / (1 + s), s = v / scale, scale > 0. Both
 * tails are evaluated in logarithms from s and ln s = ln v - ln scale, so no
 * beta argument goes through a complement close to 1: the regularised
 * incomplete beta function by its continued fraction (Numerical Recipes 6.4,
 * modified Lentz, NaN instead of a truncated value when it has not converged
 * within the cap) and ln B(a, b) by the Lanczos approximation (g = 7, nine
 * terms, about 1e-15 on Gamma for x >= 0.5) with its large terms cancelled
 * analytically. The small tail is the one inverted (upper for
 * alpha <= 1/2 against ln alpha, lower otherwise against ln(1 - alpha)), by
 * bisection on the normal doubles: the bracket grows or shrinks by factors
 * of two from 1, saturating at DBL_MAX and DBL_MIN, and is bisected with
 * lo + (hi - lo) / 2.
 *
 * Returns N4M_BETA_ROOT_OK with *v the root, N4M_BETA_ROOT_ABOVE_RANGE with
 * *v = DBL_MAX when the root exceeds DBL_MAX, N4M_BETA_ROOT_BELOW_RANGE with
 * *v = DBL_MIN when it is below DBL_MIN, and N4M_BETA_ROOT_TAIL_FAILURE
 * (*v unspecified) when a tail could not be evaluated.
 */
#ifndef N4M_CORE_COMMON_BETA_QUANTILE_H
#define N4M_CORE_COMMON_BETA_QUANTILE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    N4M_BETA_ROOT_OK = 0,
    N4M_BETA_ROOT_ABOVE_RANGE = 1,
    N4M_BETA_ROOT_BELOW_RANGE = 2,
    N4M_BETA_ROOT_TAIL_FAILURE = 3
} n4m_beta_root_status_t;

n4m_beta_root_status_t n4m_util_beta_small_tail_root(double alpha, double a, double b,
                                                     double scale, double* v);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_BETA_QUANTILE_H */
